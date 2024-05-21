import os
import cv2
import shutil
import warnings
from pathlib import Path

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from anomalib.config import get_configurable_parameters
from anomalib.data.inference import InferenceDataset
from anomalib.data.utils import InputNormalizationMethod, get_transforms
from anomalib.models import get_model
from anomalib.utils.callbacks import get_callbacks

class ArtifactDetector:
    def __init__(self, harp_config):
        config_path = harp_config["anomalib_config_path"]
        checkpoint_path = harp_config["anomalib_model_path"]
        device = [harp_config["gpu_id"]]
        self.result_path = harp_config["result_folder_path"]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=Warning)
        
            config = get_configurable_parameters(config_path=Path(config_path))
            config.trainer.resume_from_checkpoint = str(Path(checkpoint_path))
            config.trainer.devices = device
            config.visualization.show_images = False
            config.visualization.mode = "full"
            config.visualization.save_images = False

            # create model and trainer
            self.model = get_model(config)
            callbacks = get_callbacks(config)
            self.trainer = Trainer(callbacks=callbacks, logger=False, **config.trainer)

            # get the transforms
            transform_config = config.dataset.transform_config.eval if "transform_config" in config.dataset.keys() else None
            normalization = InputNormalizationMethod(config.dataset.normalization)
            self.image_size = (config.dataset.image_size[0], config.dataset.image_size[1])
            self.transform = get_transforms(
                config=transform_config, image_size=self.image_size, normalization=normalization)
            

    def artifact_detection(self, input_path):
        dataset = InferenceDataset(
            Path(input_path), image_size=self.image_size, transform=self.transform)
        dataloader = DataLoader(dataset)

        # generate predictions
        predictions = self.trainer.predict(model=self.model, dataloaders=[dataloader])
        return predictions
    

    def inpaint_ranking(self, inpainted_image_list):
        temp_path = os.path.join(self.result_path, "temp")
        os.makedirs(temp_path, exist_ok=True)
        
        try:
            for image in inpainted_image_list:
                inpainted_file_path = os.path.join(temp_path, image["name"]+"_"+str(image["index"])+".png")
                cv2.imwrite(inpainted_file_path, image["restored_image"])
                
            predictions = self.artifact_detection(temp_path)

            for pred in predictions:
                image_name = os.path.basename(pred["image_path"][0])
                name, index = image_name.split(".")[0].split("_")
                pred_score = pred["pred_scores"].numpy().item()
                for image in inpainted_image_list:
                    if image["name"] == name and image["index"] == int(index):
                        image["artifact_pred"] = pred_score

        # except Exception as e:
        #     print("Executing cleanup code before exiting...")
        finally:                
            shutil.rmtree(temp_path)
        return inpainted_image_list