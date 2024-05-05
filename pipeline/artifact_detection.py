import os
import shutil

from argparse import Namespace
from pathlib import Path

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from core.util import save_tensor_as_img

from anomalib.config import get_configurable_parameters
from anomalib.data.inference import InferenceDataset
from anomalib.data.utils import InputNormalizationMethod, get_transforms
from anomalib.models import get_model
from anomalib.utils.callbacks import get_callbacks

def artifact_detection(config_path, checkpoint_path, input_path, output_path=None):

    args = Namespace(
    config=Path(config_path),
    weights=Path(checkpoint_path),
    input=Path(input_path),
    output=output_path,
    visualization_mode="full",
    show=False)
    
    config = get_configurable_parameters(config_path=args.config)
    config.trainer.resume_from_checkpoint = str(args.weights)
    config.visualization.show_images = args.show
    config.visualization.mode = args.visualization_mode
    if args.output:  # overwrite save path
        config.visualization.save_images = True
        config.visualization.image_save_path = args.output
    else:
        config.visualization.save_images = False

    # create model and trainer
    model = get_model(config)
    callbacks = get_callbacks(config)
    trainer = Trainer(callbacks=callbacks, **config.trainer)

    # get the transforms
    transform_config = config.dataset.transform_config.eval if "transform_config" in config.dataset.keys() else None
    image_size = (config.dataset.image_size[0], config.dataset.image_size[1])
    normalization = InputNormalizationMethod(config.dataset.normalization)
    transform = get_transforms(
        config=transform_config, image_size=image_size, normalization=normalization)

    # create the dataset
    dataset = InferenceDataset(
        args.input, image_size=tuple(config.dataset.image_size), transform=transform)
    dataloader = DataLoader(dataset)

    # generate predictions
    predictions = trainer.predict(model=model, dataloaders=[dataloader])
    return predictions

def inpaint_ranking(config_path, checkpoint_path, result_path, inpainted_image_list):
    temp_path = os.path.join(result_path, "temp")
    os.makedirs(temp_path, exist_ok=True)
    for image in inpainted_image_list:
        inpainted_file_path = os.path.join(temp_path, image["name"]+"_"+str(image["index"])+".png")
        save_tensor_as_img(image["restored_image"], inpainted_file_path)
        
    predictions = artifact_detection(config_path, checkpoint_path, temp_path)

    for pred in predictions:
        image_name = os.path.basename(pred["image_path"][0])
        name, index = image_name.split(".")[0].split("_")
        pred_score = pred["pred_scores"].numpy().item()
        for image in inpainted_image_list:
            if image["name"] == name and image["index"] == int(index):
                image["artifact_pred"] = pred_score
                
    shutil.rmtree(temp_path)
    return inpainted_image_list