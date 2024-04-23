from argparse import Namespace
from pathlib import Path

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from anomalib.config import get_configurable_parameters
from anomalib.data.inference import InferenceDataset
from anomalib.data.utils import InputNormalizationMethod, get_transforms
from anomalib.models import get_model
from anomalib.utils.callbacks import get_callbacks


def artifact_detection(config_path, checkpoint_path, input_path):

    args = Namespace(
    config=Path(config_path),
    weights=Path(checkpoint_path),
    input=Path(input_path),
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

def inpaint_ranking(config_path, checkpoint_path, inpainted_image_list):
    inpaint_ranking = []
    predictions = artifact_detection(config_path, checkpoint_path, inpaint_path)

    for pred in predictions:
        image_path = pred["image_path"][0]
        pred_score = pred["pred_scores"].numpy().item()
        inpaint_ranking.append([image_path, pred_score])

    inpaint_ranking = sorted(inpaint_ranking, key=lambda x: x[1])
    return inpaint_ranking

# input_path = "/gris/gris-f/homestud/ssivakum/img2img/result/11/artifacts/air_bubble/methods/repaint/repaint"
# output_path = "/gris/gris-f/homestud/ssivakum/img2img/result/anomalib/output"
# config_path = "/gris/gris-f/homestud/ssivakum/img2img/config/config_fastflow.yaml"
# checkpoint_path = "/local/scratch/BCSS_Weights_MA_MiSc/Anomalib_Weights/fastflow.ckpt"
# inpaint_ranking(config_path, checkpoint_path, input_path)
