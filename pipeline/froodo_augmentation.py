from copy import deepcopy
import sys
from torchvision import utils as ut

sys.path.append('/gris/gris-f/homestud/ssivakum/FrOoDo/')
from froodo.quickstart import *


def get_artifacted_image(dataset_index):
    #FYI: Just a simple given dataset_index, artifact_types list -> return original image and default 5 artifacted images
    #FYI: ["original", "dark_spot", "squamos", "thread", "blood_group", "blood_cells", "compression", "cut", "air_bubble", "overlap", "folding", "slice_thickness"]
    dataset = BCSS_Adapted_Cropped_Resized_Datasets().test
    sample = dataset[dataset_index]

    augmentations = [
        [Nothing()],
        [SampledAugmentation(DarkSpotsAugmentation(sample_intervals=[(0.1, 5)], keep_ignorred=False))],
        # [SampledAugmentation(FatAugmentation(sample_intervals=[(0.1, 15)]))],
        [SampledAugmentation(SquamousAugmentation(sample_intervals=[(0.1, 6)], keep_ignorred=False))],
        [SampledAugmentation(ThreadAugmentation(sample_intervals=[(0.1, 11)], keep_ignorred=False))],
        [SampledAugmentation(BloodCellAugmentation(sample_intervals=[(1, 125)],scale_sample_intervals=[(1.0, 1.5)], keep_ignorred=False))],
        [SampledAugmentation(BloodGroupAugmentation(sample_intervals=[(0.5, 4)], num_groups=[(1,8)], keep_ignorred=False))],
        # [PartialOODAugmentaion(GaussianBlurAugmentation(sigma=10),)],
        # [PartialOODAugmentaion(BrightnessAugmentation(1.7))],
        # [PartialOODAugmentaion(ContrastAugmentation(2.5))],
        [SampledAugmentation(DeformationAugmentation(mode="squeeze",sample_intervals={"grid_points": [(1,25)], "grad_intensity": [[(1,30)]]}, keep_ignorred=False),1)],
        [SampledAugmentation(DeformationAugmentation(mode="stretch",sample_intervals={"grid_points": [(2,6)], "grad_intensity":  [[(1,6)]]}, keep_ignorred=False),1)],
        [SampledAugmentation(BubbleAugmentation(sample_intervals=[(1,1.1)], keep_ignorred=False))],
        [SampledAugmentation(OverlapAugmentation(sample_intervals=[(1,150)], keep_ignorred=False),1)],
        [SampledAugmentation(FoldingAugmentation(sample_intervals=[(1,150)], keep_ignorred=False),1)],
        #[SampledAugmentation(SliceThicknessAugmentation(keep_ignorred=False),1)]
                    ]

    images = []
    ood_mask = []
    segmentation_mask = []

    for i in range(len(augmentations)):
        s = deepcopy(sample)
        s = augmentations[i][0](s)
        images.append(s.image)
        ood_mask.append(s['ood_mask'])
    
    segmentation_mask = s['segmentation_mask']
    return images, ood_mask, segmentation_mask

artifacted_images, ood_mask, segmentation_mask = get_artifacted_image(2)
ut.save_image(artifacted_images[0], "/gris/gris-f/homestud/ssivakum/denoising_diffusion/logs/temp/temp4.png")

