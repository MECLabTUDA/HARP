#Takes in the an image, find out all of its segmentation. It will find the metric values of each mask and input them in metadata.csv
#It will then save the masks all separately as binary in another folder
#It will also then put together all the masks in one image with different colour. Also saves another images wehre the coloured masks are overlayed on top of the  original image

import cv2  
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import random
import os
from typing import Any, Dict, List
import numpy as np

#TODO: Shift this all to config files
model_type = 'default'
#TODO: Change to get from config
sam_checkpoint = "/local/scratch/sharvien/HARP_Model_Weights/sam_vit_h_4b8939.pth"
device = "cuda"

def write_masks_to_folder(masks: List[Dict[str, Any]], path: str) -> None:
    header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"  # noqa
    metadata = [header]
    for i, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        filename = f"{i}.png"
        cv2.imwrite(os.path.join(path, filename), mask * 255)
        mask_metadata = [
            str(i),
            str(mask_data["area"]),
            *[str(x) for x in mask_data["bbox"]],
            *[str(x) for x in mask_data["point_coords"][0]],
            str(mask_data["predicted_iou"]),
            str(mask_data["stability_score"]),
            *[str(x) for x in mask_data["crop_box"]],
        ]
        row = ",".join(mask_metadata)
        metadata.append(row)
    metadata_path = os.path.join(path, "metadata.csv")
    with open(metadata_path, "w") as f:
        f.write("\n".join(metadata))
    #TODO: Fix all the empty returns
    return

def write_masks_to_png(masks: List[Dict[str, Any]], image, mask_path: str, base) -> None:
    if len(masks) == 0:
        return

    sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
    
    #Initialising new img with 600x600x4 where [1,1,1,0]
    img = np.zeros((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 3))
    
    #For each mask in the list of mask in sorted mask (around 180), apply random colour
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.random.choice(range(1, 256), size=3)
        img[m] = color_mask
    
    # Applying mask to image
    img = np.uint8(img)
    blend = 0.5
    result = cv2.addWeighted(image, blend, img, 1-blend, 0)
    result = np.where(img==0, image, result)
    cv2.imwrite(os.path.join(mask_path, base + '_mask.png'), img)
    cv2.imwrite(os.path.join(mask_path, base + '_mask_overlay.png'), result)
    return

def create_sam_masks(input, mask_path=None):
    sam = sam_model_registry[model_type](sam_checkpoint)
    _ = sam.to(device)
    generator = SamAutomaticMaskGenerator(sam, output_mode="binary_mask")
    if not os.path.isdir(input):
        targets = [input]
    else:
        targets = [f for f in os.listdir(input) if not os.path.isdir(os.path.join(input, f))]
        targets = [os.path.join(input, f) for f in targets]

    for t in targets:
        print(f"Processing '{t}'...")
        image = cv2.imread(t)
        if image is None:
            print(f"Could not load '{t}' as an image, skipping...")
            continue
        
        base = os.path.basename(t)
        base = os.path.splitext(base)[0]
        masks = generator.generate(image)
   
        if mask_path is not None:
            os.makedirs(mask_path, exist_ok=True)
            write_masks_to_folder(masks, mask_path)
            write_masks_to_png(masks, image, mask_path, base)
    return(masks)

# input = "/gris/gris-f/homestud/ssivakum/denoising_diffusion/logs/images_documentation/heatmap/thread/Thread.png"
# mask_path = "/gris/gris-f/homestud/ssivakum/denoising_diffusion/logs/images_documentation/heatmap/thread/masks/sam_mask"
# create_sam_masks()

def combine_sam_mask_with_coverage(img_path: str, coverage: int):
    #TODO: High priority - config file for the image area!!
    #TODO: The np.zero and image area need to variable
    #Intentionally not being too strict with the area coverage
    image_area = 600 * 600
    threshold = coverage/100 * image_area
    bi_combined = np.zeros((600,600))
    
    masks = create_sam_masks(img_path)
    randomized_index = random.sample(range(len(masks)), len(masks))
    i = 0; mask_area = 0
    
    while mask_area < threshold and i < len(masks)+1:
        binary_mask = masks[randomized_index[i]]["segmentation"]
        bi_combined = np.where(binary_mask==0, bi_combined, 0)
        result = np.where(binary_mask==0, binary_mask, 255)
        bi_combined += result
        i += 1
        mask_area += int(masks[randomized_index[i]]["area"])

    return bi_combined