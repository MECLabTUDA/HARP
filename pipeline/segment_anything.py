import cv2  
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import os
from typing import Any, Dict, List
import numpy as np

#TODO: Shift this all to config files
model_type = 'default'
sam_checkpoint = "/local/scratch/sharvien/HARP_Model_Weights/sam_vit_h_4b8939.pth"
device = "cuda"

def write_masks_to_folder(masks: List[Dict[str, Any]]):
    header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"  # noqa
    metadata = [header]
    output = []
    for i, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        output.append({
            'name':'sam', 
            'index':i, 
            'mask':mask.astype(np.uint8) * 255
            })
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
    return output, metadata

def write_masks_to_png(masks: List[Dict[str, Any]], image, mask_path: str, base) -> None:
    #For visualisation
    if len(masks) == 0:
        return
    sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
    img = np.zeros((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 3))
    #For each mask in the list of mask in sorted mask (around 180), apply random colour
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.random.choice(range(1, 256), size=3)
        img[m] = color_mask
    #Applying mask to image
    img = np.uint8(img)
    blend = 0.5
    result = cv2.addWeighted(image, blend, img, 1-blend, 0)
    result = np.where(img==0, image, result)
    cv2.imwrite(os.path.join(mask_path, base + '_mask.png'), img)
    cv2.imwrite(os.path.join(mask_path, base + '_mask_overlay.png'), result)


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
        output, metadata = write_masks_to_folder(masks)

        if mask_path is not None:
            os.makedirs(mask_path, exist_ok=True)
            write_masks_to_png(masks, image, mask_path, base)
            metadata_path = os.path.join(mask_path, "metadata.csv")
            with open(metadata_path, "w") as f:
                f.write("\n".join(metadata))
            for mask, i in enumerate(output):
                filename = f"{i}.png"
                cv2.imwrite(os.path.join(mask_path, filename), mask)   
    return output