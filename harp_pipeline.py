import os
import cv2
import argparse

from core.util import save_tensor_as_img
from pipeline.parser import *
from pipeline.heatmap import create_heatmap
from pipeline.segment_anything import create_sam_masks
from pipeline.dbscan_colored import create_dbscan_mask
from pipeline.artifact_ranking import artifact_ranking, artifact_inverting
from pipeline.artifact_detection import *
from pipeline.inpainting import *

def harp_pipeline(config, image_name):
    image_path = os.path.join(config["image_folder_path"], image_name)
    result_path = config["result_folder_path"]
    os.makedirs(result_path, exist_ok=True)
    
    print("Detecting Artifacts: " + image_name)
    artifact_pred = artifact_detection(config["anomalib_config_path"], config["anomalib_model_path"], image_path)
    artifact_pred = artifact_pred[0]["pred_scores"].numpy().item()
    
    if artifact_pred < config["anomalib_prediction_threshold"]:
        print("No artifacts detected.")
    
    else: 
        print("Artifact detected. Restoring...")
        #FYI: Noising and denoising cycle
        noised_denoised_images = restoration_model.noising_and_denoising_map(image_path, config["n_noising_step"], config["n_iter"], config["batch_size"])
        heatmap = create_heatmap(noised_denoised_images, image_path)
        
        #FYI: Create both SAM and DBSCAN mask
        sam_mask_list = create_sam_masks(image_path, config)
        dbscan_mask_list = create_dbscan_mask(image_path)
        combined_mask_list = sam_mask_list + dbscan_mask_list
        
        #FYI: Create inverted masks for masks covering more than 50%
        combined_mask_list = artifact_inverting(combined_mask_list)

        #FYI: Calculate mask scores and sort them and rank them
        ranked_mask_list = artifact_ranking(combined_mask_list, heatmap, image_path)
        ranked_mask_list = ranked_mask_list[:config["restoration_top_k"]]

        #FYI: Do inpainting on the top k masks
        for mask in ranked_mask_list:
            #FYI: Need to increase mask size slightly to avoid the seeing artifact boundary after inpainting
            #FYI: Only saving the top k mask with slightly enlarged ones
            kernel_size = 10
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            dilated_mask = cv2.dilate(mask["mask"], kernel, iterations=1)

            #FYI: For DBSCAN mask do second higher dilations
            if mask["name"] == "dbscan":
                dilated_mask = cv2.dilate(dilated_mask, kernel, iterations=2)
            mask["mask_dilated"] = dilated_mask

        #FYI: Doing inpainting here
        print("Generating inpaint images: " + image_name)
        inpainted_image_list = restoration_model.inpaint_with_jump_sampling(image_path, ranked_mask_list, config["batch_size"])
        
        #FYI: Doing artifact detection on restored images and ranking them
        inpainted_image_list = inpaint_ranking(config["anomalib_config_path"], config["anomalib_model_path"], config["result_folder_path"], inpainted_image_list)
        
        #FYI: Saving the restored image with least score
        inpainted_image_list = sorted(inpainted_image_list, key=lambda x: x["artifact_pred"])
        best_restored_image = inpainted_image_list[0]
        save_tensor_as_img(best_restored_image["restored_image"], os.path.join(result_path, image_name))
        cv2.imwrite(os.path.join(result_path, "mask_"+image_name), best_restored_image["mask_dilated"])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=manage_path, default='./config/config_harp.json', help='JSON file for config')
    args_harp = parser.parse_args()
    config_harp = parse(args_harp)

    restoration_model = ModelLoader(config_harp)
    
    image_list = os.listdir(config_harp["image_folder_path"])

    for image in image_list:
        image_name = os.path.basename(image)
        harp_pipeline(config_harp, image_name)
