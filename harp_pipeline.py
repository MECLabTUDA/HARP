import os
import cv2

from core.util import save_tensor_as_img
from pipeline.parser import parse
from pipeline.heatmap import create_heatmap
from pipeline.segment_anything import create_sam_masks
from pipeline.dbscan_colored import create_dbscan_mask
from pipeline.artifact_ranking import artifact_ranking, artifact_inverting
from pipeline.artifact_detection import *
from pipeline.inpainting import *
from pipeline.metric_scores import *

def harp_pipeline(config, image_name):
    image_path = os.path.join(config["image_folder_path"], image_name)
    result_path = os.path.join(config["result_folder_path"], image_name)
    os.makedirs(result_path, exist_ok=True)
    
    print("Detecting Artifacts: " + image_name)
    artifact_pred = artifact_detection(config["anomalib_config_path"], config["anomalib_checkpoint_path"], image_path)
    artifact_pred = artifact_pred[0]["pred_scores"].numpy().item()
    if artifact_pred < config["anomalib_prediction_threshold"]:
        print("No artifacts detected.")
    else: 
        print("Artifact detected. Restoring...")

        #TODO: High prio (For each image do resizing individually)
        os.makedirs(result_path, exist_ok=True)

        #FYI: Noising and denoising cycle
        noised_denoised_images = noising_and_denoising_map(image_path, config["n_noising_step"], config["n_iter"], config["batch_size"])
        heatmap = create_heatmap(noised_denoised_images, image_path)
        
        #FYI: Create both SAM and DBSCAN mask
        sam_mask_list = create_sam_masks(image_path)
        dbscan_mask_list = create_dbscan_mask(image_path)
        combined_mask_list = sam_mask_list + dbscan_mask_list
        
        #FYI: Create inverted masks for masks covering more than 50%
        combined_mask_list = artifact_inverting(combined_mask_list)

        #FYI: Calculate mask scores and sort them and rank them
        ranked_mask_list = artifact_ranking(combined_mask_list, heatmap, image_path)
        ranked_mask_list = ranked_mask_list[:config["top_k"]]

        #FYI: Do inpainting on the top k masks
        inpainting_mask_path = []
        for mask in ranked_mask_list:
            #FYI: Need to increase mask size slightly to avoid the seeing artifact boundary after inpainting
            #FYI: Only saving the top k mask with slightly enlarged ones
            kernel_size = 10
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            dilated_mask = cv2.dilate(mask["mask"], kernel, iterations=1)

            #FYI: For DBSCAN mask do second higher dilations
            if mask['name'] == "dbscan":
                dilated_mask = cv2.dilate(dilated_mask, kernel, iterations=2)
            mask['mask_dilated'] = dilated_mask

        #FYI: Doing inpainting here
        print("Generating inpaint images: " + image_name)
        #TODO: The partial mask below is just doing normal inpaint without any partial mask. So change it
        inpainted_image_list = inpaint_with_jump_sampling(image_path, ranked_mask_list, config["batch_size"])
        
        #FYI: Doing artifact detection on restored images
        detection_score = inpaint_ranking(config["anomalib_config_path"], config["anomalib_checkpoint_path"], inpainted_image_list)
        
        #FYI: Save the final restored image elsewhere (Hacky)
        save_tensor_as_img(inpainted_image_list[it], inpainted_file_path)
        cv2.imwrite(os.path.join(config["result_folder_path"], "restored_"+image_name), cv2.imread(detection_score[0][0]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='/gris/gris-f/homestud/ssivakum/img2img/config/harp_bcss.json', help='JSON file for configuration')
    args_harp = parser.parse_args()
    opt_harp = parse(args_harp)
    
    #TODO: Only include images and exclude other files from list
    #TODO: This is duplicating from above
    image_list = os.listdir(opt_harp["image_folder_path"])

    for image in image_list:
        image_name = os.path.basename(image)
        harp_pipeline(opt_harp, image_name)
