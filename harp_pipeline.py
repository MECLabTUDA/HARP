import os
import cv2
import argparse

from pipeline.parser import *
from pipeline.harp_dataset import HARPDataset
from pipeline.artifact_detection import ArtifactDetector
from pipeline.artifact_segmentation import ArtifactSegment
from pipeline.artifact_restoration import RestorationModel
from pipeline.heatmap import create_heatmap
from pipeline.artifact_ranking import artifact_ranking, artifact_inverting
from core.util import save_tensor_as_img

class HARP:
    def __init__(self, config):
        self.config = config
        self.artifact_detector = ArtifactDetector(config)
        self.artifact_segmentation = ArtifactSegment(config)
        self.restoration_model = RestorationModel(config)
        os.makedirs(config["result_folder_path"], exist_ok=True)
         
    def harp_pipeline(self, image):
        print("Detecting Artifacts: " + image["image_name"])
        artifact_pred = self.artifact_detector.artifact_detection(image["image_path"])
        artifact_pred = artifact_pred[0]["pred_scores"].numpy().item()
        
        if artifact_pred < self.config["anomalib_prediction_threshold"]:
            print("No artifacts detected in: " + image["image_name"] + ". Skipping...")
        
        else: 
            print("Artifact detected. Restoring image: " + image["image_name"])
            #FYI: Noising and denoising cycle
            reconstructed_images = self.restoration_model.noising_and_denoising_map(self.config, image["image_tensor"])
            heatmap = create_heatmap(reconstructed_images, image["image"])
            
            #FYI: Create both SAM and DBSCAN mask
            sam_mask_list = self.artifact_segmentation.create_sam_masks(image["image"])
            dbscan_mask_list = self.artifact_segmentation.create_dbscan_mask(image["image"])
            combined_mask_list = sam_mask_list + dbscan_mask_list
            
            #FYI: Create inverted masks for masks covering more than 50%
            combined_mask_list = artifact_inverting(combined_mask_list)

            #FYI: Calculate mask scores and sort them and rank them
            ranked_mask_list = artifact_ranking(combined_mask_list, heatmap, image["image"])
            ranked_mask_list = ranked_mask_list[:self.config["restoration_top_k"]]

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
            print("Generating restored images: " + image["image_name"])
            inpainted_image_list = self.restoration_model.inpaint_with_jump_sampling(image["image_tensor"], ranked_mask_list, self.config["batch_size"])
            
            #FYI: Doing artifact detection on restored images and ranking them
            inpainted_image_list = self.artifact_detector.inpaint_ranking(inpainted_image_list)
            
            #FYI: Saving the restored image with least score
            inpainted_image_list = sorted(inpainted_image_list, key=lambda x: x["artifact_pred"])
            best_restored_image = inpainted_image_list[0]
            save_tensor_as_img(best_restored_image["restored_image"], os.path.join(self.config["result_folder_path"], image["image_name"]))
            cv2.imwrite(os.path.join(self.config["result_folder_path"], "mask_"+image["image_name"]), best_restored_image["mask_dilated"])
            return best_restored_image["restored_image"], best_restored_image["mask_dilated"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=manage_path, default='./config/config_harp.json', help='JSON file for config')

    args_harp = parser.parse_args()
    config_harp = parse(args_harp)
    dataset = HARPDataset(config_harp["input_folder_path"], config_harp["image_size"])
    harp = HARP(config_harp)
    
    for image in dataset:
            restored_image, restored_mask = harp.harp_pipeline(image)