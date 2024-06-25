import os
import cv2
import argparse
import numpy as np

from harp.pipeline.parser import *
from harp.pipeline.harp_dataset import HARPDataset
from harp.pipeline.artifact_detection import ArtifactDetector
from harp.pipeline.artifact_segmentation import ArtifactSegment
from harp.pipeline.artifact_restoration import RestorationModel
from harp.pipeline.heatmap import create_heatmap
from harp.pipeline.artifact_ranking import artifact_ranking, artifact_inverting

class HARP:
    def __init__(self, config_path):
        config = parse(config_path)
        self.config = config
        self.artifact_detector = ArtifactDetector(config)
        self.artifact_segmentation = ArtifactSegment(config)
        self.restoration_model = RestorationModel(config)
        os.makedirs(config["result_folder_path"], exist_ok=True)

    def save_image(self, config, best_mask_restored_image):
        cv2.imwrite(os.path.join(config["result_folder_path"], image["image_name"]), best_mask_restored_image["restored_image"])
        cv2.imwrite(os.path.join(config["result_folder_path"], "mask_"+image["image_name"]), best_mask_restored_image["mask_dilated"])

    def process_masks(self, mask_restored_image_list):
        kernel_size = 10
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        for mask in mask_restored_image_list:
            #FYI: Need to increase mask size slightly to avoid the seeing artifact boundary after inpainting
            dilated_mask = cv2.dilate(mask["mask"], kernel, iterations=1)
            #FYI: For DBSCAN mask do second higher dilations
            if mask["name"] == "dbscan":
                dilated_mask = cv2.dilate(dilated_mask, kernel, iterations=2)
            mask["mask_dilated"] = dilated_mask
        return mask_restored_image_list


    def harp_pipeline(self, image):
        print("Detecting Artifacts: " + image["image_name"])
        artifact_pred = self.artifact_detector.artifact_detection(image["image_path"])
        artifact_pred = artifact_pred[0]["pred_scores"].numpy().item()
        
        if artifact_pred < self.config["anomalib_prediction_threshold"]:
            #FYI: If no artifact detected, will return same image with empty mask
            print("No artifacts detected in: " + image["image_name"] + ". Skipping...")
            return image["image"], np.zeros_like(image["image"][0])
        
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
            mask_restored_image_list = artifact_ranking(combined_mask_list, heatmap, image["image"])
            mask_restored_image_list = mask_restored_image_list[:self.config["restoration_top_k"]]

            #FYI: Process masks for inpainting
            mask_restored_image_list = self.process_masks(mask_restored_image_list)

            #FYI: Doing inpainting here
            print("Generating restored images: " + image["image_name"])
            mask_restored_image_list = self.restoration_model.inpaint_with_jump_sampling(image["image_tensor"], mask_restored_image_list, self.config["batch_size"])
            
            #FYI: Doing artifact detection on restored images and ranking them
            mask_restored_image_list = self.artifact_detector.inpaint_ranking(mask_restored_image_list)
            
            #FYI: Saving the restored image with least score
            mask_restored_image_list = sorted(mask_restored_image_list, key=lambda x: x["artifact_pred"])
            best_mask_restored_image = mask_restored_image_list[0]
            if self.config["save_images"]:
                self.save_image(self.config, best_mask_restored_image)

            return best_mask_restored_image["restored_image"], best_mask_restored_image["mask_dilated"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=manage_path, default='./config/config_harp.json', help='JSON file for config')
    args_harp = parser.parse_args()

    harp = HARP(args_harp.config)
    dataset = HARPDataset(harp.config["input_folder_path"], harp.config["image_size"])

    for image in dataset:
            restored_image, restored_mask = harp.harp_pipeline(image)