import os, sys
import cv2
import json
import numpy as np
import torch
from torchvision import utils as ut
import matplotlib.pyplot as plt
import tifffile as tiff
from PIL import Image

from core.util import save_tensor_as_img
from pipeline.heatmap import create_heatmap
from pipeline.segment_anything import create_sam_masks
from pipeline.dbscan_colored import create_dbscan_mask
from pipeline.artifact_ranking import artifact_ranking, artifact_inverting
from pipeline.artifact_detection import *
from pipeline.froodo_augmentation import get_artifacted_image
from pipeline.inpainting import *
from pipeline.metric_scores import *

def load_config(file_path):
    json_str = ""
    with open(file_path, 'r') as file:
        for line in file:
            line = line.split('//')[0] + '\n'
            json_str += line
    config = json.loads(json_str)
    return config

config_file_path = '/gris/gris-f/homestud/ssivakum/img2img/config/harp_bcss.json'
config = load_config(config_file_path)

image_folder_path = config["pipeline"]["image_folder_path"]
result_folder_path = config["pipeline"]["result_folder_path"]
batch_size = config["pipeline"]["batch_size"]
image_size = config["pipeline"]["image_size"]
anomalib_config_path = config["pipeline"]["anomalib"]["config_path"]
anomalib_checkpoint_path = config["pipeline"]["anomalib"]["checkpoint_path"]
anomalib_prediction_threshold = config["pipeline"]["anomalib"]["prediction_threshold"]

def harp_pipeline(image_name):
    image_path = os.path.join(image_folder_path, image_name)
    result_path = os.path.join(result_folder_path, image_name)
    
    print("Detecting Artifacts: " + image_name)
    artifact_pred = artifact_detection(anomalib_config_path, anomalib_checkpoint_path, image_path)
    artifact_pred = artifact_pred[0]["pred_scores"].numpy().item()
    if artifact_pred < anomalib_prediction_threshold:
        print("No artifacts detected.")

    else: 
        print("Artifact detected. Restoring...")

        #TODO: High prio (For each image do resizing individually)
        #TODO: Don't save image and use its path
        os.makedirs(result_path, exist_ok=True)

        #FYI: Create noised_denoised_images path and add 20-50 images here
        denoised_image_path = os.path.join(result_path, "noised_denoised_images")
        os.makedirs(denoised_image_path, exist_ok=True)
        n_noising_step = config["pipeline"]["heatmap"]["n_noising_step"]
        n_iter = config["pipeline"]["heatmap"]["n_iter"]
        noising_and_denoising_map(image_path, denoised_image_path, n_noising_step, n_iter, batch_size)

        #FYI: Create heatmaps
        heatmaps_path = os.path.join(result_path, "heatmaps")
        os.makedirs(heatmaps_path, exist_ok=True)
        create_heatmap(heatmaps_path, denoised_image_path, image_path)
        
        #FYI: Create both SAM and DBSCAN mask
        masks_path = os.path.join(result_path, "masks")
        sam_mask_path = os.path.join(masks_path, "sam_mask")
        dbscan_mask_path = os.path.join(masks_path, "dbscan_mask")
        os.makedirs(masks_path, exist_ok=True)
        os.makedirs(sam_mask_path, exist_ok=True)
        os.makedirs(dbscan_mask_path, exist_ok=True)
        create_sam_masks(image_path, sam_mask_path)
        create_dbscan_mask(image_path, dbscan_mask_path)

        #FYI: Create inverted masks for masks covering more than 50%
        artifact_inverting(masks_path)

        #FYI: Calculate mask scores and sort them and rank them
        ranked_mask_path = os.path.join(masks_path, "ranked")
        os.makedirs(ranked_mask_path, exist_ok=True)
        artifact_ranking(masks_path, heatmaps_path+"/5_binary.png", image_path, ranked_mask_path)
        
        #FYI: Get the name of first five ranked masks 
        ranked_mask_name_list = os.listdir(ranked_mask_path)
        ranked_mask_name_list_filtered = []

        #FYI: Needed to removed enlarged_masks from list
        for j in ranked_mask_name_list:
            if j.split('_')[0].isdigit():
                ranked_mask_name_list_filtered.append(j)
        ranked_mask_name_list_filtered = sorted(ranked_mask_name_list_filtered, key=lambda x: int(x.split('_')[0]))[:5]
        
        #FYI: Do inpainting on the top 5 masks
        inpaint_path = os.path.join(result_path, "inpaint_repaint")
        utils_path = os.path.join(result_path, "utils")
        os.makedirs(inpaint_path, exist_ok=True)
        os.makedirs(utils_path, exist_ok=True)

        inpainting_mask_path = []
        for mask in ranked_mask_name_list_filtered:
            #FYI: Need to increase mask size slightly to avoid the seeing artifact boundary after inpainting
            #FYI: Only saving the top 5 mask with slightly enlarged ones  
            mask_full_path = os.path.join(ranked_mask_path, mask)
            binary_mask_image = cv2.imread(mask_full_path)
            kernel_size = 10
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            enlarged_mask = cv2.dilate(binary_mask_image, kernel, iterations=1)

            #FYI: For DBSCAN mask do second higher dilations
            if "dbscan" in mask:
                enlarged_mask = cv2.dilate(enlarged_mask, kernel, iterations=2)
            
            #TODO: Double check the numbers
            num_white_pixels = np.sum(enlarged_mask == 255) / 3
            area_mask = num_white_pixels / (image_size[0]*image_size[1])

            enlarged_mask_full_path = os.path.join(ranked_mask_path, "enlarged_"+mask)
            cv2.imwrite(enlarged_mask_full_path, enlarged_mask)
            inpainting_mask_path.append(enlarged_mask_full_path)

        #FYI: Doing inpainting here
        print("Generating inpaint images: " + image_name)
        #TODO: The partial mask below is just doing normal inpaint without any partial mask. So change it
        inpainted_image_list = inpaint_with_jump_sampling(image_path, inpainting_mask_path, batch_size)

        metric_file_data = ["Score,MSE,PSNR,SSIM"]
        mse_values, psnr_values, ssim_values = [], [], []
        artifact_image = cv2.imread(image_path)
        
        for it in range(len(inpainted_image_list)):
            mask_name = ranked_mask_name_list_filtered[it]
            inpainted_file_path = os.path.join(inpaint_path, mask_name)
            save_tensor_as_img(inpainted_image_list[it], inpainted_file_path)
            
            #TODO: Kinda dumb to do save and read again
            inpainted_image = cv2.imread(inpainted_file_path)
            mse_val = mse_score(artifact_image, inpainted_image) / area_mask
            mse_values.append(mse_val)
            psnr_val = psnr_score(artifact_image, inpainted_image) / area_mask
            psnr_values.append(psnr_val)
            ssim_val = ssim_score(artifact_image, inpainted_image) / area_mask
            ssim_values.append(ssim_val)
            metric_file_data.append(f'{mask_name},{mse_val},{psnr_val},{ssim_val}')

        with open(f'{utils_path}/metrics_scores.txt', "w") as f:
            f.write("\n".join(metric_file_data))

        detection_score = inpaint_ranking(anomalib_config_path, anomalib_checkpoint_path, inpaint_path)
        #FYI: Save the final restored image elsewhere (Hacky)
        cv2.imwrite(os.path.join(result_folder_path, "restored_"+image_name), cv2.imread(detection_score[0][0]))
         
        sorted_inpainting_mask = [os.path.basename(sub_list[0]) for sub_list in detection_score]
        with open(f'{utils_path}/detection_score.txt', "w") as f:
            for sub_list in detection_score:
                line = '\t'.join(map(str, sub_list))
                f.write(f"{line}\n")

        #FYI: This is done separately to remove dependecy 
        #FYI: Just for metric visualisation (Bar charts for metrics)
        bar_positions = np.arange(len(sorted_inpainting_mask))
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

        hbars1 = ax1.barh(bar_positions, mse_values, align='center')
        ax1.bar_label(hbars1, fmt=lambda x: round(x, 4))
        ax1.set_yticks(bar_positions, labels=sorted_inpainting_mask)
        ax1.invert_yaxis()
        ax1.set_xlim(int(min(mse_values)-5), int(max(mse_values)+5))
        ax1.set_xlabel('Inpainted with top five masks')
        ax1.set_title('MSE/Mask Area Comparison')

        hbars2 = ax2.barh(bar_positions, psnr_values, align='center')
        ax2.bar_label(hbars2, fmt=lambda x: round(x, 4))
        ax2.set_yticks([])
        ax2.invert_yaxis()
        ax2.set_xlabel('Inpainted with top five masks')
        ax2.set_title('PSNR/Mask Area Comparison')

        hbars3 = ax3.barh(bar_positions, ssim_values, align='center')
        ax3.bar_label(hbars3, fmt=lambda x: round(x, 4))
        ax3.set_yticks([])
        ax3.invert_yaxis()
        ax3.set_xlabel('Inpainted with top five masks')
        ax3.set_title('SSIM/Mask Area Comparison')

        plt.tight_layout()
        plt.show()
        plt.savefig(os.path.join(utils_path, "metrics_graphs.png"))
        plt.close()

        #FYI: Just for visualisation (Compiling images and masks into 1 graph)
        titles = sorted_inpainting_mask
        f, ax = plt.subplots(2, 6, figsize=(30, 5*2))
        ax[0,0].set_ylabel("Masks", fontsize=35)
        
        ax[0,0].set_title("Heatmap", fontsize=35)
        heatmap_image = cv2.cvtColor(cv2.imread(os.path.join(heatmaps_path,"4_heatmap_average_threshold.png")), cv2.COLOR_BGR2RGB)
        ax[0,0].imshow(heatmap_image)

        ax[1,0].set_ylabel("Inpainting", fontsize=35)
        ax[1,0].imshow(cv2.cvtColor(artifact_image, cv2.COLOR_BGR2RGB))

        for k in range(len(titles)): 
            mask_image = cv2.cvtColor(cv2.imread(os.path.join(ranked_mask_path, titles[k])), cv2.COLOR_BGR2RGB)
            ax[0,k+1].set_title(titles[k], fontsize=22)
            ax[0,k+1].imshow(mask_image)
            inpainted_image = cv2.cvtColor(cv2.imread(os.path.join(inpaint_path, titles[k])), cv2.COLOR_BGR2RGB)
            ax[1,k+1].imshow(inpainted_image)

            ax[0, k].set_yticks([])
            ax[0, k].set_xticks([])
            ax[1, k].set_yticks([])
            ax[1, k].set_xticks([])
        ax[0, k+1].set_yticks([])
        ax[0, k+1].set_xticks([])
        ax[1, k+1].set_yticks([])
        ax[1, k+1].set_xticks([])

        plt.show()
        plt.savefig(os.path.join(utils_path, "metrics_compiled_images.png"), dpi=600)
        plt.close()

#TODO: Only include images and exclude other files from list
image_list = os.listdir(image_folder_path)

for image in image_list:
    image_name = os.path.basename(image)
    harp_pipeline(image_name)
