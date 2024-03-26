#This will take root mask file, find all the binary mask file, and arrange the binary file based on a score
#Score would be mix of mask density (using heatmap) and area (normalised)
import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def artifact_inverting(parent_directory):
    #FYI: Go thorugh the files of mask and generated iverted masks for mask above 50%
    for root, _, files in os.walk(parent_directory):
        for filename in files:
            if os.path.splitext(filename)[0].isdigit():
                binary_mask_image = cv2.imread(os.path.join(root, filename), cv2.IMREAD_GRAYSCALE)
                num_white_pixels = np.sum(binary_mask_image == 255)
                height, width = binary_mask_image.shape
                total_pixels = height * width
                area_mask = num_white_pixels / total_pixels

                if area_mask * 100 > 49:
                    print(root)
                    inverted_binary_mask_image = np.where(binary_mask_image == 0, 255, 0)
                    #FYI: For now inverted are just saved as this
                    cv2.imwrite(os.path.join(root, "1000"+filename), inverted_binary_mask_image)

def check_similarity(mask, processed_masks):
    similarity = 0
    for processed_mask in processed_masks:
        union_mask = np.logical_or(mask, processed_mask)
        and_mask = np.logical_and(mask, processed_mask)
        dice = np.count_nonzero(and_mask) / np.count_nonzero(union_mask)
        if dice > similarity:
            similarity = dice
    processed_masks.append(mask)
    return processed_masks, similarity  

def artifact_ranking(parent_directory, heat_map_path, artifact_image_path, output_folder_path):
    score_dict = {}
    processed_masks = []
    #FYI: Searching through all files with just numbers in the parent directory to retrieve all binary masks
    for root, _, files in os.walk(parent_directory):
        for filename in files:
            if os.path.splitext(filename)[0].isdigit():
                subdirectory = os.path.relpath(root, start=parent_directory)

                #FYI: Calculate number of pixel on mask and the area of the white mask
                binary_mask_image = cv2.imread(os.path.join(root, filename), cv2.IMREAD_GRAYSCALE)
                num_white_pixels = np.sum(binary_mask_image == 255)
                height, width = binary_mask_image.shape
                total_pixels = height * width
                area_mask = num_white_pixels / total_pixels
                
                #FYI: Reject masks that are simialr than 97%
                processed_masks, similarity = check_similarity(binary_mask_image, processed_masks)
                if similarity > 0.97:
                    continue

                #FYI: Rejects masks that are smaller than 0.4 percent or bigger than 60 percent
                if area_mask * 100 < 0.4 or area_mask * 100 > 60:
                    continue
                
                #FYI: Heatmap but filtered everything except at mask
                binary_heatmap_image = cv2.imread(heat_map_path, cv2.IMREAD_GRAYSCALE) 
                filter_heatmap = np.zeros_like(binary_heatmap_image)
                filter_heatmap = np.where(binary_mask_image == 0, filter_heatmap, binary_heatmap_image)
                
                #FYI: This score can be better, but now it is combination of awarding (lower density of heatmap at mask) and (higher area of mask)
                #FYI: (+ 0.5% of num_white_pixels in mask) just to avoid 0 sums
                #FYI: Updating weightage of area to only be 0.25
                score = ((np.sum(filter_heatmap == 255) + 0.005 * num_white_pixels) / area_mask) / (0.25 * area_mask)

                #FYI: If the mask covers either 95% white or black areas in the artifacted image, then they will be rejected from mask list
                #FYI: This should fix mask covering white patches.
                #FYI: This should fix masks that very small and cover only small part of dark spots
                #TODO: Would be more efficient if done after sorting
                artifact_image = cv2.imread(artifact_image_path)
                white_pixel_mask = (binary_mask_image == 255)
                filter_artifact = artifact_image[white_pixel_mask]
                average_masked_pixel_value = np.mean(filter_artifact)

                #TODO: Double check this parameter
                # if average_masked_pixel_value > 15 and average_masked_pixel_value < 225:
                if average_masked_pixel_value > 15:
                    #FYI: Saving the score in dict format
                    score_dict[subdirectory + "/" + filename] = score

    #FYI: Sorting the files according to score            
    sorted_score_dict = dict(sorted(score_dict.items(), key=lambda item: item[1]))
    score_list = []

    #FYI: Saving the score to txt file, and saving the masks according to ranks in ranked folder
    for index, (filename, score) in enumerate(sorted_score_dict.items(), start=1):
        score_list.append(f"Filename: {filename}, Score: {score}")
        mask = cv2.imread(os.path.join(parent_directory, filename), cv2.IMREAD_GRAYSCALE) 
        cv2.imwrite(os.path.join(output_folder_path, str(index) + "_" + filename.replace('/', '')), mask)

    with open(os.path.join(output_folder_path, "masks_ranking.txt"), "w") as f:
        f.write("\n".join(score_list))