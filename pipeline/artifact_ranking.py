#This will take root mask file, find all the binary mask file, and arrange the binary file based on a score
#Score would be mix of mask density (using heatmap) and area (normalised)
import numpy as np

def artifact_inverting(combined_mask_list):
    #FYI: Go thorugh the files of mask and generated iverted masks for mask above 50%
    for mask_ in combined_mask_list:
        mask = mask_['mask']
        num_white_pixels = np.sum(mask == 255)
        height, width = mask.shape
        total_pixels = height * width
        area_mask = num_white_pixels / total_pixels

        if area_mask * 100 > 49:
            inverted_binary_mask_image = np.where(mask == 0, 255, 0)
            inverted_mask = mask_.copy()
            inverted_mask['mask'] = inverted_binary_mask_image.astype(np.uint8)
            inverted_mask['index'] = 1000 + inverted_mask['index'] 
            combined_mask_list.append(inverted_mask)     
    return combined_mask_list


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


def artifact_ranking(combined_mask_list, heat_map, artifact_image):
    processed_masks = []
    mask_restored_image_list = []
    #FYI: Searching through all files with just numbers in the parent directory to retrieve all binary masks
    for idx, mask_ in enumerate(combined_mask_list):  
        mask = mask_['mask']   
        #FYI: Calculate number of pixel on mask and the area of the white mask
        num_white_pixels = np.sum(mask == 255)
        height, width = mask.shape
        total_pixels = height * width
        area_mask = num_white_pixels / total_pixels
        
        #FYI: Reject masks that are simialr than 97%
        processed_masks, similarity = check_similarity(mask, processed_masks)
        if similarity > 0.97:
            continue
        #FYI: Rejects masks that are smaller than 0.4 percent or bigger than 60 percent
        if area_mask * 100 < 0.4 or area_mask * 100 > 60:
            continue
        #FYI: Heatmap but filtered everything except at mask
        binary_heatmap_image = heat_map
        filter_heatmap = np.zeros_like(binary_heatmap_image)
        filter_heatmap = np.where(mask == 0, filter_heatmap, binary_heatmap_image)

        #FYI: This score can be better, but now it is combination of awarding (lower density of heatmap at mask) and (higher area of mask)
        #FYI: (+ 0.5% of num_white_pixels in mask) just to avoid 0 sums. Updating weightage of area to only be 0.25
        score = ((np.sum(filter_heatmap == 255) + 0.005 * num_white_pixels) / area_mask) / (0.25 * area_mask)
        #FYI: If the mask covers either 95% white or black areas in the artifacted image, then they will be rejected from mask list
        #FYI: This should fix mask covering white patches. This should fix masks that very small and cover only small part of dark spots
        white_pixel_mask = (mask == 255)
        filter_artifact = artifact_image[white_pixel_mask]
        average_masked_pixel_value = np.mean(filter_artifact)

        if average_masked_pixel_value > 15:
            entry = mask_.copy()
            entry['score'] = score
            mask_restored_image_list.append(entry)

    #FYI: Sorting the files according to score  
    mask_restored_image_list = sorted(mask_restored_image_list, key=lambda x: x['score'])
    return mask_restored_image_list