import cv2
import numpy as np
import os

#FYI: Doing heatmap from one image denoised and artifact
def create_heatmap(heatmap_path, image_sampled_root_path, artifact_image_path):
    image_sampled = os.path.join(image_sampled_root_path, os.listdir(image_sampled_root_path)[0])
    image_sampled = cv2.imread(image_sampled)
    image_artifact = cv2.imread(artifact_image_path)
    
    diff = cv2.absdiff(image_sampled, image_artifact)
    diff_normalized = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmap = cv2.applyColorMap(diff_normalized, cv2.COLORMAP_JET)
    cv2.imwrite(heatmap_path+"/1_heatmap_standard.png", heatmap)

    diff = np.where(diff<55, diff, 255)
    diff = np.where(diff>55, diff, 0)
    diff_normalized = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite(heatmap_path+"/2_heatmap_absdiff_threshold_standard.png", diff_normalized)

    #FYI: Doing heatmaps out of multiple denoised image 
    heatmap_sum = np.zeros_like(image_artifact, dtype=np.float32)
    file_list = os.listdir(image_sampled_root_path)
    num_files = len(file_list)

    for file in file_list:
        image_denoised = cv2.imread(os.path.join(image_sampled_root_path, file))  # Convert to grayscale
        diff = cv2.absdiff(image_denoised, image_artifact)
        diff = np.where(diff<50, diff, 255)
        diff = np.where(diff>50, diff, 0)
        diff_normalized = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        heatmap_sum += diff_normalized

    heatmap_average = heatmap_sum / num_files
    heatmap_average = np.where(heatmap_average<50, heatmap_average, 255)
    heatmap_average = np.where(heatmap_average>50, heatmap_average, 0)

    new = np.zeros_like(heatmap_average) 
    new = np.where(heatmap_average == [0, 0, 0], new, 255)

    heatmap_average_normalized = cv2.normalize(heatmap_average, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmap_visualization = cv2.applyColorMap(heatmap_average_normalized, cv2.COLORMAP_JET)

    cv2.imwrite(heatmap_path+"/3_heatmap_average_threshold.png", heatmap_visualization)
    cv2.imwrite(heatmap_path+"/4_heatmap_average_threshold.png", new)

    #FYI: Create binary image that is inversed
    gray_image = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
    _, inverted_image = cv2.threshold(binary_image, 1, 255, cv2.THRESH_BINARY_INV)
                
    cv2.imwrite(heatmap_path+"/5_binary.png", binary_image)
    cv2.imwrite(heatmap_path+"/6_inversed_binary.png", inverted_image)

# heatmap_path = '/gris/gris-f/homestud/ssivakum/denoising_diffusion/logs/images_documentation/heatmap/thread/heatmaps'
# image_sampled_root_path = '/gris/gris-f/homestud/ssivakum/denoising_diffusion/logs/images_documentation/heatmap/thread/noised_denoised_images'
# image_sampled = cv2.imread('/gris/gris-f/homestud/ssivakum/denoising_diffusion/logs/images_documentation/heatmap/thread/noised_denoised_images/900_0.png')  # Convert to grayscale
# image_artifact = cv2.imread('/gris/gris-f/homestud/ssivakum/denoising_diffusion/logs/images_documentation/heatmap/thread/Thread.png')  # Convert to grayscale
# create_heatmap(heatmap_path, image_sampled_root_path, image_sampled, image_artifact)