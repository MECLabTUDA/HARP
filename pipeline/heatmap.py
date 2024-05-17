import cv2
import numpy as np
import os

#FYI: Doing heatmap from one image denoised and artifact
#FYI: Some parts are only for visualisation
def create_heatmap(reconstructed_images, artifact_image):
    image_sampled = reconstructed_images[0]
    diff = cv2.absdiff(image_sampled, artifact_image)
    diff_normalized = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmap = cv2.applyColorMap(diff_normalized, cv2.COLORMAP_JET)

    diff = np.where(diff<55, diff, 255)
    diff = np.where(diff>55, diff, 0)
    diff_normalized = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    #FYI: Doing heatmaps out of multiple denoised image 
    heatmap_sum = np.zeros_like(artifact_image, dtype=np.float32)

    for image in reconstructed_images:
        diff = cv2.absdiff(image, artifact_image)
        diff = np.where(diff<50, diff, 255)
        diff = np.where(diff>50, diff, 0)
        diff_normalized = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        heatmap_sum += diff_normalized

    heatmap_average = heatmap_sum / len(reconstructed_images)
    heatmap_average = np.where(heatmap_average<50, heatmap_average, 255)
    heatmap_average = np.where(heatmap_average>50, heatmap_average, 0)
    new = np.zeros_like(heatmap_average) 
    new = np.where(heatmap_average == [0, 0, 0], new, 255)
    heatmap_average_normalized = cv2.normalize(heatmap_average, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmap_visualization = cv2.applyColorMap(heatmap_average_normalized, cv2.COLORMAP_JET)

    #FYI: Create binary image that is inversed
    gray_image = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
    _, inverted_image = cv2.threshold(binary_image, 1, 255, cv2.THRESH_BINARY_INV)
                
    return binary_image