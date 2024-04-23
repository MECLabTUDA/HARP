import numpy as np
import cv2
import os
from skimage.metrics import structural_similarity as ssim

#TODO: Maybe delete this file entirely

#FYI: Mean Squared Error (MSE)
def mse_score(image1, image2):
    err = np.mean((image1 - image2) ** 2)
    return err

#FYI: Peak Signal-to-Noise Ratio (PSNR)
def psnr_score(image1, image2):
    mse_val = mse_score(image1, image2)
    if mse_val == 0:
        return float('inf')
    max_value = np.max(image1)
    psnr_val = 20 * np.log10(max_value / np.sqrt(mse_val))
    return psnr_val

# Structural Similarity Index (SSIM)
def ssim_score(image1, image2, data_range=255):
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    image1 = image1.astype('float64')
    image2 = image2.astype('float64')
    ssim_val = ssim(image1, image2, win_size=7, multichannel=True, data_range=data_range)
    return ssim_val

# image_1 = "/gris/gris-f/homestud/ssivakum/denoising_diffusion/logs/images_documentation/heatmap/100/dark_spot/Dark_Spot.png"
# image_2 = "/gris/gris-f/homestud/ssivakum/denoising_diffusion/logs/images_documentation/heatmap/100/dark_spot/repaint/repainted_1_dbscan_mask17.png"
# artifact_image = cv2.imread(image_1)
# repainted_image = cv2.imread(image_2)

# # Calculate and print metrics
# mse_val = mse_score(artifact_image, repainted_image)
# psnr_val = psnr_score(artifact_image, repainted_image)
# ssim_val = ssim_score(artifact_image, repainted_image)

