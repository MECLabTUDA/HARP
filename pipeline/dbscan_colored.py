#This script do DBSCAN segmentation and save the clusters separately as binary image
#This will also produced combined_masks where all cluster are saved in one image
#TODO: There is another script called dbscan also. Maybe remove that
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import os
import skimage.io
from scipy import ndimage as ndi

def create_dbscan_mask(img_path, output_dir, save_biggest=True):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Clustering low values pixels
    img = skimage.io.imread(img_path)[:,:,2]
    image_max = ndi.maximum_filter(-img, size=10, mode='constant')
    image_max = image_max > np.quantile(image_max, 0.80)
    X = np.array(np.nonzero(image_max)).transpose()
    try:
        clustering = DBSCAN(eps=10, min_samples=20).fit(X)
    except:
         print("No viable cluster found")
    else:
        # Plot overall for visualisation
        fig, axs = plt.subplots(ncols=2, sharex=True, sharey=True)
        axs[0].imshow(image_max, cmap=plt.cm.gray)
        clustering.labels_[clustering.labels_ == -1] = max(clustering.labels_) + 1
        axs[1].scatter(X[:,1], X[:,0], s = 0.01, c = clustering.labels_) 
        plt.axis('scaled')
        plt.savefig(os.path.join(output_dir, 'combined_masks_dark.png'), dpi=300)

        # Save each cluster as a binary image
        unique_labels = np.unique(clustering.labels_)
        cluster_area = 0
        for cluster_label in unique_labels:
            cluster_mask = clustering.labels_ == cluster_label
            cluster_mask_image = np.zeros_like(image_max)
            cluster_mask_image[tuple(X.T)] = cluster_mask
            cluster_mask_image = cluster_mask_image.astype(np.uint8) * 255
            if not save_biggest:
                cluster_filename = os.path.join(output_dir, f'{1000+cluster_label}.png')
                skimage.io.imsave(cluster_filename, cluster_mask_image)
            else:
                if np.sum(cluster_mask_image == 255) > cluster_area:
                    cluster_area = np.sum(cluster_mask_image == 255)
                    biggest_cluster_mask_image = cluster_mask_image
        if save_biggest:
                cluster_filename = os.path.join(output_dir, f'{1000}.png')
                skimage.io.imsave(cluster_filename, biggest_cluster_mask_image)


    #TODO: This is kinda repeating
    # Clustering high values pixels
    img = skimage.util.invert(img)
    image_max = ndi.maximum_filter(-img, size=5, mode='constant')
    image_max = skimage.util.invert(image_max)
    image_max = image_max > np.quantile(image_max, 0.8)
    X_2 = np.array(np.nonzero(image_max)).transpose()
    try:
        clustering_2 = DBSCAN(eps=10, min_samples=20).fit(X_2)
    except:
         print("No viable cluster found")
    else:
        fig, axs = plt.subplots(ncols=2, sharex=True, sharey=True)
        axs[0].imshow(image_max, cmap=plt.cm.gray)
        clustering_2.labels_[clustering_2.labels_ == -1] = max(clustering_2.labels_) + 1
        axs[1].scatter(X_2[:,1], X_2[:,0], s = 0.01, c = clustering_2.labels_) 
        plt.axis('scaled')
        plt.savefig(os.path.join(output_dir, 'combined_masks_white.png'), dpi=300)

        unique_labels_2 = np.unique(clustering_2.labels_)
        cluster_area = 0
        for cluster_label in unique_labels_2:
            cluster_mask = clustering_2.labels_ == cluster_label
            cluster_mask_image = np.zeros_like(image_max)
            cluster_mask_image[tuple(X_2.T)] = cluster_mask
            cluster_mask_image = cluster_mask_image.astype(np.uint8) * 255
            if not save_biggest:
                cluster_filename = os.path.join(output_dir, f'{1000+cluster_label}.png')
                skimage.io.imsave(cluster_filename, cluster_mask_image)
            else:
                if np.sum(cluster_mask_image == 255) > cluster_area:
                    cluster_area = np.sum(cluster_mask_image == 255)
                    biggest_cluster_mask_image = cluster_mask_image
        if save_biggest:
                cluster_filename = os.path.join(output_dir, f'{2000}.png')
                skimage.io.imsave(cluster_filename, biggest_cluster_mask_image)

# output_dir = "/gris/gris-f/homestud/ssivakum/img2img/result/dbscan"
# input_path = "/gris/gris-f/homestud/ssivakum/img2img/result/anomalib/input/air_bubble/air_bubble11.png"
# input_path_2 = "/gris/gris-f/homestud/ssivakum/img2img/result/anomalib/input/cut/cut118.png"
# input_path_3 = "/gris/gris-f/homestud/ssivakum/img2img/result/anomalib/input/compression/compression200.png"
# imput_path_4 = "/gris/gris-f/homestud/ssivakum/img2img/result/anomalib/input/thread/thread11.png"
# create_dbscan_mask(input_path, output_dir)


