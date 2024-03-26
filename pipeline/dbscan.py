#This script do DBSCAN segmentation and save the clusters separately as binary image
#This will also produced combined_masks where all cluster are saved in one image
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import os
import skimage.io
from scipy import ndimage as ndi

def create_dbscan_mask(img_path, output_dir):
    img = skimage.io.imread(img_path)[:,:,2]
    image_max = ndi.maximum_filter(-img, size=10, mode='constant')
    image_max = image_max > np.quantile(image_max, 0.9)

    # Extract coordinates of the identified maxima
    X = np.array(np.nonzero(image_max)).transpose()
    clustering = DBSCAN(eps=10, min_samples=20).fit(X)

    fig, axs = plt.subplots(ncols=2, sharex=True, sharey=True)
    axs[0].imshow(image_max, cmap=plt.cm.gray)
    clustering.labels_[clustering.labels_ == -1] = max(clustering.labels_) + 1
    axs[1].scatter(X[:,1], X[:,0], s = 0.01, c = clustering.labels_) 
    plt.axis('scaled')
    plt.savefig(os.path.join(output_dir, 'combined_masks.png'), dpi=300)

    # Save each cluster as a binary image
    unique_labels = np.unique(clustering.labels_)
    num_clusters = len(unique_labels) - 1

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for cluster_label in unique_labels:
        cluster_mask = clustering.labels_ == cluster_label
        cluster_mask_image = np.zeros_like(image_max)
        cluster_mask_image[tuple(X.T)] = cluster_mask

        # Save the cluster mask as a binary image with the cluster number as filename
        cluster_filename = os.path.join(output_dir, f'{cluster_label}.png')
        skimage.io.imsave(cluster_filename, cluster_mask_image.astype(np.uint8) * 255)

    print("Cluster masks have been saved.")

# output_dir = "/gris/gris-f/homestud/ssivakum/img2img/result/dbscan"
# input_path = "/gris/gris-f/homestud/ssivakum/img2img/result/anomalib/input/air_bubble/air_bubble11.png"
# input_path_2 = "/gris/gris-f/homestud/ssivakum/img2img/result/anomalib/input/cut/cut118.png"
# create_dbscan_mask(input_path, output_dir)