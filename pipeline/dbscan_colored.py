import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import os
import skimage.io
from scipy import ndimage as ndi

def create_dbscan_mask(input, mask_path=None):
    output = []
    if mask_path is not None:
        if not os.path.exists(mask_path):
            os.makedirs(mask_path)

    # Clustering low values pixels
    img = skimage.io.imread(input)[:,:,2]
    image_max = ndi.maximum_filter(-img, size=10, mode='constant')
    image_max = image_max > np.quantile(image_max, 0.80)
    X = np.array(np.nonzero(image_max)).transpose()
    try:
        clustering = DBSCAN(eps=10, min_samples=20).fit(X)
    except:
         print("No viable cluster found")
    else:
        if mask_path is not None:
            # Plot overall for visualisation
            fig, axs = plt.subplots(ncols=2, sharex=True, sharey=True)
            axs[0].imshow(image_max, cmap=plt.cm.gray)
            clustering.labels_[clustering.labels_ == -1] = max(clustering.labels_) + 1
            axs[1].scatter(X[:,1], X[:,0], s = 0.01, c = clustering.labels_) 
            plt.axis('scaled')
            plt.savefig(os.path.join(mask_path, 'combined_masks_dark.png'), dpi=300)

        unique_labels = np.unique(clustering.labels_)
        cluster_area = 0
        for cluster_label in unique_labels:
            cluster_mask = clustering.labels_ == cluster_label
            cluster_mask_image = np.zeros_like(image_max)
            cluster_mask_image[tuple(X.T)] = cluster_mask
            cluster_mask_image = cluster_mask_image.astype(np.uint8) * 255
            if np.sum(cluster_mask_image == 255) > cluster_area:
                cluster_area = np.sum(cluster_mask_image == 255)
                biggest_low_value_cluster = cluster_mask_image
        if mask_path is not None:
            cluster_filename = os.path.join(mask_path, f'{1000}.png')
            skimage.io.imsave(cluster_filename, biggest_low_value_cluster)
            output = []
        output.append({'name':'dbscan', 'index':1, 'mask':biggest_low_value_cluster})

    # Clustering high values pixels
    img = skimage.util.invert(img)
    image_max = ndi.maximum_filter(-img, size=5, mode='constant')
    image_max = skimage.util.invert(image_max)
    image_max = image_max > np.quantile(image_max, 0.80)
    X_2 = np.array(np.nonzero(image_max)).transpose()
    try:
        clustering_2 = DBSCAN(eps=10, min_samples=20).fit(X_2)
    except:
         print("No viable cluster found")
    else:
        if mask_path is not None:
            fig, axs = plt.subplots(ncols=2, sharex=True, sharey=True)
            axs[0].imshow(image_max, cmap=plt.cm.gray)
            clustering_2.labels_[clustering_2.labels_ == -1] = max(clustering_2.labels_) + 1
            axs[1].scatter(X_2[:,1], X_2[:,0], s = 0.01, c = clustering_2.labels_) 
            plt.axis('scaled')
            plt.savefig(os.path.join(mask_path, 'combined_masks_white.png'), dpi=300)

        unique_labels_2 = np.unique(clustering_2.labels_)
        cluster_area = 0
        for cluster_label in unique_labels_2:
            cluster_mask = clustering_2.labels_ == cluster_label
            cluster_mask_image = np.zeros_like(image_max)
            cluster_mask_image[tuple(X_2.T)] = cluster_mask
            cluster_mask_image = cluster_mask_image.astype(np.uint8) * 255
            if np.sum(cluster_mask_image == 255) > cluster_area:
                    cluster_area = np.sum(cluster_mask_image == 255)
                    biggest_high_value_cluster = cluster_mask_image
        if mask_path is not None:
                cluster_filename = os.path.join(mask_path, f'{2000}.png')
                skimage.io.imsave(cluster_filename, biggest_high_value_cluster)
        output.append({'name':'dbscan', 'index':2, 'mask':biggest_high_value_cluster})
    return output

