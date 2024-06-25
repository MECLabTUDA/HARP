import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, List
from skimage import io as skio, util as skutil
from skimage import morphology as skmorph
from sklearn.cluster import DBSCAN
from scipy import ndimage as ndi
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

class ArtifactSegment:
    def __init__(self, harp_config):
        self.config = harp_config

    def create_dbscan_mask(self, input, mask_path=None):
        output = []
        if mask_path is not None:
            if not os.path.exists(mask_path):
                os.makedirs(mask_path)

        # Clustering low values pixels
        img = input[:, :, 2]
        image_max = ndi.maximum_filter(-img, size=10, mode='constant')
        image_max = image_max > np.quantile(image_max, 0.80)
        X = np.array(np.nonzero(image_max)).transpose()
        try:
            clustering = DBSCAN(eps=10, min_samples=20).fit(X)
        except:
            print("No viable cluster found")
        else:
            if mask_path is not None:
                fig, axs = plt.subplots(ncols=2, sharex=True, sharey=True)
                axs[0].imshow(image_max, cmap=plt.cm.gray)
                clustering.labels_[clustering.labels_ == -1] = max(clustering.labels_) + 1
                axs[1].scatter(X[:, 1], X[:, 0], s=0.01, c=clustering.labels_)
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
                skio.imsave(cluster_filename, biggest_low_value_cluster)
            output.append({'name': 'dbscan', 'index': 1, 'mask': biggest_low_value_cluster})

        # Clustering high values pixels
        img = skutil.invert(img)
        image_max = ndi.maximum_filter(-img, size=5, mode='constant')
        image_max = skutil.invert(image_max)
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
                axs[1].scatter(X_2[:, 1], X_2[:, 0], s=0.01, c=clustering_2.labels_)
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
                skio.imsave(cluster_filename, biggest_high_value_cluster)
            output.append({'name': 'dbscan', 'index': 2, 'mask': biggest_high_value_cluster})

        return output
    
    def create_sam_masks(self, input):
        sam = sam_model_registry[self.config["sam_model_type"]](self.config["sam_checkpoint"])
        _ = sam.to("cuda:{}".format(self.config["gpu_id"]))
        generator = SamAutomaticMaskGenerator(sam, output_mode="binary_mask")

        masks = generator.generate(input)
        output, metadata = self.write_masks_to_folder(masks)

        return output

    def write_masks_to_folder(self, masks: List[Dict[str, Any]]):
        header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"
        metadata = [header]
        output = []
        for i, mask_data in enumerate(masks):
            mask = mask_data["segmentation"]
            output.append({
                'name': 'sam',
                'index': i,
                'mask': mask.astype(np.uint8) * 255
            })
            mask_metadata = [
                str(i),
                str(mask_data["area"]),
                *[str(x) for x in mask_data["bbox"]],
                *[str(x) for x in mask_data["point_coords"][0]],
                str(mask_data["predicted_iou"]),
                str(mask_data["stability_score"]),
                *[str(x) for x in mask_data["crop_box"]],
            ]
            row = ",".join(mask_metadata)
            metadata.append(row)
        return output, metadata

    def write_masks_to_png(self, masks: List[Dict[str, Any]], image, mask_path: str, base: str) -> None:
        if len(masks) == 0:
            return
        sorted_anns = sorted(masks, key=lambda x: x['area'], reverse=True)
        img = np.zeros((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 3))
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.random.choice(range(1, 256), size=3)
            img[m] = color_mask
        img = np.uint8(img)
        blend = 0.5
        result = cv2.addWeighted(image, blend, img, 1 - blend, 0)
        result = np.where(img == 0, image, result)
        cv2.imwrite(os.path.join(mask_path, base + '_mask.png'), img)
        cv2.imwrite(os.path.join(mask_path, base + '_mask_overlay.png'), result)
