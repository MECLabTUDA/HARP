import glob
import cv2
import os
import torch
import numpy as np
from torchmetrics.image import FrechetInceptionDistance, KernelInceptionDistance
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
from PIL import Image


def eval_fid_kid(real_path: str, gen_path: str, device: str = 'cuda'):

    fid = FrechetInceptionDistance().to(device)
    kid = KernelInceptionDistance(normalize=True).to(device)
    real_imgs = glob.glob(real_path + "/*.png")
    gen_imgs = glob.glob(gen_path + "/*.png")

    for real_img_path in tqdm(real_imgs):
        real = cv2.resize(np.array(Image.open(real_img_path)), (256, 256))
        real = torch.from_numpy(real).to(device).unsqueeze(0).permute(0, 3, 1, 2)
        fid.update(real, real=True)

    for gen_img_path in tqdm(gen_imgs):
        gen = torch.from_numpy(np.array(Image.open(gen_img_path))).to(device).unsqueeze(0).permute(0, 3, 1, 2)
        fid.update(gen, real=False)

    fid_score = fid.compute()
    print(f"FID: {fid_score}")


def eval_psnr_ssim(real_path: str, gen_path: str, artifact: str):
    
    psnr_list = []
    ssim_list = []
    real_imgs = glob.glob(real_path + "/*.png")

    for real_img_path in tqdm(real_imgs):
        real = cv2.resize(cv2.imread(real_img_path), (256, 256))
        file_name = os.path.basename(real_img_path).replace('original', artifact)
        gen_img_path = os.path.join(gen_path, file_name)
        gen = cv2.imread(gen_img_path)

        psnr_score = cv2.PSNR(real, gen)
        psnr_list.append(psnr_score)
        ssim_score = ssim(real, gen, channel_axis=-1, data_range=255)
        ssim_list.append(ssim_score)

    print(f"PSNR: {np.mean(psnr_list)}")
    print(f"SSIM: {np.mean(ssim_list)}")



if __name__ == "__main__":
    
    artifact = ["dark_spot", "squamos", "thread", "blood_cells", "blood_group", "compression", "cut", "air_bubble", "overlap", "folding"]

    for arti in artifact:
        print("Artifact: " + arti)
        real_path = "/local/scratch/sharvien/OOD_Removal_Restoration_Methods/artifacted_images/original"
        gen_path = os.path.join("/local/scratch/sharvien/OOD_Removal_Restoration_Methods/method_outputs/inpaint_unsupervised", arti)
        eval_fid_kid(real_path, gen_path)
        eval_psnr_ssim(real_path, gen_path, arti)

