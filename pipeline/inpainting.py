import argparse
import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import core.praser as Praser
from core.util import set_device
from models.network import Network

#TODO: Fix this urgently
def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="/gris/gris-f/homestud/ssivakum/img2img/config/config_restoration_model.json")
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'], default='test')
    parser.add_argument('-b', '--batch', type=int, default=4)
    parser.add_argument('-gpu', '--gpu_ids', type=str, default="0")
    parser.add_argument('-d', '--debug', action='store_true')
    args = parser.parse_args()
    opt = Praser.parse(args)
    return opt

opt = parse_config()
model_args = opt["model"]["which_networks"][0]["args"]
model = Network(**model_args)
#TODO: Change to get from config
model_pth = "/local/scratch/sharvien/HARP_Model_Weights/restoration_model.pth"
state_dict = torch.load(model_pth)
model.load_state_dict(state_dict, strict=False)
model = set_device(model)
model.set_new_noise_schedule(phase='test')
model.eval()


def noising_and_denoising_map(input_path, n_noising_step, n_iter, batch_size):
    #TODO: Check why 100 steps not 250 steps
    #TODO: Do percentage of noising instead of number
    #TODO: Resize based on config param -> transforms.Resize((256, 256))
    img = Image.open(input_path).convert('RGB')
    transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
                ])
    img = transform(img)
    img = set_device(img.unsqueeze(0))
    img = torch.cat([img] * n_iter, dim=0)

    sampled_images = []
    for n in range(0, n_iter, batch_size):
        img_batch = img[n:n+batch_size,:,:,:]
        sampled_image_batch = model.noising_denoising(img_batch, n_noising_step)
        sampled_image_batch = torch.chunk(sampled_image_batch, batch_size, dim=0)
    
        for sample in sampled_image_batch:        
            sample = sample.detach().cpu()
            sample = sample.squeeze(0).permute(1, 2, 0).numpy()
            sample = ((sample+1) * 127.5).round().astype(np.uint8)
            sample = cv2.cvtColor(sample, cv2.COLOR_RGB2BGR)
            sampled_images.append(sample)

    return sampled_images


def inpaint(input_path, ranked_mask_list, batch_size):
    #TODO: Resize based on config param
    transform = transforms.Compose([
                #TODO: Image resize parametrize
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
                ])
    transform_mask = transforms.Compose([
                transforms.ToPILImage(),
                #TODO: Image resize parametrize
                transforms.Resize((256, 256)), 
                transforms.ToTensor()
                ])
    
    img = Image.open(input_path).convert('RGB')
    img = transform(img)
    gt_image_list, cond_image_list, mask_list, inpainted_image_list = [], [], [], []
    
    for mask_ in ranked_mask_list:
        mask = mask_["mask_dilated"]
        mask = transform_mask(mask)
        mask = (mask[0,:,:]).unsqueeze(0)

        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        cond_image = cond_image

        gt_image_list.append(img.unsqueeze(0))
        cond_image_list.append(cond_image.unsqueeze(0))
        mask_list.append(mask.unsqueeze(0))
    
    gt_images = torch.cat(gt_image_list, dim=0)
    cond_images = torch.cat(cond_image_list, dim=0)
    masks = torch.cat(mask_list, dim=0)

    gt_images = set_device(gt_images)
    cond_images = set_device(cond_images)
    masks = set_device(masks)
    
    for n in range(0, len(mask_list), batch_size):
        gt_image = gt_images[n:n+batch_size,:,:,:]
        cond_image = cond_images[n:n+batch_size,:,:,:]
        mask = masks[n:n+batch_size,:,:,:]

        with torch.no_grad():
            output, visuals = model.restoration(cond_image, y_t=cond_image,
                                                y_0=gt_image, mask=mask)
        
        sampled_image = torch.chunk(output, batch_size, dim=0)
        for i in range(len(sampled_image)):
            ranked_mask_list[n+i]["restored_image"] = sampled_image[i]
        return ranked_mask_list


#TODO: Add jump samples and resample parameter here
def inpaint_with_jump_sampling(input_path, ranked_mask_list, batch_size):
    #TODO: Resize based on config param
    transform = transforms.Compose([
                #TODO: Image resize parametrize
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
                ])
    transform_mask = transforms.Compose([
                transforms.ToPILImage(),
                #TODO: Image resize parametrize
                transforms.Resize((256, 256)), 
                transforms.ToTensor()
                ])
    
    img = Image.open(input_path).convert('RGB')
    img = transform(img)
    gt_image_list, cond_image_list, mask_list, inpainted_image_list = [], [], [], []
    
    for mask_ in ranked_mask_list:
        mask = mask_["mask_dilated"]
        mask = transform_mask(mask)
        mask = (mask[0,:,:]).unsqueeze(0)

        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        cond_image = cond_image

        gt_image_list.append(img.unsqueeze(0))
        cond_image_list.append(cond_image.unsqueeze(0))
        mask_list.append(mask.unsqueeze(0))
    
    gt_images = torch.cat(gt_image_list, dim=0)
    cond_images = torch.cat(cond_image_list, dim=0)
    masks = torch.cat(mask_list, dim=0)

    gt_images = set_device(gt_images)
    cond_images = set_device(cond_images)
    masks = set_device(masks)
    
    for n in range(0, len(mask_list), batch_size):
        gt_image = gt_images[n:n+batch_size,:,:,:]
        cond_image = cond_images[n:n+batch_size,:,:,:]
        mask = masks[n:n+batch_size,:,:,:]

        with torch.no_grad():
            output = model.restoration_with_jump_sampling(cond_image, y_t=cond_image,
                                                y_0=gt_image, mask=mask)
        
        sampled_image = torch.chunk(output, batch_size, dim=0)
        for i in range(len(sampled_image)):
            ranked_mask_list[n+i]["restored_image"] = sampled_image[i]
        return ranked_mask_list