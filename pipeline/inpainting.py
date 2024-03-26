import argparse
import torch
import cv2
import os
import core.praser as Praser
from core.util import set_device, save_tensor_as_img
from models.network import Network
from PIL import Image
from torchvision import transforms, utils

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


def noising_and_denoising_map(input_path, result_path, n_noising_step, n_iter, batch_size):
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
        sampled_image = model.noising_denoising(img_batch, n_noising_step)
        sampled_image = torch.chunk(sampled_image, batch_size, dim=0)
        sampled_images.extend(sampled_image)

    for i in range(len(sampled_images)):
        output_img = sampled_images[i].detach().float().cpu()
        output_img = output_img[0,:,:,:].permute(1, 2, 0).numpy()
        output_img = ((output_img+1) * 127.5).round()
        cv2.imwrite(os.path.join(result_path, str(n_noising_step)+"_"+str(i)+".png"), cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
        
# n_noising_step = 100
# n_iter = 1
# batch_size = 8
# input_path = "/gris/gris-f/homestud/ssivakum/temp/11/artifacts/air_bubble/air_bubble.png"
# result_path = "/gris/gris-f/homestud/ssivakum/temp/temp"
# noising_and_denoising_map(n_noising_step, n_iter, batch_size, input_path, result_path)


def inpaint(input_path, masks_path, batch_size):
    #TODO: Resize based on config param
    transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
                ])
    transform_mask = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 256)), 
                transforms.ToTensor()
                ])
    
    img = Image.open(input_path).convert('RGB')
    img = transform(img)
    gt_image_list, cond_image_list, mask_list, inpainted_image_list = [], [], [], []
    
    for mask in masks_path:
        mask = cv2.imread(mask)
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
        inpainted_image_list.extend(sampled_image)

        return inpainted_image_list

# input_path = "/gris/gris-f/homestud/ssivakum/img2img/result/11/original.png"
# masks_path = ["/local/scratch/BCSS_ood_removal/112/artifacts/thread/methods/repaint/masks/ranked/1_sam_mask0.png", "/local/scratch/BCSS_ood_removal/112/artifacts/thread/methods/repaint/masks/ranked/2_sam_mask65.png"]
# inpaint(input_path, masks_path, 8)

#TODO: Add jump samples and resample parameter here
def inpaint_with_jump_sampling(input_path, masks_path, batch_size):
    #TODO: Resize based on config param
    transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
                ])
    transform_mask = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 256)), 
                transforms.ToTensor()
                ])
    
    img = Image.open(input_path).convert('RGB')
    img = transform(img)
    gt_image_list, cond_image_list, mask_list, inpainted_image_list = [], [], [], []
    
    for mask in masks_path:
        mask = cv2.imread(mask)
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
        inpainted_image_list.extend(sampled_image)

        return inpainted_image_list

# input_path = "/local/scratch/BCSS_ood_removal/11/artifacts/air_bubble/air_bubble.png"
# masks_path = ["/local/scratch/BCSS_ood_removal/11/artifacts/air_bubble/methods/inpaint/masks/ranked/enlarged_1_sam_mask1.png", "/local/scratch/BCSS_ood_removal/11/artifacts/air_bubble/methods/inpaint/masks/ranked/enlarged_1_sam_mask1.png", "/local/scratch/BCSS_ood_removal/11/artifacts/air_bubble/methods/inpaint/masks/ranked/enlarged_1_sam_mask1.png", "/local/scratch/BCSS_ood_removal/11/artifacts/air_bubble/methods/inpaint/masks/ranked/enlarged_1_sam_mask1.png"]
# output_path = "/gris/gris-f/homestud/ssivakum/img2img/result/img2img"
# inpainted_image_list, video = inpaint_with_jump_sampling(input_path, masks_path, 8)

# output_filename = "/gris/gris-f/homestud/ssivakum/img2img/result/img2img/1.animation.mp4"
# frame_rate = 30 
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# video_writer = cv2.VideoWriter(output_filename, fourcc, frame_rate, (256, 256))

# for i in range(len(video)):
#     tensor_img = video[i].detach().float().cpu()
#     output_img = tensor_img[0,:,:,:].permute(1, 2, 0).numpy()
#     output_img = ((output_img+1) * 127.5).round()
#     video_writer.write(output_img.astype('uint8'))

# video_writer.release()

# for it in range(len(inpainted_image_list)):
#     save_tensor_as_img(inpainted_image_list[it], output_path+"/"+str(it)+".png")


