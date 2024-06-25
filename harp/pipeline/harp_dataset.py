from torch.utils.data import Dataset
from torchvision import transforms
from harp.core.util import set_device
from PIL import Image
import cv2
import os

class HARPDataset(Dataset):
    def __init__(self, folder_path, image_size=[256, 256]):
        self.folder_path = folder_path
        self.image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        self.transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
            ])
        self.image_size = image_size
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        example = {}
        img_path = os.path.join(self.folder_path, self.image_files[idx])

        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        img = set_device(img.unsqueeze(0))

        img_cv = cv2.imread(img_path)
        img_cv = cv2.resize(img_cv, (self.image_size[0], self.image_size[1]))

        example["image_name"] = self.image_files[idx]
        example["image_path"] = img_path
        example["image_tensor"] = img
        example["image"] = img_cv

        return example
    


