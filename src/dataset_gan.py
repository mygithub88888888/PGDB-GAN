import numpy as np
import torch
import torch.utils.data
from PIL import Image
import torchvision.transforms as transforms
import os

class DataLoader(torch.utils.data.Dataset):
    def __init__(self, img_dir, mask_dir, task):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.task = task
        self.image_paths = []
        self.mask_paths = []

        for root, _, names in os.walk(img_dir):
            for name in names:
                if name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(root, name))
        self.image_paths.sort()

        for root, _, names in os.walk(mask_dir):
            for name in names:
                if name.lower().endswith(('.png', '.npy')):
                    self.mask_paths.append(os.path.join(root, name))
        self.mask_paths.sort()

        self.transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        img = Image.open(self.image_paths[index]).convert('RGB')
        img = self.transform(img).float()

        mask_path = self.mask_paths[index]
        if mask_path.endswith('.npy'):
            mask = np.load(mask_path)
            mask = torch.from_numpy(mask)
        else:
            mask = Image.open(mask_path).convert('L')
            mask = self.transform(mask)
        mask = mask.squeeze().float()

        return img, os.path.basename(self.image_paths[index]), mask

    def __len__(self):
        return len(self.image_paths)