import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import os

class FaceTextureDataset(Dataset):
    def __init__(self, root_dir, phase='train'):
        self.root_dir = root_dir
        self.phase = phase
        self.image_dir = os.path.join(root_dir, phase, './datasets/data_choose_enhance/image')
        self.mask_dir = os.path.join(root_dir, phase, './datasets/data_choose_enhance/mask')
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace('.png', '.npy'))

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = np.load(mask_path).astype(np.float32)
        mask = np.expand_dims(mask, axis=0)

        image = (image / 127.5) - 1.0
        mask = mask * 2.0 - 1.0

        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).float()

        return {'image': image, 'mask': mask}

    @staticmethod
    def create_dataloader(root_dir, phase, batch_size=4, shuffle=True):
        dataset = FaceTextureDataset(root_dir, phase)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)