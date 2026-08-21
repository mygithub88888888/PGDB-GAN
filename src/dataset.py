import numpy as np
import torch
import torch.utils.data
from PIL import Image
import torchvision.transforms as transforms
import os
from utils_gan import load_face_mask

class DataLoader(torch.utils.data.Dataset):
    def __init__(self, img_dir, task, mask_dir=None):
        self.low_img_dir = img_dir
        self.task = task
        self.mask_dir = mask_dir
        self.train_low_data_names = []
        self.train_target_data_names = []

        for root, dirs, names in os.walk(self.low_img_dir):
            for name in names:
                self.train_low_data_names.append(os.path.join(root, name))

        self.train_low_data_names.sort()
        self.count = len(self.train_low_data_names)
        transform_list = []
        transform_list += [transforms.ToTensor()]
        self.transform = transforms.Compose(transform_list)


    def load_images_transform(self, file):

        im = Image.open(file).convert('RGB')
        img_norm = self.transform(im).numpy()
        img_norm = np.transpose(img_norm, (1, 2, 0))
        return img_norm


    def __getitem__(self, index):

        low = self.load_images_transform(self.train_low_data_names[index])
        low = np.asarray(low, dtype=np.float32)
        low = np.transpose(low[:, :, :], (2, 0, 1))
        img_name = os.path.basename(self.train_low_data_names[index])

        if self.mask_dir is not None:
            h, w = low.shape[1], low.shape[2]
            face_mask = load_face_mask(img_name, self.mask_dir, (h, w))
            return torch.from_numpy(low), img_name, face_mask

        return torch.from_numpy(low), img_name

    def __len__(self):
        return self.count