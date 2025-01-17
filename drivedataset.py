
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class DriveDataset(Dataset):
    def __init__(self, images_path, masks_path):
        self.images_path = images_path
        self.masks_path = masks_path
        self.n_samples = len(images_path)

    def __getitem__(self, index):
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        image = cv2.resize(image, (512, 512)) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image).float()

        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (512, 512)) / 255.0
        mask = np.expand_dims(mask, axis=0)
        mask = torch.from_numpy(mask).float()

        return image, mask

    def __len__(self):
        return self.n_samples
