import torch
import os
import torchvision
import sys
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import numpy as np
import cv2


class CustomDataset(Dataset):
    def __init__(self, path, transform):
        self.path_mtx = datasets.ImageFolder(path)
        self.transform = transform
        # print(self.path_mtx.samples)
        self.norm = transforms.Compose([transforms.Normalize(std=0.5, mean=0.5)])


    def __len__(self):
        return len(self.path_mtx.samples)

    def __getitem__(self, idx):
        img = cv2.imread(self.path_mtx.samples[idx][0])
        label = self.path_mtx.samples[idx][1]

        img = self.transform(img)
        img = self.norm(img)

        return img, label