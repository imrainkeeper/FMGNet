# single_image_loader系列都是为了进行可视化工作而特地将每一张被指定的图片打包作为输入

import torch
from torch.autograd import Variable
import os
import scipy.io
import cv2
import torch.utils.data as data
import numpy as np
import random
import pandas as pd
import h5py
from PIL import Image


class SingleImageDataset(data.Dataset):
    def __init__(self, img_path, gt_path, transform=None):
        self.img_path = img_path
        self.gt_path = gt_path
        self.transform = transform

    def __getitem__(self, item):
        assert item < 1, 'index range error'
        img_path = os.path.join(self.img_path)
        img = Image.open(img_path).convert('RGB')
        img = img.resize((1024, 1024), Image.ANTIALIAS)

        gt_path = os.path.join(self.gt_path)
        gt_file = h5py.File(gt_path)
        gt_density_map = np.asarray(gt_file['density'])

        original_gt_sum = np.sum(gt_density_map)
        gt_density_map = cv2.resize(gt_density_map, (128, 128), interpolation=cv2.INTER_CUBIC)
        current_gt_sum = np.sum(gt_density_map)
        gt_density_map = gt_density_map * (original_gt_sum / current_gt_sum)

        if img is None:
            print('Unable to read image %s, Exiting ...', self.img_path)
            exit(0)
        if self.transform is not None:
            img = self.transform(img)

        return img, gt_density_map

    def __len__(self):
        return 1




