# 标准data_loader，1024x1024大小

import os
import random
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import h5py
import cv2

class ImageDataset(Dataset):
    def __init__(self, img_dir, gt_dir, train=False, transform=None):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.img_names = [img_name for img_name in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, img_name))]
        if train:        # 训练时数据扩增
            # self.img_names = self.img_names * 4
            pass
        random.shuffle(self.img_names)
        self.train = train
        self.transform = transform
        assert len(self.img_names) > 0

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        assert index < len(self.img_names), 'index range error'
        img_path = os.path.join(self.img_dir, self.img_names[index])
        img = Image.open(img_path).convert('RGB')
        img = img.resize((1024, 1024), Image.ANTIALIAS)

        gt_path = os.path.join(self.gt_dir, self.img_names[index].replace('.jpg', '.h5'))
        gt_file = h5py.File(gt_path)
        gt = np.asarray(gt_file['density'])

        original_gt_sum = np.sum(gt)
        gt = cv2.resize(gt, (128, 128), interpolation=cv2.INTER_CUBIC)
        current_gt_sum = np.sum(gt)
        gt = gt * (original_gt_sum / current_gt_sum)

        gt = gt.reshape((1, gt.shape[0], gt.shape[1]))
        gt = gt.astype(np.float32, copy=False)

        if img is None:
            print('Unable to read image %s, Exiting ...', self.img_path)
            exit(0)
        if self.transform is not None:
            img = self.transform(img)

        return img, gt
