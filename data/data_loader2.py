# 比data_loader1增加了64个有无人的label

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
        gt_density_map = np.asarray(gt_file['density'])

        original_gt_sum = np.sum(gt_density_map)
        gt_density_map = cv2.resize(gt_density_map, (128, 128), interpolation=cv2.INTER_CUBIC)
        current_gt_sum = np.sum(gt_density_map)
        gt_density_map = gt_density_map * (original_gt_sum / current_gt_sum)

        gt_label = np.zeros(8 * 8, dtype=int)
        for i in range(8):
            for j in range(8):
                temp = gt_density_map[8 * i: 8 * (i + 1), 8 * j: 8 * (j + 1)]
                temp_sum = np.sum(temp)
                if temp_sum > 0:
                    gt_label[8 * i + j] = 1    # 有人
                else:
                    gt_label[8 * i + j] = 0    # 没有人

        gt_label = gt_label.reshape((8 * 8, 1))    # reshape为batch为64，方便交叉熵计算

        gt_density_map = gt_density_map.reshape((1, gt_density_map.shape[0], gt_density_map.shape[1]))
        gt_density_map = gt_density_map.astype(np.float32, copy=False)

        if img is None:
            print('Unable to read image %s, Exiting ...', self.img_path)
            exit(0)
        if self.transform is not None:
            img = self.transform(img)

        return img, gt_density_map, gt_label
