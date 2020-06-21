# 查看某个checkpoints里面保存的网络在“有人-无人”这个任务上的性能

import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import scipy.spatial
import scipy
import cv2
from matplotlib import cm as CM
import random
import torchvision
import sys
from saver.checkpoints0_local_try1.net0_local import net0_local
import torch
import torchvision.transforms.functional as F
from saver.checkpoints0_local_try1.data_loader2 import ImageDataset
from torchvision import datasets, transforms


def train_val_test(image_dir, gt_dir, model, device, mode=None):
    if mode == 'train':
        print('analysis train...')
    elif mode == 'val':
        print('analysis val')
    elif mode == 'test':
        print('analysis test')
    else:
        print('error')

    dataset = ImageDataset(img_dir=image_dir,
                          gt_dir=gt_dir,
                          train=False,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                   std=[0.229, 0.224, 0.225])
                          ]),
                          )
    data_loader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=1,
                                             num_workers=1)
    model.eval()

    total = 0
    correct = 0
    person_exist_count = 0
    person_not_exist_count = 0
    class_correct = list(0 for p in range(2))
    for i, (img, gt_density_map, gt_label) in enumerate(data_loader):
        img = img.to(device)
        predict_value, predict_density_map = model(img)
        gt_label = gt_label.squeeze()

        _, binary_predict = torch.max(predict_value.data.cpu(), 1)     # binary_predict是predict里面每一行最大值的索引
        total += gt_label.size(0)                                      # 总图片块的数量
        correct += (binary_predict == gt_label.data.cpu()).sum().item() # 预测类别正确的图片块的数量
        temp_label1 = np.ones(8 * 8, dtype=int)
        temp_label2 = np.zeros(8 * 8, dtype=int)
        person_exist_count += (gt_label.data.cpu().numpy() == temp_label1).sum().item()    # 有人的图片块的数量
        person_not_exist_count += (gt_label.data.cpu().numpy() == temp_label2).sum().item()  # 没有人的图片块的数量
        c = gt_label.data.cpu().numpy() == binary_predict.numpy()
        for p in range(64):
            label = gt_label.data.cpu().numpy()[p]
            class_correct[label] += c[p]

    print('total:%d' % total)
    print('correct:%d' % correct)
    print('precision:%.6f' % (correct / total))
    print('person_exist_count:%d' % person_exist_count)
    print('person_exist_proportion:%.6f' % (person_exist_count / total))
    print('no_person_exist_count:%d' % person_not_exist_count)
    print('person_not_exist_proportion:%.6f' % (person_not_exist_count / total))
    print('person_exist_correct:%d' % class_correct[1])
    print('person_exist_correct_proportion:%.6f' % (class_correct[1] / person_exist_count))
    print('person_not_exist_correct:%d' % class_correct[0])
    print('person_not_exist_correct_proportion:%.6f' % (class_correct[0] / person_not_exist_count))
    print('\n')


if __name__ == '__main__':
    dataset = 'A'
    train_img_dir = '/home/rainkeeper/Projects/Datasets/shanghaiTech/processed_CSRNet_uncrop_data_gpu0/part_' + dataset + '/train_image'
    val_img_dir = '/home/rainkeeper/Projects/Datasets/shanghaiTech/processed_CSRNet_uncrop_data_gpu0/part_' + dataset + '/val_image'
    test_img_dir = '/home/rainkeeper/Projects/Datasets/shanghaiTech/processed_CSRNet_uncrop_data_gpu0/part_' + dataset + '/test_image'

    checkpoint_save_dir = '/home/rainkeeper/Projects/PycharmProjects/rain4/saver/checkpoints0_local_try1'
    checkpoint_name = 'Dataset_A252_72.6_102.38checkpoint.pth.tar'

    train_gpu_id = 'cuda:0'   # checkpoints是由哪个gpu生成的
    test_gpu_id = 'cuda:0'    # 当前analysis任务所用的gpu id

    device = torch.device(test_gpu_id if torch.cuda.is_available() else 'cpu')
    net = net0_local(load_weights=True)

    if train_gpu_id != test_gpu_id:
        checkpoint = torch.load(os.path.join(checkpoint_save_dir, checkpoint_name),
                                map_location={train_gpu_id: test_gpu_id})
    else:
        checkpoint = torch.load(os.path.join(checkpoint_save_dir, checkpoint_name))

    net.load_state_dict(checkpoint['state_dict'])
    net.to(device)
    net.eval()

    train_val_test(train_img_dir, train_img_dir.replace('.jpg', '.h5').replace('image', 'gt'), net, device, mode='train')
    train_val_test(val_img_dir, val_img_dir.replace('.jpg', '.h5').replace('image', 'gt'), net, device, mode='val')
    train_val_test(test_img_dir, test_img_dir.replace('.jpg', '.h5').replace('image', 'gt'), net, device, mode='test')


