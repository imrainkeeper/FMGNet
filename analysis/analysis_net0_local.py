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
from saver.checkpoints1_local_try0.net1_local_temp import net1_local
import torch
import torchvision.transforms.functional as F
from data.single_image_loader2 import SingleImageDataset
from torchvision import datasets, transforms


def get_one_image(img_path):     # 返回dataset里面的那张image
    img_path = os.path.join(img_path)
    img = Image.open(img_path).convert('RGB')
    img = img.resize((1024, 1024), Image.ANTIALIAS)
    return img


def get_one_gt(gt_path):        # dataset中image大小和gt大小相同，为了和dataloader1_temp保持一致，将gt resize到128x128大小
    gt_file = h5py.File(gt_path, 'r')
    gt_density_map = np.asarray(gt_file['density'])

    original_gt_sum = np.sum(gt_density_map)
    gt_density_map = cv2.resize(gt_density_map, (128, 128), interpolation=cv2.INTER_CUBIC)
    current_gt_sum = np.sum(gt_density_map)
    gt_density_map = gt_density_map * (original_gt_sum / current_gt_sum)

    return gt_density_map


# 得到第一个任务中0-1任务得到的人群分布图以及最终的密度图
def get_images(img_path, gt_path, checkpoint_save_dir, checkpoint_name, train_gpu_id, test_gpu_id, transform=None):
    device = torch.device(test_gpu_id if torch.cuda.is_available() else 'cpu')
    if train_gpu_id != test_gpu_id:
        checkpoint = torch.load(os.path.join(checkpoint_save_dir, checkpoint_name), map_location={train_gpu_id:test_gpu_id})
    else:
        checkpoint = torch.load(os.path.join(checkpoint_save_dir, checkpoint_name))

    net = net1_local()
    net.load_state_dict(checkpoint['state_dict'])
    net.to(device)
    net.eval()

    dataset = SingleImageDataset(img_path=img_path,
                                 gt_path=gt_path,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])
                                 ]),
                                 )
    data_loader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=1, num_workers=1)

    for i, data in enumerate(data_loader, 0):
        image, gt_density_map = data
        image = image.to(device)
        predict_density_map, predict_value, predict_cam64_image = net(image)   # 分别是:第二个任务预测的density map, 第一个任务预测的有人-无人的value， 第一个任务对应的cam64的图

    # predict_data = predict.data.squeeze()
    # image = image.data.squeeze()
    # gt = gt.data.squeeze()
    # print('1', predict_data.shape)

    return predict_density_map.data.squeeze(), predict_cam64_image.data.squeeze()


def convert_one_img_path_2_gt_path(img_path):
    return img_path.replace('.jpg', '.h5').replace('image', 'gt')


def show_images(img_path, checkpoint_save_dir, checkpoint_name, train_gpu_id, test_gpu_id, transform=None):
    gt_path = convert_one_img_path_2_gt_path(img_path)
    img = get_one_image(img_path)
    gt = get_one_gt(gt_path)

    density_map, cam64_img = get_images(img_path, gt_path, checkpoint_save_dir, checkpoint_name, train_gpu_id, test_gpu_id, transform=None)

    plt.figure(figsize=(50, 50))

    plt.subplot(3, 3, 1)
    plt.title('img')
    plt.imshow(img)

    plt.subplot(3, 3, 2)
    plt.title('gt')
    plt.imshow(gt)

    plt.subplot(3, 3, 3)
    plt.title('cam64_0')
    plt.imshow(cam64_img[0])

    plt.subplot(3, 3, 4)
    plt.title('cam64_1')
    plt.imshow(cam64_img[1])

    plt.subplot(3, 3, 5)
    plt.title('cam64_add')
    plt.imshow(cam64_img[1] - cam64_img[0])

    plt.subplot(3, 3, 6)
    plt.title('density map')
    plt.imshow(density_map)

    plt.show()
    plt.close()


if __name__ == '__main__':
    dataset = 'A'
    test_img_dir = '/home/rainkeeper/Projects/Datasets/shanghaiTech/processed_CSRNet_uncrop_data_gpu0/part_' + dataset + '/test_image'

    test_img_name = 'IMG_110.jpg'

    checkpoint_save_dir = '/home/rainkeeper/Projects/PycharmProjects/rain4/saver/checkpoints1_local_try0/'
    checkpoint_name = 'Dataset_A108_69.4_90.03checkpoint.pth.tar'

    train_gpu_id = 'cuda:0'   # checkpoints是由哪个gpu生成的
    test_gpu_id = 'cuda:0'    # 当前analysis任务所用的gpu id

    test_img_path = os.path.join(test_img_dir, test_img_name)
    show_images(test_img_path, checkpoint_save_dir, checkpoint_name, train_gpu_id, test_gpu_id, transform=None)
