#################################################
# 更换辅助网络中0-1label方式为hard mask方式， boundary=1e-2
#################################################

import sys
import os
import warnings
from models.net10_local_try3 import net10_local
from utils.utils import save_checkpoint
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
import argparse
import json
import cv2
from data.data_loader3 import ImageDataset
import time

parser = argparse.ArgumentParser(description='PyTorch rain4')
parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None, type=str, help='path to the pretrained model')
args = parser.parse_args()
args.best_precision = 0
args.best_mae = sys.maxsize
args.best_mse = sys.maxsize
args.best_epoch = -1
args.boundary = 1e-2
checkpoint_save_dir = '/home/njuciairs/rainkeeper/Projects/PycharmProjects/rain4/checkpoints10_local_try3_1e-2'
terminal_log_file = os.path.join(checkpoint_save_dir, 'terminal_log.txt')
terminal_file = open(terminal_log_file, 'a')


def main():
    args.original_lr = 1 * (1e-5)
    args.lr = 1 * (1e-5)
    args.batch_size = 1
    args.momentum = 0.95
    args.decay = 5 * 1e-4
    args.start_epoch = 0
    args.epochs = 1000
    args.workers = 1
    args.seed = time.time()
    args.print_freq = 400
    args.gpu_id = "cuda:1"
    args.dataset = 'A'

    train_image_dir = '/home/njuciairs/rainkeeper/Projects/Datasets/shanghaiTech/processed_CSRNet_uncrop_data_gpu1/part_' + args.dataset + '/train_image'
    train_gt_dir = '/home/njuciairs/rainkeeper/Projects/Datasets/shanghaiTech/processed_CSRNet_uncrop_data_gpu1/part_' + args.dataset + '/train_gt'
    val_image_dir = '/home/njuciairs/rainkeeper/Projects/Datasets/shanghaiTech/processed_CSRNet_uncrop_data_gpu1/part_' + args.dataset + '/val_image'
    val_gt_dir = '/home/njuciairs/rainkeeper/Projects/Datasets/shanghaiTech/processed_CSRNet_uncrop_data_gpu1/part_' + args.dataset + '/val_gt'
    test_image_dir = '/home/njuciairs/rainkeeper/Projects/Datasets/shanghaiTech/processed_CSRNet_uncrop_data_gpu1/part_' + args.dataset + '/test_image'
    test_gt_dir = '/home/njuciairs/rainkeeper/Projects/Datasets/shanghaiTech/processed_CSRNet_uncrop_data_gpu1/part_' + args.dataset + '/test_gt'

    args.device = torch.device(args.gpu_id if torch.cuda.is_available() else "cpu")
    torch.cuda.manual_seed(args.seed)

    model = net10_local(load_weights=True)
    model.to(args.device)

    criterion1 = nn.MSELoss(size_average=False).to(args.device)
    criterion2 = nn.MSELoss(size_average=False).to(args.device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # if args.pre:
    if False:
        if os.path.isfile(args.pre):
            print("=> loading checkpoint '{}'".format(args.pre))
            checkpoint = torch.load(args.pre)
            args.start_epoch = checkpoint['epoch']
            args.best_mae = checkpoint['best_mae']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.pre, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pre))

    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
        train_mae, train_mse, train_gt_sum, train_predict_sum = train(train_image_dir, train_gt_dir, model, criterion1, criterion2, optimizer, epoch)
        val_mae, val_mse, val_gt_sum, val_predict_sum = validate(val_image_dir, val_gt_dir, model)
        test_mae, test_mse, test_gt_sum, test_predict_sum = sssss(test_image_dir, test_gt_dir, model)

        is_best = (test_mae <= args.best_mae)
        if is_best:
            args.best_mae = test_mae
            args.best_epoch = epoch

        print(
            'current train mae: %.6f, current train mse: %.6f, current_train_gt_sum: %.6f, current_train_predict_sum:%.6f' % (
                train_mae, train_mse, train_gt_sum, train_predict_sum))
        print(
            'current train mae: %.6f, current train mse: %.6f, current_train_gt_sum: %.6f, current_train_predict_sum:%.6f' % (
                train_mae, train_mse, train_gt_sum, train_predict_sum), file=terminal_file)

        print('current val mae: %.6f, current val mse: %.6f, current_gt_sum: %.6f, current_predict_sum:%.6f' % (
            val_mae, val_mse, val_gt_sum, val_predict_sum))
        print('current val mae: %.6f, current val mse: %.6f, current_gt_sum: %.6f, current_predict_sum:%.6f' % (
            val_mae, val_mse, val_gt_sum, val_predict_sum), file=terminal_file)

        print(
            'current test mae: %.6f, current test mse: %.6f, current_test_gt_sum: %.6f, current_test_predict_sum:%.6f' % (
                test_mae, test_mse, test_gt_sum, test_predict_sum))
        print(
            'current test mae: %.6f, current test mse: %.6f, current_test_gt_sum: %.6f, current_test_predict_sum:%.6f' % (
                test_mae, test_mse, test_gt_sum, test_predict_sum), file=terminal_file)

        print('best test mae: %.6f, best test mse: %.6f' % (args.best_mae, args.best_mse))
        print('best test mae: %.6f, best test mse: %.6f' % (args.best_mae, args.best_mse), file=terminal_file)

        print('best epoch:%d' % args.best_epoch)
        print('best epoch:%d' % args.best_epoch, file=terminal_file)

        if is_best:
            val_mae_2f = round(val_mae, 2)
            test_mae_2f = round(test_mae, 2)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.pre,
                'state_dict': model.state_dict(),
                'current_mae': test_mae,
                'best_mae': args.best_mae,
                'optimizer': optimizer.state_dict()
            }, checkpoint_save_dir, args.dataset, epoch, test_mae_2f, val_mae_2f)


def train(train_image_dir, train_gt_dir, model, criterion1, criterion2, optimizer, epoch):
    losses = AverageMeter()
    losses1 = AverageMeter()
    losses2 = AverageMeter()

    trainset = ImageDataset(img_dir=train_image_dir,
                            gt_dir=train_gt_dir,
                            train=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])
                            ]),
                            boundary=args.boundary,
                            )
    train_loader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=args.batch_size,
                                               num_workers=args.workers)
    print('epoch %d, processed %d samples, dataset %s, lr %.10f' % (
        epoch, epoch * len(train_loader.dataset), args.dataset, args.lr))
    print('epoch %d, processed %d samples, dataset %s, lr %.10f' % (
        epoch, epoch * len(train_loader.dataset), args.dataset, args.lr), file=terminal_file)

    model.train()
    end = time.time()

    train_mae = 0.0
    train_mse = 0.0
    train_gt_sum = 0.0
    train_predict_sum = 0.0
    for i, (img, gt_density_map, gt_attention_map) in enumerate(train_loader):
        img = img.to(args.device)
        predict_attention_map, predict_density_map = model(img)  # predict的shape为[64, 2]

        gt_attention_map = gt_attention_map.to(args.device)
        gt_density_map = gt_density_map.to(args.device)

        loss1 = criterion1(predict_attention_map, gt_attention_map)
        loss2 = criterion2(predict_density_map, gt_density_map)
        loss = loss1 + loss2

        losses.update(loss.item(), img.size(0))
        losses1.update(loss1.item(), img.size(0))
        losses2.update(loss2.item(), img.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Loss1 {loss1.val:.4f} ({loss1.avg:.4f})\t'
                  'Loss2 {loss1.val:.4f} ({loss2.avg:.4f})\t'
                .format(
                epoch, i, len(train_loader), loss=losses, loss1=losses1, loss2=losses2), file=terminal_file)
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Loss1 {loss1.val:.4f} ({loss1.avg:.4f})\t'
                  'Loss2 {loss2.val:.4f} ({loss2.avg:.4f})\t'
                .format(
                epoch, i, len(train_loader), loss=losses, loss1=losses1, loss2=losses2))

        train_gt_count = np.sum(gt_density_map.data.cpu().numpy())
        train_predict_count = np.sum(predict_density_map.data.cpu().numpy())
        train_mae += abs(train_gt_count - train_predict_count)
        train_mse += (train_gt_count - train_predict_count) * (train_gt_count - train_predict_count)
        train_gt_sum += train_gt_count
        train_predict_sum += train_predict_count
    train_mae = train_mae / len(train_loader.dataset)
    train_mse = np.sqrt(train_mse / len(train_loader.dataset))
    train_gt_sum = train_gt_sum / len(train_loader.dataset)
    train_predict_sum = train_predict_sum / len(train_loader.dataset)

    return train_mae, train_mse, train_gt_sum, train_predict_sum


def validate(val_image_dir, val_gt_dir, model):
    print('begin validate')
    print('begin validate', file=terminal_file)
    valset = ImageDataset(img_dir=val_image_dir,
                          gt_dir=val_gt_dir,
                          train=False,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                   std=[0.229, 0.224, 0.225])
                          ]),
                          boundary=args.boundary,
                          )
    val_loader = torch.utils.data.DataLoader(valset, shuffle=False, batch_size=args.batch_size,
                                             num_workers=args.workers)
    model.eval()

    mae = 0
    mse = 0
    gt_sum = 0
    predict_sum = 0
    for i, (img, gt_density_map, gt_attention_map) in enumerate(val_loader):
        img = img.to(args.device)
        predict_attention_map, predict_density_map = model(img)
        gt_attention_map = gt_attention_map.to(args.device)
        gt_density_map = gt_density_map.to(args.device)

        gt_count = np.sum(gt_density_map.data.cpu().numpy())
        predict_count = np.sum(predict_density_map.data.cpu().numpy())
        mae += abs(gt_count - predict_count)
        mse += ((gt_count - predict_count) * (gt_count - predict_count))
        gt_sum += gt_count
        predict_sum += predict_count
    mae = mae / len(val_loader.dataset)
    mse = np.sqrt(mse / len(val_loader.dataset))
    gt_sum = gt_sum / len(val_loader.dataset)
    predict_sum = predict_sum / len(val_loader.dataset)

    return mae, mse, gt_sum, predict_sum


def sssss(test_image_dir, test_gt_dir, model):
    print('begin test')
    print('begin test', file=terminal_file)
    testset = ImageDataset(img_dir=test_image_dir,
                           gt_dir=test_gt_dir,
                           train=False,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                           ]),
                           boundary=args.boundary,
                           )
    test_loader = torch.utils.data.DataLoader(testset, shuffle=False, batch_size=args.batch_size,
                                              num_workers=args.workers)
    model.eval()

    mae = 0
    mse = 0
    gt_sum = 0
    predict_sum = 0
    for i, (img, gt_density_map, gt_attention_map) in enumerate(test_loader):
        img = img.to(args.device)
        predict_attention_map, predict_density_map = model(img)
        gt_attention_map = gt_attention_map.to(args.device)
        gt_density_map = gt_density_map.to(args.device)

        gt_count = np.sum(gt_density_map.data.cpu().numpy())
        predict_count = np.sum(predict_density_map.data.cpu().numpy())
        mae += abs(gt_count - predict_count)
        mse += ((gt_count - predict_count) * (gt_count - predict_count))
        gt_sum += gt_count
        predict_sum += predict_count
    mae = mae / len(test_loader.dataset)
    mse = np.sqrt(mse / len(test_loader.dataset))
    gt_sum = gt_sum / len(test_loader.dataset)
    predict_sum = predict_sum / len(test_loader.dataset)

    return mae, mse, gt_sum, predict_sum


def adjust_learning_rate(optimizer, epoch):
    args.lr = args.original_lr
    for i in range(len(args.steps)):
        scale = args.scales[i] if i < len(args.scales) else 1
        if epoch >= args.steps[i]:
            args.lr = args.lr * scale
            if epoch == args.steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr


class AverageMeter(object):
    def __init__(self):
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
