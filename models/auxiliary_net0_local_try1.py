# 辅助网络

import torch.nn as nn
import torch
from torchvision import models
from utils.utils import save_net, load_net
import sys
import math
import torch.nn.functional as F


class single_SPPLayer(torch.nn.Module):  # 这里的single表示只进行一层分解，这和传统spp金字塔多层分解不同,此处可以算作是GAP的一个细化版
    def __init__(self, pool_type='avg_pool', split_coef=8):
        super(single_SPPLayer, self).__init__()
        self.pool_type = pool_type
        self.split_coef = split_coef

    def forward(self, x):
        num, c, h, w = x.size()  # num:样本数量 c:通道数 h:高 w:宽
        kernel_size = (math.ceil(h / self.split_coef), math.ceil(w / self.split_coef))
        stride = (math.ceil(h / self.split_coef), math.ceil(w / self.split_coef))
        if self.pool_type == 'max_pool':
            tensor = F.max_pool2d(x, kernel_size=kernel_size, stride=stride)
        else:
            tensor = F.avg_pool2d(x, kernel_size=kernel_size, stride=stride)  # tensor的size应该是(1, 1, 8, 8)
        tensor_reshape = tensor.squeeze().reshape(self.split_coef * self.split_coef, 1)  # (64, 1)
        tensor_reshape_reverse = 1 - tensor_reshape  # (64, 1)
        tensor_concat = torch.cat((tensor_reshape_reverse, tensor_reshape), 1)  # (64, 2)
        return tensor_concat


class auxiliary_net0_local(nn.Module):
    def __init__(self, load_weights=True):
        super(auxiliary_net0_local, self).__init__()
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.frontend = make_layers(self.frontend_feat, dilation=False)
        self.auxiliary_backend_feat = [512, 512, 512, 256, 128, 64]
        self.auxiliary_backend = make_layers(self.auxiliary_backend_feat, in_channels=512, dilation=True)
        self.auxiliary_backend_output_layer = nn.Conv2d(64, 1, kernel_size=1)

        self.single_spp_layer = single_SPPLayer(pool_type='avg_pool')

        if load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            for i in range(len(self.frontend.state_dict().items())):
                list(self.frontend.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]
            for k in self.frontend.children():
                for param in k.parameters():
                    param.requires_grad = False
        else:
            self._initialize_weights()

    def forward(self, x):
        x = self.frontend(x)
        x = self.auxiliary_backend(x)
        x = self.auxiliary_backend_output_layer(x)
        x = F.tanh(x)
        x = self.single_spp_layer(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


if __name__ == '__main__':
    net = pre_net0_local()
