# 多任务，一个判断有无人(附加产物是attention map)，一个是生成密度图(本实验是attention map和主网络feature map相乘的方式)

##########################################
# 1、不使用0-1label，而是使用hard mask作为attention map
# 2、相比net7,从0-1mask变为-1,1mask
##########################################

import torch.nn as nn
import torch
from torchvision import models
from utils.utils import save_net, load_net
import sys
import math
import torch.nn.functional as F


class net8_local(nn.Module):
    def __init__(self, load_weights=True):
        super(net8_local, self).__init__()
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.frontend = make_layers(self.frontend_feat, dilation=False)
        self.auxiliary_backend_feat = [512, 512, 512, 256]
        self.auxiliary_backend = make_layers(self.auxiliary_backend_feat, in_channels=512, dilation=True)
        self.trunk_backend_feat = [512, 512, 512, 256]
        self.trunk_backend = make_layers(self.trunk_backend_feat, in_channels=512, dilation=True)

        self.auxiliary_backend_output_layer = nn.Conv2d(256, 1, kernel_size=1)

        self.density_map_layer = nn.Conv2d(256, 1, kernel_size=1)

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
        x_share = self.frontend(x)
        x_auxiliary_backend = self.auxiliary_backend(x_share)
        x_auxiliary_output_layer = self.auxiliary_backend_output_layer(x_auxiliary_backend)

        x_trunk = F.relu(x_share * x_auxiliary_output_layer)    # F.relu从物理上直接消除背景信息
        x_trunk_backend = self.trunk_backend(x_trunk)
        x_trunk_output = self.density_map_layer(x_trunk_backend)

        return x_auxiliary_output_layer, x_trunk_output

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
    for index, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                if index < 12:
                    layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


if __name__ == '__main__':
    net = pre_net0_local()
