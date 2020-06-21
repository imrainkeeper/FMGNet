#####################################################
# 新的mask map生成方式+简约trunk backend
#####################################################

import torch.nn as nn
import torch
from torchvision import models
from utils.utils import save_net, load_net
import sys
import math
import torch.nn.functional as F


class net10_local(nn.Module):
    def __init__(self, load_weights=True):
        super(net10_local, self).__init__()
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.frontend = make_layers(self.frontend_feat, dilation=False)
        # self.build_feature_layer = nn.Sequential(nn.Conv2d(512, 512, kernel_size=1, padding=0))   # 加上这一层是对固定vgg16提取的特征进行一个再加工，方便与auxiliary mask的结合

        self.auxiliary_backend_feat = ['M', 512, 512, 512, 'M', 512, 256]
        self.auxiliary_backend = make_layers(self.auxiliary_backend_feat, in_channels=512, dilation=False)
        self.auxiliary_backend_output_layer = nn.Conv2d(256, 1, kernel_size=1)

        self.trunk_backend = nn.Sequential(nn.Conv2d(768, 512, kernel_size=3, padding=1),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(512, 256, kernel_size=3, padding=1),
                                           nn.ReLU(inplace=True),
                                           )

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

        x_auxiliary = self.auxiliary_backend(x_share)
        x_auxiliary_mask_feature = F.upsample_bilinear(x_auxiliary, size=(128, 128))         # 利用256层的信息，因为auxiliary_backend_output_layer从256到1 channel可能会损失信息
        x_auxiliary = self.auxiliary_backend_output_layer(x_auxiliary)
        x_auxiliary_output = F.upsample_bilinear(x_auxiliary, size=(128, 128))  # 从32x32插值到128x128得到mask map

        # x_trunk = x_share * x_auxiliary_mask_feature
        x_trunk_combine = torch.cat((x_share, x_auxiliary_mask_feature), 1)         # 采用cat的方式累加feature
        x_trunk = self.trunk_backend(x_trunk_combine)
        x_trunk_output = self.density_map_layer(x_trunk)

        return x_auxiliary_output, x_trunk_output

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
