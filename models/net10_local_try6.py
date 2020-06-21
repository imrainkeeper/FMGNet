#####################################################
# 使用deconv代替bilinear upsampling
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

        self.auxiliary_upsample1 = nn.Sequential(nn.UpsamplingBilinear2d(size=(64, 64)),
                                                 nn.Conv2d(256, 256, kernel_size=1, padding=0),
                                                 nn.ReLU(inplace=True))
        self.auxiliary_upsample2 = nn.Sequential(nn.UpsamplingBilinear2d(size=(128, 128)),
                                                 nn.Conv2d(256, 256, kernel_size=1, padding=0),
                                                 nn.ReLU(inplace=True))

        self.auxiliary_backend_output_layer = nn.Conv2d(256, 1, kernel_size=1)

        self.trunk_backend1 = nn.Sequential(nn.Conv2d(768, 512, kernel_size=3, padding=1, stride=1),
                                            nn.ReLU(inplace=True))
        self.trunk_backend2_1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1, stride=1),
                                              nn.ReLU(inplace=True))
        self.trunk_backend2_2 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=2, stride=1, dilation=2),
                                              nn.ReLU(inplace=True))
        self.trunk_backend3_1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1, stride=1),
                                              nn.ReLU(inplace=True))
        self.trunk_backend3_2 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=2, stride=1, dilation=2),
                                              nn.ReLU(inplace=True))
        self.trunk_backend3_3 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=4, stride=1, dilation=4),
                                              nn.ReLU(inplace=True))
        self.trunk_backend4_1 = nn.Sequential(nn.Conv2d(768, 256, kernel_size=3, padding=1, stride=1),
                                              nn.ReLU(inplace=True))
        self.trunk_backend4_2 = nn.Sequential(nn.Conv2d(768, 256, kernel_size=3, padding=2, stride=1, dilation=2),
                                              nn.ReLU(inplace=True))
        self.trunk_backend5 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1, stride=1),
                                            nn.ReLU(inplace=True))

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
        x_auxiliary_mask_feature = self.auxiliary_upsample2(self.auxiliary_upsample1(x_auxiliary))   # 使用deconv
        x_auxiliary_output = self.auxiliary_backend_output_layer(x_auxiliary_mask_feature)

        x_trunk = torch.cat((x_share, x_auxiliary_mask_feature), 1)         # 采用cat的方式累加feature
        x_trunk1 = self.trunk_backend1(x_trunk)
        x_trunk2_1 = self.trunk_backend2_1(x_trunk1)
        x_trunk2_2 = self.trunk_backend2_2(x_trunk1)
        x_trunk2 = torch.cat((x_trunk2_1, x_trunk2_2), 1)
        x_trunk3_1 = self.trunk_backend3_1(x_trunk2)
        x_trunk3_2 = self.trunk_backend3_2(x_trunk2)
        x_trunk3_3 = self.trunk_backend3_3(x_trunk2)
        x_trunk3 = torch.cat((x_trunk3_1, x_trunk3_2, x_trunk3_3), 1)
        x_trunk4_1 = self.trunk_backend4_1(x_trunk3)
        x_trunk4_2 = self.trunk_backend4_2(x_trunk3)
        x_trunk4 = torch.cat((x_trunk4_1, x_trunk4_2), 1)
        x_trunk5 = self.trunk_backend5(x_trunk4)
        x_trunk_output = self.density_map_layer(x_trunk5)

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
