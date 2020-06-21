# 辅助网络

#################################################
# try0   1、利用更多的pooling而不是dilated convolution(避免空洞效应)   2、只对最后的那张mask map进行upsample，从32x32upsample到128x128
# try2   对前面layer的mask feature maps进行upsample，然后再利用卷积进行处理
#################################################


import torch.nn as nn
import torch
from torchvision import models
from utils.utils import save_net, load_net
import sys
import math
import torch.nn.functional as F


class auxiliary_net2_local(nn.Module):
    def __init__(self, load_weights=True):
        super(auxiliary_net2_local, self).__init__()
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.frontend = make_layers(self.frontend_feat, dilation=False)

        self.auxiliary_backend_feat = ['M', 512, 512, 512, 'M', 512, 256]
        self.auxiliary_backend = make_layers(self.auxiliary_backend_feat, in_channels=512, dilation=False)

        self.auxiliary_upsample1 = nn.Sequential(nn.UpsamplingBilinear2d(size=(64, 64)),
                                                 nn.Conv2d(256, 256, kernel_size=1, padding=0),
                                                 nn.ReLU(inplace=True))
        self.auxiliary_upsample2 = nn.Sequential(nn.UpsamplingBilinear2d(size=(128, 128)),
                                                 nn.Conv2d(256, 256, kernel_size=1, padding=0),
                                                 nn.ReLU(inplace=True))

        self.auxiliary_backend_output_layer = nn.Conv2d(256, 1, kernel_size=1)

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
        x = self.auxiliary_upsample1(x)
        x = self.auxiliary_upsample2(x)
        x = self.auxiliary_backend_output_layer(x)
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
