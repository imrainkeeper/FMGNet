# trunk net系列是为了获得更好地base net

#####################################################
# try0 backend多层改进
#####################################################


import torch.nn as nn
import torch
from torchvision import models
from utils.utils import save_net, load_net
import sys
import math
import torch.nn.functional as F


class mp_1(nn.Module):
    def __init__(self, load_weights=True):
        super(mp_1, self).__init__()
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.frontend = make_layers(self.frontend_feat, dilation=False)

        self.branch1 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(512, 256, kernel_size=3, padding=1, stride=1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(256, 64, kernel_size=3, padding=1, stride=1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
                                     nn.ReLU(inplace=True))
        self.branch2 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(512, 256, kernel_size=3, padding=1, stride=1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(256, 256, kernel_size=3, padding=2, stride=1, dilation=2),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(256, 64, kernel_size=3, padding=2, stride=1, dilation=2),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
                                     nn.ReLU(inplace=True))
        self.branch3 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(512, 256, kernel_size=3, padding=2, stride=1, dilation=2),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(256, 256, kernel_size=3, padding=4, stride=1, dilation=4),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(256, 64, kernel_size=3, padding=2, stride=1, dilation=2),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
                                     nn.ReLU(inplace=True))

        self.output_layer = nn.Conv2d(192, 1, kernel_size=1)

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
        x0 = self.frontend(x)
        x1_1 = self.branch1(x0)
        x1_2 = self.branch2(x0)
        x1_3 = self.branch3(x0)
        x1 = torch.cat((x1_1, x1_2, x1_3), 1)

        x_out = self.output_layer(x1)
        return x_out

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
