#####################################################
# try3 backend多层改进
#####################################################


import torch.nn as nn
import torch
from torchvision import models
from utils.utils import save_net, load_net
import sys
import math
import torch.nn.functional as F


class net9_local(nn.Module):
    def __init__(self, load_weights=True):
        super(net9_local, self).__init__()
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.frontend = make_layers(self.frontend_feat, dilation=False)
        self.build_feature_layer = nn.Sequential(nn.Conv2d(512, 512, kernel_size=1, padding=0))   # 加上这一层是对固定vgg16提取的特征进行一个再加工，方便与auxiliary mask的结合

        self.auxiliary_backend1 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),
                                                nn.ReLU(inplace=True))
        self.auxiliary_backend2_1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1, stride=1),
                                                  nn.ReLU(inplace=True))
        self.auxiliary_backend2_2 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=2, stride=1, dilation=2),
                                                  nn.ReLU(inplace=True))
        self.auxiliary_backend3_1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1, stride=1),
                                                  nn.ReLU(inplace=True))
        self.auxiliary_backend3_2 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=2, stride=1, dilation=2),
                                                  nn.ReLU(inplace=True))
        self.auxiliary_backend3_3 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=4, stride=1, dilation=4),
                                                  nn.ReLU(inplace=True))
        self.auxiliary_backend4_1 = nn.Sequential(nn.Conv2d(768, 256, kernel_size=3, padding=1, stride=1),
                                                  nn.ReLU(inplace=True))
        self.auxiliary_backend4_2 = nn.Sequential(nn.Conv2d(768, 256, kernel_size=3, padding=2, stride=1, dilation=2),
                                                  nn.ReLU(inplace=True))
        self.auxiliary_backend5 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1, stride=1),
                                                nn.ReLU(inplace=True))
        self.auxiliary_backend_output_layer = nn.Sequential(nn.Conv2d(256, 1, kernel_size=1),
                                                            nn.ReLU(inplace=True))

        self.trunk_backend1 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),
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

        x_auxiliary1 = self.auxiliary_backend1(x_share)
        x_auxiliary2_1 = self.auxiliary_backend2_1(x_auxiliary1)
        x_auxiliary2_2 = self.auxiliary_backend2_2(x_auxiliary1)
        x_auxiliary2 = torch.cat((x_auxiliary2_1, x_auxiliary2_2), 1)
        x_auxiliary3_1 = self.auxiliary_backend3_1(x_auxiliary2)
        x_auxiliary3_2 = self.auxiliary_backend3_2(x_auxiliary2)
        x_auxiliary3_3 = self.auxiliary_backend3_3(x_auxiliary2)
        x_auxiliary3 = torch.cat((x_auxiliary3_1, x_auxiliary3_2, x_auxiliary3_3), 1)
        x_auxiliary4_1 = self.auxiliary_backend4_1(x_auxiliary3)
        x_auxiliary4_2 = self.auxiliary_backend4_2(x_auxiliary3)
        x_auxiliary4 = torch.cat((x_auxiliary4_1, x_auxiliary4_2), 1)
        x_auxiliary5 = self.auxiliary_backend5(x_auxiliary4)
        x_auxiliary_output = self.auxiliary_backend_output_layer(x_auxiliary5)   # mask map

        x_build_feature = self.build_feature_layer(x_share)
        x_trunk = x_build_feature * x_auxiliary_output

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


class trunk_net2_local(nn.Module):
    def __init__(self, load_weights=True):
        super(trunk_net2_local, self).__init__()
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.frontend = make_layers(self.frontend_feat, dilation=False)

        self.backend1 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),
                                      nn.ReLU(inplace=True))
        self.backend2_1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1, stride=1),
                                        nn.ReLU(inplace=True))
        self.backend2_2 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=2, stride=1, dilation=2),
                                        nn.ReLU(inplace=True))
        self.backend3_1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1, stride=1),
                                        nn.ReLU(inplace=True))
        self.backend3_2 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=2, stride=1, dilation=2),
                                        nn.ReLU(inplace=True))
        self.backend3_3 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=4, stride=1, dilation=4),
                                        nn.ReLU(inplace=True))
        self.backend4_1 = nn.Sequential(nn.Conv2d(768, 256, kernel_size=3, padding=1, stride=1),
                                        nn.ReLU(inplace=True))
        self.backend4_2 = nn.Sequential(nn.Conv2d(768, 256, kernel_size=3, padding=2, stride=1, dilation=2),
                                        nn.ReLU(inplace=True))
        self.backend5 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1, stride=1),
                                        nn.ReLU(inplace=True))
        self.output_layer = nn.Conv2d(256, 1, kernel_size=1)

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
        x1 = self.backend1(x0)

        x2_1 = self.backend2_1(x1)
        x2_2 = self.backend2_2(x1)
        x2 = torch.cat((x2_1, x2_2), 1)

        x3_1 = self.backend3_1(x2)
        x3_2 = self.backend3_2(x2)
        x3_3 = self.backend3_3(x2)
        x3 = torch.cat((x3_1, x3_2, x3_3), 1)

        x4_1 = self.backend4_1(x3)
        x4_2 = self.backend4_2(x3)
        x4 = torch.cat((x4_1, x4_2), 1)

        x5 = self.backend5(x4)

        x_out = self.output_layer(x5)
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
