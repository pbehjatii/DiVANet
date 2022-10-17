import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

import cv2
import numpy as np
from Attention_module import *


def init_weights(modules):
    pass


class MeanShift(nn.Module):
    def __init__(self, mean_rgb, sub):
        super(MeanShift, self).__init__()

        sign = -1 if sub else 1
        r = mean_rgb[0] * sign
        g = mean_rgb[1] * sign
        b = mean_rgb[2] * sign

        self.shifter = nn.Conv2d(3, 3, 1, 1, 0)
        self.shifter.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.shifter.bias.data   = torch.Tensor([r, g, b])

        # Freeze the mean shift layer
        for params in self.shifter.parameters():
            params.requires_grad = False

    def forward(self, x):
        x = self.shifter(x)
        return x


class UpsampleBlock(nn.Module):
    def __init__(self, n_channels, scale, multi_scale, wn, group=1):
        super(UpsampleBlock, self).__init__()

        if multi_scale:
            self.up2 = _UpsampleBlock(n_channels, scale=2, wn=wn, group=group)
            self.up3 = _UpsampleBlock(n_channels, scale=3, wn=wn, group=group)
            self.up4 = _UpsampleBlock(n_channels, scale=4, wn=wn, group=group)
        else:
            self.up = _UpsampleBlock(n_channels, scale=scale, wn=wn, group=group)

        self.multi_scale = multi_scale

    def forward(self, x, scale):
        if self.multi_scale:
            if scale == 2:
                return self.up2(x)
            elif scale == 3:
                return self.up3(x)
            elif scale == 4:
                return self.up4(x)
        else:
            return self.up(x)


class _UpsampleBlock(nn.Module):
    def __init__(self, n_channels, scale, wn,  group=1):
        super(_UpsampleBlock, self).__init__()

        modules = []

        if scale == 2 or scale == 4 or scale == 8:
            for _ in range(int(math.log(scale, 2))):
                modules += [wn(nn.Conv2d(n_channels, 4 * n_channels, 3, 1, 1, groups=group)),
                            nn.ReLU(inplace=True)]
                modules += [nn.PixelShuffle(2)]

        elif scale == 3:
            modules += [wn(nn.Conv2d(n_channels, 9 * n_channels, 3, 1, 1, groups=group)), nn.ReLU(inplace=True)]
            modules += [nn.PixelShuffle(3)]

        elif scale == 5:
            modules += [wn(nn.Conv2d(n_channels, 25 * n_channels, 3, 1, 1, groups=group)),nn.ReLU(inplace=True)]
            modules += [nn.PixelShuffle(5)]

        self.body = nn.Sequential(*modules)
        init_weights(self.modules)

    def forward(self, x):
        out = self.body(x)
        return out



class BasicConv2d(nn.Module):

    def __init__(self, wn, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = wn(nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=True)) # verify bias false

        self.LR = nn.ReLU(inplace=True)
        init_weights(self.modules)

    def forward(self, x):
        x = self.conv(x)
        x = self.LR(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self,
                 wn, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.DiVA = DiVA_attention()
        body = []
        expand = 6
        linear = 0.8
        body.append(
            wn(nn.Conv2d(64, 64*expand, 1, padding=1//2)))
        body.append(nn.ReLU(inplace=True))
        body.append(
            wn(nn.Conv2d(64*expand, int(64*linear), 1, padding=1//2)))
        body.append(
            wn(nn.Conv2d(int(64*linear), 64, 3, padding=3//2)))

        self.body = nn.Sequential(*body)

        init_weights(self.modules)

    def forward(self, x):

        out_x = self.body(x)
        out = out_x + x

        out_DiVA = self.DiVA(out_x)

        return out, out_DiVA
