from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import with_statement
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import os




def mean_channels_h(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True)
    return spatial_sum / F.size(3)

def stdv_channels_h(F):
    assert(F.dim() == 4)
    F_mean = mean_channels_h(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True) / F.size(3)
    return F_variance


def mean_channels_w(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(2, keepdim=True)
    return spatial_sum / F.size(2)

def stdv_channels_w(F):
    assert(F.dim() == 4)
    F_mean = mean_channels_w(F)
    F_variance = (F - F_mean).pow(2).sum(2, keepdim=True) / F.size(2)
    return F_variance



class DiVA_attention(nn.Module):
    def __init__(self):
        super(DiVA_attention, self).__init__()


        self.contrast_h = stdv_channels_h
        self.contrast_w = stdv_channels_w

        self.conv_h = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        identity = x

        n,c,h,w = x.size()

        c_h = self.contrast_h(x)
        c_w = self.contrast_w(x)

        a_h = self.conv_h(c_h).sigmoid()
        a_w = self.conv_w(c_w).sigmoid()

        out = identity * a_w * a_h

        return out
