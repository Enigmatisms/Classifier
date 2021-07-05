# -*- coding: utf-8 -*-
"""
    ResNet Classifiers
    @author 刘睿康
"""
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class Residual(nn.Module):
    def __init__(self, in_channel, out_channel, bottleneck_channel=None, stride=1):
        super(Residual, self).__init__()
        bottleneck_channel = in_channel // 2 if bottleneck_channel is None else bottleneck_channel

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channel, bottleneck_channel, 1, bias=False),
            nn.BatchNorm2d(bottleneck_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channel, bottleneck_channel, 3, stride, padding=1, bias=False),
            nn.BatchNorm2d(bottleneck_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channel, out_channel, 1, bias=False),
            nn.BatchNorm2d(out_channel),
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1, stride),
                nn.BatchNorm2d(out_channel),
        )

    def forward(self, x):
        out = self.bottleneck(x)
        identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    blk = nn.Sequential()
    for i in range(num_residuals):
        blk.add_module(str(i), Residual(in_channels, out_channels, stride=2 if i == 0 and not first_block else 1))
        in_channels = out_channels
    return blk

class ResNet(nn.Module):
    def __init__(self, in_channel, class_num, res_block_num):
        super(ResNet, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channel, 16, 7, stride=2, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        resnet_blocks = []
        in_channels = [16, 16, 32, 64]
        for i in range(res_block_num):
            resnet_blocks += [resnet_block(in_channels[i], in_channels[i + 1], 2, first_block=True if i == 0 else False)]
        self.resnet_blocks = nn.Sequential(*resnet_blocks)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, class_num)

    def forward(self, x):
        out = self.stem(x)
        out = self.resnet_blocks(out)
        out = self.avg_pool(out).view(-1, 64)
        out = self.fc(out)
        return out
