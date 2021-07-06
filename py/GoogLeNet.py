# -*- coding: utf-8 -*-
"""
GoogLeNet implementation
@author: 刘睿康
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class Inceptionv1(nn.Module):
    def __init__(self, in_dim, c1, c2, c3, c4):
        super(Inceptionv1, self).__init__()
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_dim, c1, 1),
            nn.ReLU(inplace=True)
        )
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_dim, c2[0], 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2[0], c2[1], 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_dim, c3[0], 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3[0], c3[1], 5, padding=2),
            nn.ReLU(inplace=True)
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_dim, c4, 1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b1 = self.branch1x1(x)
        b2 = self.branch3x3(x)
        b3 = self.branch5x5(x)
        b4 = self.branch_pool(x)
        output = torch.cat((b1, b2, b3, b4), dim=1)
        return output

class GoogLenetv1(nn.Module):
    def __init__(self, in_dim, class_num):
        super(GoogLenetv1, self).__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(64, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.b3 = nn.Sequential(
            Inceptionv1(192, 64, (96, 128), (16, 32), 32),
            Inceptionv1(256, 128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.b4 = nn.Sequential(
            Inceptionv1(480, 192, (96, 208), (16, 48), 64),
            Inceptionv1(512, 160, (112, 224), (24, 64), 64),
            Inceptionv1(512, 128, (128, 256), (24, 64), 64),
            Inceptionv1(512, 112, (144, 288), (32, 64), 64),
            Inceptionv1(528, 256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.b5 = nn.Sequential(
            Inceptionv1(832, 256, (160, 320), (32, 128), 128),
            Inceptionv1(832, 384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d(1),
        )
        self.net = nn.Sequential()
        for i in range(1, 6):
            exec('self.net.add_module(str(i), self.b' + str(i) + ')')
        self.fc = nn.Linear(1024, class_num)
        
    def forward(self, x):
        output = self.net(x)
        output = output.view(-1, 1024) 
        output = self.fc(output)
        
        return output


if __name__ == '__main__':
    pass


