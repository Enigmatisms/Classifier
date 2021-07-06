# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 13:13:18 2021
@author: 占炎根
"""
import torch.nn as nn

class ResFcl(nn.Module):
    def __init__(self, dim, dr=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Linear(dim, dim),
        )
    def forward(self, x):
        return x + self.block(x)

class ResConv(nn.Module):
    def __init__(self, dim, dr=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)

class ResCNN(nn.Module):
    def __init__(self, class_out = 9, dr=0):
        super().__init__()
        self.conv = nn.Sequential(     
            nn.Conv2d(1, 32, 4, 2, 1),    # (64-4+2*1)/2+1=32, out n*32*32*32
            nn.BatchNorm2d(32),
            ResConv(32, dr),
            ResConv(32, dr),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Conv2d(32, 1, 3, 1, 1),    # (32-3+2*1)/1+1=32, out n*1*32*32
            )
        self.fcl = nn.Sequential(
            nn.Linear(1024, 1024),
            ResFcl(1024, dr),
            ResFcl(1024, dr),
            nn.ReLU(),
            nn.Dropout(dr),
            )
        self.out = nn.Sequential(
            nn.Linear(1024, class_out),
            nn.Softmax(),
            )
    def forward(self, input):
        z=self.conv(input)
        z=z.view(-1,1024)
        out_fcl=self.fcl(z)
        pred=self.out(out_fcl)
        return pred
            
            
            
            
            
            
            
