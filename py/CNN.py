#-*-coding:utf-8-*-
"""
    CNN Classifiers
    @author 何千越
"""

import torch
from torch import nn     
class CNN(nn.Module):
    def __init__(self, class_out = 9):
        super().__init__()
        self.conv1 = nn.Sequential(     
            nn.Conv2d(1, 8, 3, padding = 1),
            nn.Dropout2d(0.2, True),
            nn.ReLU(True),                             
            nn.MaxPool2d(2)                       
        )       # out (n, 6, 32, 32)
        self.conv2 = nn.Sequential(     
            nn.Conv2d(8, 16, 5, padding = 2),
            nn.Dropout2d(0.2, True),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )       # out (n, 16, 16, 16)
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 64, 3, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.AvgPool2d(2)
        )       # out (n, 64, 8, 8)
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, 5, padding = 2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.AvgPool2d(2)
        )       # out (n, 64, 4, 4)
        self.fc = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.Dropout(0.2),
            nn.Sigmoid(),
            nn.Linear(64, class_out),
        )

    def forward(self, x):               
        x = self.conv1(x)              
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    # 只修改输出层
    def transferFromModel(self, model, out_name = "fc.4"):
        model_dict = self.state_dict()              # 当前网络参数
        state_dict = {k:v for k, v in model.items() if k in model_dict}    # 找出在当前网络中的参数
        weight_name = "%s.weight"%(out_name)
        bias_name = "%s.bias"%(out_name)
        state_dict.pop(weight_name)
        state_dict.pop(bias_name)
        weight_shape = model_dict[weight_name].shape
        bias_shape = model_dict[bias_name].shape
        state_dict[weight_name] = torch.normal(0, 1, weight_shape)
        state_dict[bias_name] = torch.normal(0, 1, bias_shape)
        model_dict.update(state_dict)
        self.load_state_dict(model_dict) 
        print("Model is transfered.")