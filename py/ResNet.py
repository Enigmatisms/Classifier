# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import models
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import json
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import h5py

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

class MyDataset(Dataset):
    def __init__(self, img_size=64, data_path='./data/trainset1.mat', transforms=None):
        # 初始化，读取数据集
        self.data_path = data_path
        self.transforms = transforms
        self.file = h5py.File(data_path, 'r')
        self.xset = torch.from_numpy(self.file['xset'][()]).view(-1, 1, img_size, img_size)
        self.label = torch.from_numpy(self.file['label'][()]).view(-1).long()
        # idlist 中有十个元素，表示 xset 和 label 中从 idlist[i] 到 idlist[i + 1] 个元素是第 i 个分类
        self.idlist = torch.from_numpy(self.file['idlist'][()]).view(-1).long()
        
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):
        img = self.xset[index]
        annotation = self.label[index]
        if self.transforms:
            img = self.transforms(img)
            
        return img, annotation

if __name__ == '__main__':
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print('Using device:', device_str)

    transform=transforms.Compose([
                transforms.ToPILImage(),
                # transforms.Resize(200),               # 将图像最短边 resize 至 size 大小，宽高比例不变
                transforms.RandomHorizontalFlip(),      # 以 0.5 的概率左右翻转图像
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(degrees=5, expand=False, fill=None),
                transforms.ToTensor(),                  # 将 PIL 图像转为 Tensor，并且进行归一化
                transforms.Normalize([0.5], [0.5]) # 进行 mean 与 std 为 0.5 的标准化
            ])

    trainset = MyDataset(img_size=64, data_path='./data/trainset1.mat', transforms=transform)
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True)# , num_workers=2) # num_workers表示使用几个线程来加载数据
  
    testset = MyDataset(img_size=64, data_path='./data/valset1.mat', transforms=transform)
    testloader = DataLoader(testset, batch_size=4, shuffle=True)# , num_workers=2) # num_workers表示使用几个线程来加载数据

    print('Size of train set:', len(trainset))
    print('Size of test set:', len(testset))

    net = ResNet(in_channel=1, class_num=10)
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    # 将全连接层的学习率设为其他层学习率的十倍
    lr = 0.001 / 10
    fc_params = list(map(id, net.fc.parameters())) # 取得全连接层的参数内存地址的列表
    base_params = filter(lambda p: id(p) not in fc_params, net.parameters()) # 取得其他层参数的列表
    optimizer = optim.Adam([
                {'params': base_params},
                {'params': net.fc.parameters(), 'lr': lr * 10}],
                lr=lr, betas=(0.9, 0.999))

    # 迁移学习 / 继续训练
    # save = torch.load('./model/res_test.pth')   # 保存的优化器以及模型参数
    # save_model = save['model']                  # 保存的模型参数
    # model_dict =  net.state_dict()              # 当前网络参数
    # state_dict = {k:v for k, v in save_model.items() if k in model_dict}    # 找出在当前网络中的参数
    # model_dict.update(state_dict)
    # net.load_state_dict(model_dict) 

    check_freq = 100
    eval_freq = 500
    n_epochs = 100
    best_acc = 0
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_loss = 0.0
        for i, data in enumerate(trainloader):
            # get the inputs
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_loss += loss.item()
            if (i + 1) % check_freq == 0:
                print('[%d, %5d] loss: %.3f' %
                    (epoch, i + 1, running_loss / check_freq))
                
                with SummaryWriter('runs/exp-1') as w:
                    w.add_scalar('TrainLoss/epoch' + str(epoch), running_loss / check_freq, i // check_freq)             
                running_loss = 0.0

                print('Current Model saved!')
                torch.save({
                    'model': net.state_dict(),
                    'optimizer': optimizer.state_dict()},
                    './model/resnet.pth')
            
            # evaluate
            if (i + 1) % eval_freq == 0:
                correct = 0
                total = 0
                with torch.no_grad():
                    for data in testloader:
                        images, labels = data
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                acc = 100 * correct / total
                if acc > best_acc:
                    best_acc = acc
                    print('New best model!')
                    print('New Best Acc: %.2f %%'%best_acc)
                    torch.save({
                        'model': net.state_dict(),
                        'optimizer': optimizer.state_dict()},
                        './model/resnet_best_model.pth')
                
        with SummaryWriter('runs/exp-1') as w:
            w.add_scalar('TrainLoss/all', epoch_loss / len(trainloader), epoch)
            epoch_loss = 0.0
    
    print('Finished Training')

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test set: %.2f %%' % (
        100 * correct / total))

