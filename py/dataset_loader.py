# -*- coding: utf-8 -*-
"""
    mat加载器
    @author 刘睿康
"""

import h5py
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, img_size=64, data_path = './data/trainset1.mat', transforms = None):
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