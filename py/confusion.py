# -*- coding: utf-8 -*-
"""
    Confusion Matrix绘制
    @author 何千越
"""

import matplotlib.pyplot as plt
import numpy as np

class ConfusionMatrix:
    def __init__(self, class_num):
        self.mat = np.zeros((class_num, class_num))
        self.class_num = class_num

    def addElement(self, truth, pred):
        self.mat[truth, pred] += 1

    def saveConfusionMatrix(self, path):
        plt.cla()
        plt.clf()
        plt.imshow(self.mat, cmap = 'inferno')
        plt.colorbar()
        plt.xlabel("Prediction result")
        plt.ylabel("Ground truth")
        plt.savefig(path)
