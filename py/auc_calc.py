import torch
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt   

class AUCPlotter:
    def __init__(self, batch_size, test_raw_len, class_num = 9):
        self.batch_size = batch_size
        self.test_raw_len = test_raw_len
        self.class_num = class_num
        self.y_one_hot = np.zeros([test_raw_len, class_num])
        self.valxset = np.zeros([test_raw_len, 1, 64, 64])
        self.test_batch_num = test_raw_len // batch_size

    def setup(self, test_raw):
        for j, (bx, by) in enumerate(test_raw):
            bx = bx.numpy()
            self.y_one_hot[j, by] = 1
            self.valxset[j] = bx[0]

    def saveAucFig(self, net, batch_cnt):
        y_score = np.zeros([self.test_raw_len, self.class_num])
        for j in range(self.test_batch_num):
            bx = torch.from_numpy(self.valxset[j * self.batch_size:(j + 1) * self.batch_size]).float().cuda()
            y_score[j * self.batch_size:(j + 1) * self.batch_size] = net(bx).detach().cpu().numpy()
        auc = metrics.roc_auc_score(self.y_one_hot, y_score, average = 'micro')
        fpr, tpr, _ = metrics.roc_curve(self.y_one_hot.ravel(), y_score.ravel())
        
        plt.cla()
        plt.clf()
        plt.plot(fpr, tpr, c = 'r', lw = 2, alpha = 0.7, label = u'AUC=%.3f'% auc)
        plt.plot((0, 1), (0, 1), c = '#808080', lw = 1, ls = '--', alpha = 0.7)
        plt.xlim((-0.01, 1.02))
        plt.ylim((-0.01, 1.02))
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.xlabel('False Positive Rate', fontsize = 13)
        plt.ylabel('True Positive Rate', fontsize = 13)
        plt.grid(b = True, ls = ':')
        plt.legend(loc = 'lower right', fancybox = True, framealpha = 0.8, fontsize = 12)
        plt.title(u'工业缺陷分类后的ROC和AUC', fontsize=17)
        plt.savefig("../auc/auc_%d.png"%(batch_cnt))
        return auc
