#-*-coding:utf-8-*-
"""
    MNIST 手写数字数据集分类的CNN实现
    基于我二月份某个代码以及另一篇论文复现时的代码
    - LeNet
"""
import os
import shutil
import argparse
from datetime import datetime
import torch
import random
from torch import nn                                
from torchvision import datasets                    
from torch.utils.data.dataloader import DataLoader 
from torch import optim                             
from torchvision import transforms     
from torch.utils.tensorboard import SummaryWriter
from ResNet import ResNet

name_list = ["cr", "gg", "in", "pa", "ps", "rp", "rs", "sc", "sp"]

def getLoader(_tf, path, shuffle = False, eval = False):
    if eval == False:
        folder = datasets.ImageFolder("..\\train\\%s\\"%(path), transform = _tf)
    else:
        folder = datasets.ImageFolder("..\\val\\%s\\"%(path), transform = _tf)
    loader = DataLoader(folder, shuffle = shuffle)
    return loader

def makeBatch(samples):
    img_batch = []
    y_batch = []
    for img, y in samples:
        img_batch.append(img)
        y_batch.append(y)
    return torch.cat(img_batch, dim = 0), torch.LongTensor(y_batch).cuda()

class CNN(nn.Module):
    def __init__(self, class_out = 9):
        super().__init__()
        self.conv1 = nn.Sequential(     
            nn.Conv2d(1, 8, 3, padding = 1),    
            nn.ReLU(True),                             
            nn.MaxPool2d(2)                       
        )       # out (n, 6, 32, 32)
        self.conv2 = nn.Sequential(     
            nn.Conv2d(8, 16, 5, padding = 2),
            nn.BatchNorm2d(16),  
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
        self.out = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(True),
            nn.Linear(256, 64),
            # nn.Dropout(0.1),
            nn.Sigmoid(),
            nn.Linear(64, class_out),
        )

    def forward(self, x):               
        x = self.conv1(x)              
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        return self.out(x)

if __name__ == "__main__":
    torch.set_default_tensor_type(torch.FloatTensor)
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type = float, default = 4e-3, help = "Learning rate")
    parser.add_argument("--epochs", type = int, default = 30, help = "Training lasts for . epochs")
    parser.add_argument("--class_num", type = int, default = 9, help = "How many classes we have")
    parser.add_argument("--batch_sz", type = int, default = 40, help = "Batch size for miniBatch")
    parser.add_argument("--inspect", type = int, default = 10, help = "Print loss information every <inspect> batches")
    parser.add_argument("-d", "--del_dir", action = "store_true", help = "Delete dir ./logs and start new tensorboard records")
    parser.add_argument("-l", "--load", action = "store_true", help = "Use saved model to train")
    parser.add_argument("-c", "--net", action = "store_true", help = "Use CNN? If false, use ResNet")
    args = parser.parse_args()

    batch_size      = args.batch_sz
    epochs          = args.epochs
    lrate           = args.lr
    inspect_point   = args.inspect
    del_dir         = args.del_dir
    class_num       = args.class_num
    load_saved      = args.load
    use_cnn         = args.net

    tf = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

    train_raw = []
    test_raw = []
    for i, name in enumerate(name_list):
        train_loader = getLoader(tf, name, True)
        for img, _ in train_loader:
            train_raw.append((img, i))
        test_loader = getLoader(tf, name, True, eval = True)
        for img, _ in test_loader:
            test_raw.append((img, i))
        print("%s loaded"%(name))
    random.shuffle(train_raw)

    logdir = '../logs/'
    if os.path.exists(logdir) and del_dir:
        shutil.rmtree(logdir)
    
    time_stamp = "{0:%Y-%m-%d/%H-%M-%S}-epoch{1}/".format(datetime.now(), epochs)
    writer = SummaryWriter(log_dir = logdir+time_stamp)

    if use_cnn:
        net = CNN(class_num)
        path = '../models/cnn_best_model.pth'
    else:
        net = ResNet(1, class_num, 3)
        path = '../models/resnet_best_model.pth'
    opt = optim.Adam(net.parameters(), lr = lrate)      
    sch = optim.lr_scheduler.ExponentialLR(opt, 0.9997, -1)
    loss_func = nn.CrossEntropyLoss()                   
    batch_cnt = 0
    train_acc_cnt = 0
    eval_batch_cnt = 0
    batch_num = len(train_raw) // batch_size

    if load_saved:
        save = torch.load(path)   # 保存的优化器以及模型参数
        save_model = save['model']                  # 保存的模型参数
        model_dict = net.state_dict()              # 当前网络参数
        state_dict = {k:v for k, v in save_model.items() if k in model_dict}    # 找出在当前网络中的参数
        model_dict.update(state_dict)
        net.load_state_dict(model_dict) 
        print("Trained model is loaded from '%s'."%(path))
    net = net.cuda()                                          
    for i in range(epochs):    
        for k in range(batch_num):
            samples = train_raw[k * batch_size : (k + 1) * batch_size]
            bx, by = makeBatch(samples)
            _bx = bx.cuda()
            _by = by.cuda()
            out = net(_bx)  

            loss = loss_func(out, _by)                  
            opt.zero_grad()                             
            loss.backward()
            opt.step()
            sch.step()          
            batch_cnt += 1
            if k % inspect_point != 0: 
                continue
            _, pred = torch.max(out, dim = 1)
            train_acc_cnt += (pred == by).sum()
            eval_batch_cnt += batch_size
            train_acc = train_acc_cnt / eval_batch_cnt
            net.eval()
            eval_cnt = 0
            for j, (bx, by) in enumerate(test_raw):
                bx = bx.cuda()
                out = net(bx)
                _, pred = torch.max(out, dim = 1)
                eval_cnt += (pred.item() == by)
            acc = eval_cnt / len(test_raw)
            print("(%d) Batch Counter: %d / %d\tloss: %.5f\tacc: %.5f\ttrain_acc: %.5f"%(
                i, batch_cnt, batch_num, loss.data.item(), acc, train_acc
            ))
            writer.add_scalar('Eval/Total Loss', loss.data.item(), batch_cnt)
            writer.add_scalar('Eval/Acc', acc, batch_cnt)
            writer.add_scalar('Eval/Train acc', train_acc, batch_cnt)
            net.train()
    print("Training completed.")

    torch.save({
        'model': net.state_dict(),
        'optimizer': opt.state_dict()},
        path
    )