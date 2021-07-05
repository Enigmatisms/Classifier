#-*-coding:utf-8-*-
"""
    SIGS Flaw Classifier
    Pipeline contructor: 何千越, 刘睿康
"""
import os
import torch
import shutil
import random
import argparse
import matplotlib as mpl 

from torch import nn                                
from torch import optim                             
from datetime import datetime
from torchvision import transforms     
from torch.utils.tensorboard import SummaryWriter
from auc_calc import AUCPlotter
from ResNet import ResNet
from CNN import CNN
from utils import makeBatch, getLoader
from confusion import ConfusionMatrix

name_list = ["cr", "in", "pa", "ps", "rs", "sc", "gg", "rp", "sp"]

if __name__ == "__main__":
    mpl.rcParams['font.sans-serif'] = u'SimHei'
    mpl.rcParams['axes.unicode_minus'] = False
    torch.set_default_tensor_type(torch.FloatTensor)
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type = float, default = 1e-3, help = "Learning rate")
    parser.add_argument("--ratio", type = float, default = 0.2, help = "Ratio for validation set")
    parser.add_argument("--epochs", type = int, default = 20, help = "Training lasts for . epochs")
    parser.add_argument("--class_num", type = int, default = 9, help = "How many classes we have")
    parser.add_argument("--batch_sz", type = int, default = 40, help = "Batch size for miniBatch")
    parser.add_argument("--inspect", type = int, default = 10, help = "Print loss information every <inspect> batches")
    parser.add_argument("-d", "--del_dir", action = "store_true", help = "Delete dir ./logs and start new tensorboard records")
    parser.add_argument("-t", "--transfer", action = "store_true", help = "Using transfer learning")
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
    ratio           = args.ratio
    transfer        = args.transfer
    mannual_split   = bool(class_num == 6)
    path_prefix = "cnn"

    tf = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(degrees=7, expand=False, fill=None),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
    test_tf = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

    train_raw = []
    test_raw = []
    for i, name in enumerate(name_list[:class_num]):
        post_fix = "%d\\%s\\"%(class_num, name)
        train_loader = getLoader(tf, post_fix, True)
        for img, _ in train_loader:
            train_raw.append((img, i))
        print("%s loaded"%(name))
        if mannual_split == True: continue
        test_loader = getLoader(test_tf, post_fix, True, eval = True)
        for img, _ in test_loader:
            test_raw.append((img, i))
    random.seed(0)              # 固定shuffle
    random.shuffle(train_raw)
    if mannual_split:           # 代码内部分训练集测试集？如果分类数为6就需要手动分类
        length = len(train_raw)
        train_part = int(ratio * length)
        test_raw = train_raw[train_part:]
        train_raw = train_raw[:train_part]

    logdir = '../logs/'
    if os.path.exists(logdir) and del_dir:
        shutil.rmtree(logdir)
    
    time_stamp = "{0:%Y-%m-%d/%H-%M-%S}-epoch{1}/".format(datetime.now(), epochs)
    writer = SummaryWriter(log_dir = logdir+time_stamp)
    plotter = AUCPlotter(batch_size, len(test_raw), class_num)
    plotter.setup(test_raw)
    confmat = ConfusionMatrix(class_num)

    path = '../models/%s_best_model_%d.pth'%(path_prefix, class_num)
    if use_cnn:
        net = CNN(class_num)
    else:
        net = ResNet(1, class_num, 3)
    if transfer:
        fc_params = list(map(id, net.fc.parameters())) # 取得全连接层的参数内存地址的列表
        base_params = filter(lambda p: id(p) not in fc_params, net.parameters()) # 取得其他层参数的列表
        optimizer = optim.Adam([
            {'params': base_params},
            {'params': net.fc.parameters(), 'lr': lrate}],
            lr=lrate / 10, betas=(0.9, 0.999)
        )
    else:
        opt = optim.Adam(net.parameters(), lr = lrate)      
    sch = optim.lr_scheduler.ExponentialLR(opt, 0.9998, -1)
    loss_func = nn.CrossEntropyLoss()   
    batch_cnt = 0
    train_acc_cnt = 0
    eval_batch_cnt = 0
    batch_num = len(train_raw) // batch_size

    if load_saved:
        if mannual_split and transfer:
            transfer_path = '../models/%s_best_model_9.pth'(path_prefix)
            save = torch.load(transfer_path)
            save_model = save['model']
            net.transferFromModel(save_model)
        else:
            save = torch.load(path)   # 保存的优化器以及模型参数
            save_model = save['model']                  # 保存的模型参数
            model_dict = net.state_dict()              # 当前网络参数
            state_dict = {k:v for k, v in save_model.items() if k in model_dict}    # 找出在当前网络中的参数
            model_dict.update(state_dict)
            net.load_state_dict(model_dict) 
            print("Trained model is loaded from '%s'."%(path))
        if 'optimizer' in save:
            saved_opt = save['optimizer']
            opt_dict = opt.state_dict()
            state_dict = {k:v for k, v in saved_opt.items() if k in model_dict}
            opt_dict.update(state_dict)
            opt.load_state_dict(opt_dict) 

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
            with torch.no_grad():
                for j, (bx, by) in enumerate(test_raw):
                    bx = bx.cuda()
                    out = net(bx)
                    _, pred = torch.max(out, dim = 1)
                    eval_cnt += (pred.item() == by)
                    confmat.addElement(by, pred.item())
            acc = eval_cnt / len(test_raw)

            print("(%d) Batch Counter: %d / %d\tloss: %.5f\tacc: %.5f\ttrain_acc: %.5f\tlr: %.5f"%(
                i, batch_cnt, batch_num, loss.data.item(), acc, train_acc, sch.get_last_lr()[-1]
            ))
            auc = plotter.saveAucFig(net, batch_cnt)
            writer.add_scalar('Eval/Total Loss', loss.data.item(), batch_cnt)
            writer.add_scalar('Eval/Acc', acc, batch_cnt)
            writer.add_scalar('Eval/Train acc', train_acc, batch_cnt)
            writer.add_scalar('Eval/AUC', auc, batch_cnt)
            net.train()
    print("Training completed.")
    confmat.saveConfusionMatrix("../confusion_matrix_%s_%d.png"%(path_prefix, class_num))
    torch.save({
        'model': net.state_dict(),
        'optimizer': opt.state_dict()},
        path
    )
    