import torch
from CNN import CNN
from torchvision import datasets                    
from torch.utils.data.dataloader import DataLoader 

def getLoader(_tf, path, shuffle = False, eval = False):
    if eval == False:
        folder = datasets.ImageFolder("..\\train%s"%(path), transform = _tf)
    else:
        folder = datasets.ImageFolder("..\\val%s"%(path), transform = _tf)
    loader = DataLoader(folder, shuffle = shuffle)
    return loader

def makeBatch(samples):
    img_batch = []
    y_batch = []
    for img, y in samples:
        img_batch.append(img)
        y_batch.append(y)
    return torch.cat(img_batch, dim = 0), torch.LongTensor(y_batch).cuda()

def stateDictRenamer(netType, path, args):
    net = netType(*args)
    save = torch.load(path)
    save_model = save['model']
    model_dict = net.state_dict()
    state_dict = dict()
    for k, v in save_model.items():
        if k in model_dict:
            state_dict[k] = v
        else:
            if "out" in k:
                post_fix = k.split("out")[-1]
                state_dict["fc%s"%(post_fix)] = v
    model_dict.update(state_dict)
    net.load_state_dict(model_dict) 
    torch.save({
        'model': net.state_dict(),
        'optimizer': save['optimizer'],},
        path
    )
    print("State dict renamed.")


if __name__ == "__main__":
    stateDictRenamer(CNN, "../models/cnn_best_model_9.pth", (9, ))
