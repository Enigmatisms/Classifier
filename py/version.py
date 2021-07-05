#torch_version==1.x
import torch
from CNN import CNN

checkpoint = '../models/cnn_best_model2_9.pth'

model = CNN(9)
saved = torch.load(checkpoint)
state_dict = saved['model']
model.load_state_dict(state_dict)
model.eval()

torch.save(model.state_dict(), checkpoint, _use_new_zipfile_serialization=False)