import torch
from utils import get_data_loader_distributed
import numpy as np
from utils.YParams import YParams
from networks.vit import ViT
import matplotlib.pyplot as plt

params = YParams('./config/ViT.yaml', 'short')
params.global_batch_size = 1
params.local_batch_size = 1

valid_dataloader, dataset_valid  = get_data_loader_distributed(params, params.valid_data_path, distributed=False, train=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
params.device = device
model = ViT(params)
model = model.to(device)

with torch.no_grad():
  for i, data in enumerate(valid_dataloader, 0):
    if i >= 1:
        break
    print("Doing iteration {}".format(i))
    inp, tar = map(lambda x: x.to(device, dtype = torch.float), data)
    print("input shape = {}".format(inp.shape))
    print("target shape = {}".format(tar.shape))
    plt.rcParams["figure.figsize"] = (20,20)
    plt.figure()
    for ch in range(inp.shape[1]):
        plt.subplot(inp.shape[1],1, ch+1)
        plt.imshow(inp[0,ch,:,:].cpu(), cmap = 'RdBu')
        plt.colorbar()
    plt.savefig("figs/minibatch_" + str(i) + ".jpg")
    gen = model(inp)
    print("prediction shape = {}".format(gen.shape))

