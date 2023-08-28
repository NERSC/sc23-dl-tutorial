import torch
from networks.vit import ViT
from utils.YParams import YParams
from torchinfo import summary

params = YParams('./config/ViT.yaml', 'base')
params.device = 'gpu'
model = ViT(params)
summary(model, input_size=(1,20,720,1440))
