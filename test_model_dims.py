import torch
from networks.vit import ViT
from utils.YParams import YParams
from torchinfo import summary

params = YParams('./config/ViT.yaml', 'short_mp')
model = ViT(params)
summary(model, input_size=(16,20,360,720))
