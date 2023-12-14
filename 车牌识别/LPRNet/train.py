# -*- coding: utf-8 -*-
# @Time    : 2023/12/13 10:09
# @Author  : LiShiHao
# @FileName: train.py
# @Software: PyCharm
import torch

from dataset import *
from net import *
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

def get_transform(input_size):
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.PILToTensor()
    ])
    return transform

# TODO:多卡分布式训练策略
def get_dataset_and_dataloader(file_path,transform):
    dataset = PlateDataset(file_path,transform)
    dataloader = DataLoader(dataset=dataset,batch_size=16,shuffle=True,num_workers=8,drop_last=True)
    return dataloader

# TODO:不同层对应不同的初始化方法，后续研究
def weights_init(m):
    for key in m.state_dict():
        if key.split(".")[-1] == "weight":
            if key == "loc_linear.3.weight":
                continue
            nn.init.kaiming_normal_(m.state_dict()[key],mode="fan_out")
        elif key.split(".")[-1] == "bias":
            if key == "loc_linear.3.bias":
                continue
            m.state_dict()[key][...] = 0.01


def compile_model(class_num,pretrained_model_path):
    net = LPRNet(3,32,class_num)
    if pretrained_model_path != "":
        net.load_state_dict(torch.load(pretrained_model_path))
    else:
        net.loc_net.apply(weights_init)
        net.backbone.apply(weights_init)
        net.conv.apply(weights_init)
        net.linear.apply(weights_init)

def freeze_model():
    pass

def warm_up():
    pass

def lr_schedule():
    pass

def run_training(train_dataloader,val_dataloader,epoch):

    for image,label in tqdm(train_dataloader):
        pass