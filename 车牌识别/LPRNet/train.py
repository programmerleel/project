# -*- coding: utf-8 -*-
# @Time    : 2023/12/13 10:09
# @Author  : LiShiHao
# @FileName: train.py
# @Software: PyCharm
import torch

from dataset import *
from net import *
import os
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm


def get_transform(input_size):
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.PILToTensor()
    ])
    return transform


# TODO:多卡分布式训练策略
def get_dataset_and_dataloader(file_path, transform, batch_size, num_workers):
    dataset = PlateDataset(file_path, transform)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                            drop_last=True)
    return dataloader


# TODO:根据网络结构的不同，可以使用不同的初始化方法
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv"):
        nn.init.kaiming_normal_(m.weight, mode='fn_out')
    elif classname.find("BatchNorm"):
        nn.init.xavier_uniform_(m.weight)
    elif classname.find("Linear"):
        nn.init.xavier_uniform_(m.weight)
    if hasattr(m, "bias") and m.bias != None:
        m.bias.data[...] = 0.001


def compile_model(class_num, pretrained_model_path):
    net = LPRNet(3, 32, class_num)
    if pretrained_model_path != "":
        net.load_state_dict(torch.load(pretrained_model_path))
        return net
    else:
        net.apply(weights_init)
        net.loc_net.loc_linear[3].weight.data.zero_()
        net.loc_net.loc_linear[3].bias.data.copy_(torch.tensor([1, 0, 0,
                                                                0, 1, 0], dtype=torch.float))
        return net


def freeze_model(epoch, freeze_epochs, model, freeze_dict):
    for name, param in model.named_parameters():
        if name in freeze_dict and epoch < freeze_epochs:
            param.requires_grad = False
        else:
            param.requires_grad = True
    return model


def warm_up(epoch, warmup_epochs, warmup_start_leaning_rate, train_learning_rate):
    if epoch < warmup_epochs:
        warmup_learning_rate = warmup_start_leaning_rate + (
                (train_learning_rate - warmup_start_leaning_rate) / warmup_epochs) * epoch
        return warmup_learning_rate


def lr_schedule(optimizer, max):
    torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max, eta_min=0, last_epoch=-1)

def dead_output(output):
    for i in output.shape[0]:
        # [88,68]
        prediction = output[i]
        prediction_indexes = []
        # 长度为88的预测序列 每个位置所属字符的概率
        for j in prediction.shape[0]:
            prediction_indexes.append(torch.argmax(prediction[j]))
        result = []
        prediction_index = prediction_indexes[0]
        if prediction_index != len(CHARS)-1:
            result.append(prediction_index)
        for index in prediction_indexes:
            if index == len(CHARS)-1:
                prediction_index = index
                continue
            if index == prediction_index:
                prediction_index = index

def run_training(config):
    transform = get_transform(config.transform.input_size)
    train_dataloader = get_dataset_and_dataloader(config.dataset.train_daaset_fie_path, transform,
                                                  config.dataset.train_batch_size, config.dataset.train_num_workers)
    summary_writer = SummaryWriter(config.result.metrics_dir)
    net = compile_model(config.net.class_num, config.pretrained_model_path)
    learning_rate = config.train.learning_rate
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=config.train.momentum)
    freeze_dict = net.loc_net.named_parameters()
    loss_func = torch.nn.CTCLoss(blank=len(CHARS) - 1, reduction="mean")
    tmp_loss = 0
    for epoch in range(config.train.epochs):
        if config.train.freeze.is_use:
            net = freeze_model(epoch, config.train.freeze.epochs, net, freeze_dict)
            optimizer.params = net.parameters()
        if config.train.warmup.is_use:
            learning_rate = warm_up(epoch, config.train.warmup.epochs, config.train.warmup.start_learning_rate,
                                    config.train.lr.learning_rate)
            optimizer.lr = learning_rate
        if epoch == config.train.warmup.epochs:
            learning_rate = config.train.learning_rate
            optimizer.lr = learning_rate
            if config.train.lr_schedule.is_use:
                lr_schedule(optimizer, config.train.lr_schedule.max)
        epoch_loss = 0
        epoch_acc = 0
        with tqdm(total=len(train_dataloader)) as _tqdm:
            _tqdm.set_description("{}/{}".format(epoch + 1, len(train_dataloader)))
            for image_data, label_data, label_length, target_length in train_dataloader:
                image_data, label_data, label_length, target_length = image_data.cuda(), label_data.cuda(), label_length.cuda(), target_length.cuda()
                output = net(image_data)
                output_ = output.transpose(0, 1)
                loss = loss_func(output_, label_data, label_length, target_length)
                _tqdm.set_postfix(loss="{.6f}".format(loss), lr="{.6f}".format(optimizer.lr))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss = epoch_loss + loss
        summary_writer.add_scalar("loss",epoch_loss,epoch)
        if epoch == 0:
            torch.save(net, os.path.join(config.result.save_dir, "last.pt"))
            torch.save(net, os.path.join(config.result.save_dir, "best.pt"))
            tmp_loss = epoch_loss
        else:
            torch.save(net, os.path.join(config.result.save_dir, "last.pt"))
            if epoch_loss > tmp_loss:
                torch.save(net, os.path.join(config.result.save_dir, "best.pt"))
                tmp_loss = epoch_loss
