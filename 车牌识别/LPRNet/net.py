# -*- coding: utf-8 -*-
# @Time    : 2023/12/11 09:37
# @Author  : LiShiHao
# @FileName: net.py.py
# @Software: PyCharm

import torch
from torch import nn


# 采用LocNet网络进行stn操作
class LocNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # 论文LocNet的结构感觉不是很清晰，在复现的时候会出现尺寸问题，对应凑了一下结构，可以在网上找一找其他的复现
        self.with_pool = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 2),
            nn.BatchNorm2d(out_ch),
            nn.AvgPool2d(3, 2),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 5, 3),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

        self.without_pool = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 5, 3),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 5, 5),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(0.5)
        self.loc_linear = nn.Sequential(
            nn.Linear(out_ch * 2 * 6, out_ch),
            nn.BatchNorm1d(out_ch),
            nn.Tanh(),
            nn.Linear(out_ch, 6),
            nn.BatchNorm1d(6),
            nn.Tanh()
        )

        self.loc_linear[3].weight.data.zero_()
        self.loc_linear[3].bias.data.copy_(torch.tensor([1, 0, 0,
                                                         0, 1, 0], dtype=torch.float))

    def stn(self, x):
        feature_conv_with_pool = self.with_pool(x)
        feature_conv_without_pool = self.without_pool(x)
        feature_cat = torch.cat([feature_conv_with_pool, feature_conv_without_pool], dim=1)
        feature_drop = self.dropout(feature_cat).flatten(1, -1)
        feature_linear = self.loc_linear(feature_drop)
        theta = feature_linear.view(feature_linear.shape[0], 2, 3)
        grid = nn.functional.affine_grid(theta, x.size(), True)
        sample_x = nn.functional.grid_sample(x, grid, align_corners=True)
        return sample_x


class SmallBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch // 4, 1, 1),
            nn.BatchNorm2d(out_ch // 4),
            nn.ReLU(),
            nn.Conv2d(out_ch // 4, out_ch // 4, (3, 1), 1, (1, 0)),
            nn.BatchNorm2d(out_ch // 4),
            nn.ReLU(),
            nn.Conv2d(out_ch // 4, out_ch // 4, (1, 3), 1, (0, 1)),
            nn.BatchNorm2d(out_ch // 4),
            nn.ReLU(),
            nn.Conv2d(out_ch // 4, out_ch, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


# 论文中的结构，网上的复现都是千奇百怪的，最终在openvino的仓库中找了LPRNet的网络模型，结合进行复现
class LPRNet(nn.Module):
    def __init__(self, in_ch, out_ch, class_num):
        super().__init__()
        self.loc_net = LocNet(in_ch, out_ch)
        self.backbone = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * 2, 1, 1),
            nn.BatchNorm2d(out_ch * 2),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), (1, 1)),
            SmallBlock(out_ch * 2, out_ch * 4),
            nn.MaxPool2d((3, 3), (2, 1)),
            SmallBlock(out_ch * 4, out_ch * 8),
            SmallBlock(out_ch * 8, out_ch * 8),
            nn.MaxPool2d((3, 3), (2, 1)),
            nn.Dropout(0.5),
            nn.Conv2d(out_ch * 8, out_ch * 8, (4, 1), 1),
            nn.BatchNorm2d(out_ch * 8),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(out_ch * 8, class_num, (1, 13), 1, padding=(0, 6)),
            nn.BatchNorm2d(class_num),
            nn.ReLU()
        )
        self.linear = nn.Sequential(
            nn.Linear(class_num * 88, 128),
            nn.ReLU()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=128 + class_num, out_channels=class_num, kernel_size=(1, 1), stride=(1, 1))
        )

    def forward(self, x):
        sample_x = self.loc_net.stn(x)
        feature_backbone = self.backbone(sample_x)
        feature_linear = self.linear(feature_backbone.flatten(1, -1)).reshape(-1, 128, 1, 1).repeat(1, 1, 1, 88)
        feature_concat = torch.cat([feature_backbone, feature_linear], dim=1)
        feature_conv = self.conv(feature_concat)
        # batch 字典 序列长度
        feature_result = torch.squeeze(feature_conv, 2)
        return feature_result


if __name__ == '__main__':
    net = LPRNet(3, 32, 68)
    from torchsummary import summary

    summary(net, (3, 24, 94), device="cpu")
