# -*- coding: utf-8 -*-
# @Time    : 2023/12/13 11:05
# @Author  : LiShiHao
# @FileName: test.py
# @Software: PyCharm

from net import *

net = LPRNet(3,32,68)

opt = torch.optim.SGD(net.parameters(),lr=0.001,momentum=0.9)

opt.lr = 0.005
print(opt.lr)

# for key in net.loc_net.state_dict():
#     print(key)

# print(net.loc_net)

# def weights_init(m):
#     classname=m.__class__.__name__
#
#     # print(classname,hasattr(m,"weight"),hasattr(m,"bias"))
#
# #     print(m.state_dict().keys())
# net.loc_net.apply(weights_

# for (name,param) in net.named_parameters():
#     print(name)

