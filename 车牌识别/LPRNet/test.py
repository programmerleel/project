# -*- coding: utf-8 -*-
# @Time    : 2023/12/13 11:05
# @Author  : LiShiHao
# @FileName: test.py
# @Software: PyCharm

from net import *

net = LPRNet(3,32,68)

print(net.loc_net.state_dict())

for key in net.loc_net.state_dict():
    print(key)