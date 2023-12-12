# -*- coding: utf-8 -*-
# @Time    : 2023/12/12 16:55
# @Author  : LiShiHao
# @FileName: utils.py
# @Software: PyCharm

from easydict import EasyDict
import logging
import yaml


def get_config(log_path):
    logging.basicConfig(filename=log_path, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s-%(funcName)s',
                        level=logging.INFO)


def load_config(config_path):
    file = open(config_path, "r", encoding="utf-8")
    config = EasyDict(yaml.full_load(file))
    return config
