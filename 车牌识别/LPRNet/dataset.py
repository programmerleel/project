# -*- coding: utf-8 -*-
# @Time    : 2023/12/12 16:55
# @Author  : LiShiHao
# @FileName: dataset.py.py
# @Software: PyCharm

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-'
         ]



class PlateDataset(Dataset):
    def __init__(self, file_path, transform):
        super().__init__()
        self.image_paths = []
        self.labels = []
        self.transform = transform
        file_path = file_path
        file = open(file_path, "r")
        file_paths = file.readlines()
        for file_path in file_paths:
            image_path = file_path[0:-1]
            self.image_paths.append(image_path)
            label = image_path.split("/")[-1].split(".")[0]
            self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        image_data = self.transform(image)
        label_data = []
        label = self.labels[index]
        for char in label:
            label_data.append(CHARS.index(char))
        label_data = np.array(label_data, dtype=np.int32)
        label_data_ = np.zeros(88, dtype=np.int32)
        label_data_[0:] = label_data
        label_data_ = torch.from_numpy(label_data_)
        label_length = torch.tensor(88, dtype=torch.int64)
        target_length = torch.tensor(len(label_data),dtype=torch.int64)
        return image_data, label_data_, label_length,target_length
