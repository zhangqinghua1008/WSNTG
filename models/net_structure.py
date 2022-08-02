# -*- coding:utf-8 -*-
# @Time   : 2021/10/18 14:15
# @Author : 张清华
# @File   : net_utils.py
# @Note   :  模型层的函数

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def vgg_conv():
    return nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        # nn.Conv2d(64, 64, kernel_size=3, padding=1),
        # nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.AdaptiveAvgPool2d(output_size=(8, 8))
    )

def vgg_one_conv():
    return  nn.Sequential(
        # 1*1 卷积
        nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1, stride=1, padding=0),
        nn.ReLU(),
        # nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0),
        # nn.ReLU(),
        # nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0),
        # nn.ReLU(),
        # nn.Conv2d(in_channels=64, out_channels=8, kernel_size=1, stride=1, padding=0),
        # nn.ReLU(),
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(output_size=(6, 6))
    )

def vgg_classifier():
    return nn.Sequential(
        nn.Linear(in_features=256 * 6 * 6, out_features=1024, bias=True),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 2),
        nn.Softmax(dim=1)
    )

