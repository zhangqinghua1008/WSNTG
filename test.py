# -*- coding:utf-8 -*-
# @Time   : 2021/11/13 15:36
# @Author : 张清华
# @File   : test.py.py
# @Note   :

import numpy as np
from pathlib import Path
import os
from tqdm import tqdm
import time
import torch

test = torch.rand((3,10))
print(test)
print(test[:,:10//2])
print(test[:,10//2:])
