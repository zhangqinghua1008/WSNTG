"""
数据集测试
"""
import argparse
import csv
import os
import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from skimage.measure import label
from joblib import Parallel, delayed

if __name__ == '__main__':
    root_dir = r"D:\组会内容\data\PanNuke Dataset\Fold 1"  #'带掩码级注释的数据根目录的路径'
    point_ratio =4.5e-5# 参数用于控制已标记像素的百分比。default=1e-4


    if not os.path.exists(root_dir):
        print('数据地址不存在')
        sys.exit(1)

    img_dir = os.path.join(root_dir, 'images')
    mask_dir = os.path.join(root_dir, 'masks')
    if not os.path.exists(mask_dir):
        print('没有masks无法生成点注释。')
        sys.exit(1)

    label_dir = os.path.join(root_dir, f'points-{str(point_ratio)}')
    print("存放点标签的地址:",label_dir)


    # #对某一个图像的mask 进行处理
    # def para_func(fname):
    #     basename = os.path.splitext(fname)[0]
    #     # mask = np.array(Image.open(os.path.join(mask_dir, fname)))  # 读取 mask图像 并转换成array
    #     mask = np.load(os.path.join(mask_dir, fname))  # 读取 mask.npy 并转换成array
    #
    #
    img = np.load(os.path.join(img_dir, "images.npy"))  # 读取 mask.npy 并转换成array

    # 对mask_dir中的每个图像mask,进行para_func函数.
    for fname in tqdm(os.listdir(mask_dir) ):
        # para_func(fname)
        mask = np.load(os.path.join(mask_dir, fname))  # 读取 mask.npy 并转换成array

