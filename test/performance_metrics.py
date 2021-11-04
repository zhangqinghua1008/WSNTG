# -*- coding:utf-8 -*-
# @Time   : 2021/11/1 12:14
# @Author : 张清华
# @File   : performance_metrics.py
# @Note   :
"""
    通过预测出的图片 和 GT比较，预测指标
"""
import skimage.io as io
from skimage.io import imread, imsave
from utils_network.metrics import iou_score,dice_coef
from pathlib import Path
import numpy as np


def postprocess(pred):
    pred[pred < 127] = 0
    pred[pred >= 127] = 1
    return pred

def accuracy(P, G):
    return (P == G).mean()


def performance_metrics(pre_dir,lable_dir):
    iou_list = []
    dice_list = []
    ac_list = []

    print("pre地址：",pre_dir)
    print("lable地址：",lable_dir)
    for fire in Path(pre_dir).iterdir():
        print('%-6s' % fire.name , end="-> ")
        pred = io.imread(fire)
        pred = postprocess(pred)
        lable = io.imread(Path(lable_dir) / fire.name)
        lable = postprocess(lable)

        iou = iou_score(pred, lable)
        iou_list.append(iou)

        dice_sc = dice_coef(pred, lable)
        dice_list.append(dice_sc)

        ac = accuracy(pred, lable)
        ac_list.append(ac)

        print("Iou:{:.4f}  Dice:{:.4f}  AC:{:.4f}".format(iou, dice_sc, ac))

    print("-------------")
    print("测试集数量：",len(iou_list))
    print("Iou:",np.mean(iou_list))
    print("Dice",np.mean(dice_list))
    print("AC",np.mean(ac_list))

if __name__ == '__main__':
    # pre_dir = r"D:\组会内容\实验报告\MedT\records\Digestpath_WSI_results_new20"
    pre_dir = r"D:\组会内容\实验报告\MedT\records\Digestpath_WSI_results_new20"
    lable_dir = r"D:\组会内容\data\Digestpath2019\MedT\test\labelcol"
    performance_metrics(pre_dir,lable_dir)