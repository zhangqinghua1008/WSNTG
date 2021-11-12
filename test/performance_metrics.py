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
from infer_test_tile_utils import fast_pred_postprocess
from joblib import Parallel,delayed
from PIL import Image
Image.MAX_IMAGE_PIXELS = None


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


def run_post(pre_dir,post_dir):
    print("pre地址：", pre_dir)
    print("post地址：", post_dir)
    post_dir = Path(post_dir)
    post_dir.mkdir(exist_ok=True)

    def fun(fire):
        print('%-6s' % fire.name, end="-> ")
        Image.MAX_IMAGE_PIXELS = None
        pred = io.imread(fire)
        print(pred.shape)
        post = fast_pred_postprocess(pred,pred.size*0.0001)

        io.imsave(post_dir/ fire.name,post)

    executor = Parallel(n_jobs=12)
    executor(delayed(fun)(fire) for fire in Path(pre_dir).iterdir())

if __name__ == '__main__':

    lable_dir = r"D:\组会内容\data\Digestpath2019\MedT\test\all_test\labelcol"

    # 模型预测输出地址
    pre_dir = r"D:\组会内容\实验报告\MedT\records\Digestpath_WSI_results_WESUP\temp_all\_pre"
    # performance_metrics(pre_dir,lable_dir)

    # 执行后处理 并 比较指标
    post_dir = r"D:\组会内容\实验报告\MedT\records\Digestpath_WSI_results_WESUP\temp_all\_post"
    run_post(pre_dir,post_dir)
    performance_metrics(post_dir, lable_dir)