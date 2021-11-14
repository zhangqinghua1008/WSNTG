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
from tqdm import tqdm
Image.MAX_IMAGE_PIXELS = None
import time


def postprocess(pred):
    pred[pred < 127] = 0
    pred[pred >= 127] = 1
    return pred

def accuracy(P, G):
    return (P == G).mean()


def performance_metrics(pre_dir,lable_dir):
    print("pre地址：",pre_dir)
    print("lable地址：",lable_dir)

    def fun(fire):
        Image.MAX_IMAGE_PIXELS = None
        pred = io.imread(fire)
        pred = postprocess(pred)
        lable = io.imread(Path(lable_dir) / fire.name)
        lable = postprocess(lable)

        iou = iou_score(pred, lable)
        dice_sc = dice_coef(pred, lable)
        ac = accuracy(pred, lable)

        # print('%-6s' % fire.name,"-> ","AC:{:.4f}  Iou:{:.4f}  Dice:{:.4f}  ".format(ac, iou, dice_sc))
        return ac, iou, dice_sc

    # 多线程
    executor = Parallel(n_jobs=12)
    imgs_metrics = executor(delayed(fun)(fire) for fire in tqdm( Path(pre_dir).iterdir(), total=100))

    print("-------------")
    print("测试集数量：",len(imgs_metrics))
    mean_metrics = np.mean(imgs_metrics, axis=0)
    print("AC:{:.4f}".format( mean_metrics[0]))
    print("Iou:{:.4f}".format(mean_metrics[1]))
    print("Dice:{:.4f}\n".format(mean_metrics[2]))


def run_post(pre_dir,post_dir):
    print("pre地址：", pre_dir)
    print("post地址：", post_dir)
    post_dir = Path(post_dir)
    post_dir.mkdir(exist_ok=True)

    def fun(fire):
        Image.MAX_IMAGE_PIXELS = None
        pred = io.imread(fire)
        post = fast_pred_postprocess(pred,pred.size*0.0005)
        print('%-6s' % fire.name, end="-> ")

        post = Image.fromarray(np.uint8(post))
        post.save(post_dir/ fire.name)

    executor = Parallel(n_jobs=12)
    executor(delayed(fun)(fire) for fire in Path(pre_dir).iterdir())

if __name__ == '__main__':
    start = time.time()
    lable_dir = r"D:\组会内容\data\Digestpath2019\MedT\test\all_test" + "/labelcol"
    modelPre_dir = r"D:\组会内容\实验报告\MedT\records\Digestpath_WSI_results_Tgcn\all_test/"  # 模型预测输出地址

    # 模型直接预测的指标
    pre_dir = modelPre_dir + "/_pre"
    performance_metrics(pre_dir,lable_dir)

    # 后处理地址
    post_dir = modelPre_dir + "/_post"
    # 执行后处理
    run_post(pre_dir,post_dir)
    # 比较后处理后图像指标
    performance_metrics(post_dir, lable_dir)

    print("花费：",time.time()-start)  # 不open():272   把小图进行open_3 : 250  大图open_30,小图open_4:252
