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
import matplotlib.pyplot as plt
import time


def postprocess(pred):
    pred[pred < 127] = 0
    pred[pred >= 127] = 1
    return pred

def accuracy(P, G):
    return (P == G).mean()


def save_curve(pre_dir,imgs_metrics,type_index=2,type = "dice"):
    '''
        保存每个测试图像指标的图表
        imgs_metrics: 指标
        type_index: 代表哪个种类的下标 (DICE:2)
    '''
    curve_dir = pre_dir.parent
    dice_save = curve_dir / (pre_dir.name + "_"+type+".png")
    plt.scatter(x=imgs_metrics[:,3],y=imgs_metrics[:,type_index],
                marker="o",c='g',label=type)

    # 绘制平均线
    mean_metrics = np.mean(imgs_metrics, axis=0)
    plt.axhline(y=mean_metrics[type_index], color="red",
                linestyle=":",label = "Mean")  # linestyle: '-', '--', '-.', ':',

    plt.ylim((0, 1)) # 设置y轴范围
    plt.xlabel("Number")
    plt.ylabel(type)
    plt.title(type+pre_dir.name)
    plt.grid()      # 生成网格
    plt.legend()    # 显示label
    plt.savefig(dice_save)
    plt.clf()


def performance_metrics(pre_dir,lable_dir):
    pre_dir = Path(pre_dir)
    lable_dir = Path(lable_dir)
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

        # print('%-6s' % fire.name,fire.name[:-4],"-> ","AC:{:.4f}  Iou:{:.4f}  Dice:{:.4f}  ".format(ac, iou, dice_sc))
        return ac, iou, dice_sc, int(fire.name[:-4])

    # 多线程
    executor = Parallel(n_jobs=12)
    imgs_metrics = executor(delayed(fun)(fire) for fire in tqdm( pre_dir.iterdir(), total=100))
    imgs_metrics = np.array(imgs_metrics)

    print("-------------")
    print("测试集数量：",len(imgs_metrics))
    mean_metrics = np.mean(imgs_metrics, axis=0)
    print("AC:{:.4f}".format( mean_metrics[0]))
    print("Iou:{:.4f}".format(mean_metrics[1]))
    print("Dice:{:.4f}\n".format(mean_metrics[2]))

    save_curve(pre_dir,imgs_metrics,type_index=2,type = "dice") #绘制DICE散点图

    low_dice = imgs_metrics[imgs_metrics[:, 2] < mean_metrics[2]]  # 获取小于DICE平均值的图像
    low_dice = low_dice[np.argsort(low_dice[:,2])]  # 根据DICE排序
    print(" 低于平均值的图像：")
    for metrics in low_dice:
        print('%4s.png' % int(metrics[3]), "-> ",
              "AC:{:.4f}  Iou:{:.4f}  Dice:{:.4f}".format(metrics[0],metrics[1],metrics[2]))

# 执行后处理
def run_post(pre_dir,post_dir):
    print(" -------- - - - - - - - 经过后处理：")
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


# 评价模型预测出的图像指标
def evaluate_img(modelPre_dir,gt_lable_dir):
    '''
        modelPre_dir : 模型预测输出地址
        gt_lable_dir : 真实标签地址
    '''
    start = time.time()
    # 模型直接预测的指标
    pre_dir = modelPre_dir / "_pre"
    performance_metrics(pre_dir, gt_lable_dir)

    # 后处理地址
    post_dir = modelPre_dir / "_post"
    # 执行后处理
    run_post(pre_dir, post_dir)
    # 比较后处理后图像指标
    performance_metrics(post_dir, gt_lable_dir)

    print("花费：", time.time() - start)  # 不open():272   把小图进行open_3 : 250  大图open_30,小图open_4:252


if __name__ == '__main__':

    modelPre_dir = r"D:\组会内容\实验报告\MedT\records\Digestpath_WSI_results_Tgcn\all_test/"  # 模型预测输出地址
    lable_dir = r"D:\组会内容\data\Digestpath2019\MedT\test\all_test" + "/labelcol"

    evaluate_img(modelPre_dir,lable_dir)

