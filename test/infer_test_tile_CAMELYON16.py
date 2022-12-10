import argparse
import lib
import torch
from pathlib import Path
from infer_test_tile_utils import *
from models import initialize_trainer
from performance_metrics import *
from models.wesup import WESUPPixelInference
from models.FSTGN import FSTGNPixelInference
from PIL import Image
import cv2 as cv2
from skimage import io as skio
import xml.etree.ElementTree as et
import math
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os.path as osp
import openslide
from pathlib import Path
# scipy.misc.imsave is deprecated! imsave is deprecated in SciPy 1.0.0,
# and will be removed in 1.2.0. Use imageio.imwrite instead.
#from scipy.misc import imsave as saveim
from imageio import imwrite as saveim


from performance_metrics_CAMELYON16 import run_performance_CAMELYON16
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import PIL.Image as Image
# 解决 pillow 打开图片大于 20M 的限制
Image.MAX_IMAGE_PIXELS = None
"""
    Inference module for window-based strategy.  基于窗口滑动的推理模块。
"""


def infer(trainer,data_dir,patch_size,resize_size=None,device='cuda',output_dir=None):
    data_dir = Path(data_dir)
    img_paths = list((data_dir / 'img').iterdir())
    print(f'Predicting {len(img_paths)} images from {data_dir} ...')

    trainer.model.eval()
    for img_path in tqdm(img_paths,ncols=50):
        pre = predict_bigimg_CAMELYON16(trainer, img_path, patch_size, resize_size=resize_size,device=device)

        if output_dir is not None:
            save_pre(pre, img_path, output_dir / "_pre")

def infer_run_CAMELYON16(model_type = None,checkpoint = None,data_path = ""):
    begin = time.time()
    patch_size = 512
    resize_size = 256  # None

    output_dir = checkpoint.parent.parent / (str(checkpoint.stem)+'_results')
    print(output_dir)
    output_dir.mkdir(exist_ok=True)

    data_dir = Path(data_path)
    output_dir = output_dir
    output_dir.mkdir(exist_ok=True)

    # 加载模型并推理
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = initialize_trainer(model_type, device=device)
    if checkpoint is not None:
        trainer.load_checkpoint(checkpoint)

    # 推理
    infer(trainer,data_dir,patch_size = patch_size ,output_dir=output_dir,resize_size=resize_size,device=device)
    print("\ncheckpoint:", checkpoint, "\n")

    # 评价模型输出
    # phase_lable_dir = data_dir / "labelcol"  # GT 地址
    # evaluate_img(modelPre_dir=output_dir, gt_lable_dir=phase_lable_dir, need_post=True)
    print("时间：", time.time() - begin)


# ==============================================================================================================

# 像素级别推理
def pixel_infer(model,data_dir,patch_size,resize_size=None,device='cuda',output_dir=None):
    data_dir = Path(data_dir)
    img_paths = list((data_dir / 'img').iterdir())
    print(f'像素级 Predicting {len(img_paths)} images from {data_dir} ...')

    with torch.no_grad():
        for img_path in tqdm(img_paths):
            pre = pixel_predict_bigimg_CAMELYON16(model, img_path, patch_size, resize_size=resize_size,device=device)

            if output_dir is not None:
                save_pre(pre, img_path, output_dir / "_pre")

# 执行像素级别推理
def infer_pixel_run_CAMELYON16(model_type = 'FSTGN',checkpoint = None,data_path = ""):
    '''
        model_type : wesup / FSTGN
    '''
    begin = time.time()
    patch_size = 512
    resize_size = 256  # None

    output_dir = checkpoint.parent.parent / (str(checkpoint.stem)+'_results')
    print(output_dir)
    output_dir.mkdir(exist_ok=True)

    data_dir = Path(data_path)
    output_dir = output_dir
    output_dir.mkdir(exist_ok=True)

    # 加载模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if model_type == 'wesup':
        model = WESUPPixelInference().to(device)
    elif model_type == 'FSTGN':
        model = FSTGNPixelInference().to(device)
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint)['model_state_dict'])

    # 推理
    pixel_infer(model, data_dir, patch_size=patch_size, output_dir=output_dir, resize_size=resize_size, device=device)

    print("\ncheckpoint:", checkpoint, "\n")
    # 评价模型输出
    # phase_lable_dir = phase_data_dir / "labelcol"  # GT 地址
    # evaluate_img(modelPre_dir=phase_output_dir, gt_lable_dir=phase_lable_dir, need_post=True)
    # print("时间：", time.time() - begin)


if __name__ == '__main__':

    model_type = "FSTGN"     # sizeloss || unet / fcn / cdws / sizeloss / FSTGN /wesup  1  / yamu

    # data_folder = r"G:\dataG\CAMELYON16\training\patches_level2\test"   # 测试集地址父目录  2 #F:\data2\Camelyon16\testing\masks_level2
    data_folder = r"C:\DataC\16\test"   # 测试集地址父目录  2 #F:\data2\Camelyon16\testing\masks_level2

    ckpts = ["ckpt.0030.pth"]
    for ckpt in ckpts:
        checkpoint = Path(r"D:\组会内容\实验报告\MedT\records_16\0对比算法\20221116-1648-PM_FSTGN") ### 3

        checkpoint =  checkpoint / "checkpoints" / ckpt
        if model_type == "FSTGN" or model_type == "wesup":
            # 执行像素级别推理
            infer_pixel_run_CAMELYON16(model_type=model_type,checkpoint=checkpoint,data_path = data_folder)
        else:
            # 正常模型推理
            infer_run_CAMELYON16(model_type=model_type,checkpoint=checkpoint,data_path = data_folder)

        # 评测
        out = checkpoint.parent.parent / (str(checkpoint.stem)+'_results')
        run_performance_CAMELYON16(model_Pos_Pre_dir = out )
