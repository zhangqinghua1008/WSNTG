import argparse
import lib
import torch
from pathlib import Path
from models import initialize_trainer
from performance_metrics import *
from models.wesup import WESUPPixelInference
from models.TGCN.tgcn import TGCNPixelInference
from models.WSGNet import WSGNetPixelInference
from PIL import Image

from infer_utils import *

Image.MAX_IMAGE_PIXELS = None
"""
    Inference module for window-based strategy.  基于窗口滑动的推理模块。
"""


def infer(trainer, data_dir, resize_size=None, device='cuda', output_dir=None):
    img_paths = list((data_dir / 'images').iterdir())
    print(f'Predicting {len(img_paths)} images from {data_dir} ...')

    trainer.model.eval()
    # with torch.no_grad():
    for img_path in tqdm(img_paths, ncols=len(img_paths)):
        # 像素级推理
        pre = predict_img(trainer, img_path, resize_size=resize_size, device=device)

        if output_dir is not None:
            save_pre(pre, img_path, output_dir / "_pre")


# 像素级别推理
def pixel_infer(trainer, data_dir, resize_size=None, device='cuda', output_dir=None):
    img_paths = list((data_dir / 'images').iterdir())
    print(f'像素级 Predicting {len(img_paths)} images from {data_dir} ...')

    with torch.no_grad():
        for img_path in tqdm(img_paths, ncols=len(img_paths)):
            pre = pixel_predict_img(trainer, img_path, resize_size=resize_size, device=device)
            if output_dir is not None:
                save_pre(pre, img_path, output_dir / "_pre")


# 执行模型正常推理
def infer_run(model_type=None, checkpoint=None, data_dir=""):
    begin = time.time()
    resize_size = 512  # None

    output_dir = checkpoint.parent.parent / 'results'
    output_dir.mkdir(exist_ok=True)

    output_dir = output_dir / checkpoint.stem
    output_dir.mkdir(exist_ok=True)

    # 加载模型并推理
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = initialize_trainer(model_type, device=device)
    if checkpoint is not None:
        trainer.load_checkpoint(checkpoint)

    # 推理
    print("\n++++++++++++++++++++ 当前正在处理：")

    infer(trainer, data_dir, output_dir=output_dir, resize_size=resize_size, device=device)
    print("\ncheckpoint:", checkpoint, "\n")
    # 评价模型输出
    phase_lable_dir = data_dir / "masks"  # GT 地址
    evaluate_img(modelPre_dir=output_dir, gt_lable_dir=phase_lable_dir, need_post=True)
    print("时间：", time.time() - begin)


# 执行像素级别推理
def infer_pixel_run(model_type='WSGNet', checkpoint=None, data_dir=""):
    '''
        model_type : wesup / tgcn
    '''
    begin = time.time()
    resize_size = 400

    output_dir = checkpoint.parent.parent / ('results_' + checkpoint.stem[-4:])
    output_dir.mkdir(exist_ok=True)

    output_dir = output_dir
    output_dir.mkdir(exist_ok=True)

    # 加载模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if model_type == 'wesup':
        model = WESUPPixelInference().to(device)
    elif model_type == 'WSGNet':
        model = WSGNetPixelInference().to(device)
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint)['model_state_dict'])

    # 推理
    print("\n++++++++++++++++++++ 当前正在处理：")
    pixel_infer(model, data_dir, output_dir=output_dir, resize_size=resize_size, device=device)

    print("\ncheckpoint:", checkpoint, "\n")
    # 评价模型输出
    lable_dir = data_dir / "masks"  # GT 地址
    evaluate_img(modelPre_dir=output_dir, gt_lable_dir=lable_dir, need_post=True)
    print("时间：", time.time() - begin)


if __name__ == '__main__':
    model_type = "unet"  # sizeloss || unet / fcn / cdws / WSGNet  / wesup
    data_dir = Path(r"D:\组会内容\data\CRAG\CRAG\val/")

    ckpts = ["ckpt.0100.pth","ckpt.0200.pth"]
    for ckpt in ckpts:
        checkpoint = Path(r"E:\records\CRAG\0对比算法\20221106-1225-PM_unet") / "checkpoints" / ckpt

        if model_type == "WSGNet" or model_type == "wesup":
            # 执行像素级别推理
            infer_pixel_run(model_type=model_type, checkpoint=checkpoint, data_dir=data_dir)
        else:
            # 正常模型推理
            infer_run(model_type=model_type, checkpoint=checkpoint, data_dir=data_dir)
