"""
Inference mode 普通推理模块工具类。
"""
import math
import os
import os.path as osp
from itertools import product
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm
from PIL import Image
from skimage.io import imread
from skimage import measure
from skimage import color
import cv2
from skimage.filters import threshold_otsu
from skimage import io, transform
import time
from PIL import Image
from skimage.transform import resize
from matplotlib import pyplot as plt


def resize_img(img, target_size):
    img = resize(img, target_size, order=1, anti_aliasing=False)
    return (img * 255).astype('uint8')


def resize_mask(mask, target_size):
    mask = (mask * 255).astype('uint8')
    mask = resize(mask, target_size, order=1, anti_aliasing=False)
    mask = (mask * 255).astype('uint8')
    mask[mask < 127] = 0
    mask[mask > 127] = 1
    return mask


def preprocess(device, *data):
    return [datum.to(device) for datum in data]


def predict(dataset, trainer, device='cuda'):
    # 加载数据集
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=4)

    print(f'\nPredicting {len(dataset)} images ...')
    predictions = []
    for data in tqdm(dataloader, total=len(dataset)):
        img = data[0].to(device)

        # infer 单个 img
        img = trainer.preprocess(img)  # w
        img = img[0]
        prediction = trainer.postprocess(trainer.model(img))

        # UneXt 专属
        prediction = torch.sigmoid(prediction).squeeze(0).detach().cpu().numpy()
        prediction[prediction >= 0.5] = 1
        prediction[prediction < 0.5] = 0
        predictions.append(prediction[0])

    return predictions


def predict_img(trainer, img_path, resize_size=None, device='cuda'):
    """ Predict on a single input image.  预测单一输入图像。
    Returns:
        predictions: list of model predictions of size (H, W)
    """
    img = imread(img_path)

    if resize_size:
        w, h = img.shape[0], img.shape[1]
        img = resize_img(img, (resize_size, resize_size))  # resize 成特定大小

    img = TF.to_tensor(Image.fromarray(img)).to(device).unsqueeze(0)
    # infer 单个 img
    img = trainer.preprocess(img)  # w
    input_ = img[0]
    prediction = trainer.postprocess(trainer.model(input_))
    prediction = prediction.detach().cpu().numpy().astype('uint8')

    if resize_size:
        prediction = resize_mask(prediction, (1, w, h))  # 恢复到原来大小

    return prediction


def pixel_predict_img(model, img_path, resize_size=None, device='cpu'):
    """
        WESUP/tgcn 像素级推测
    Returns:
        predictions: list of model predictions of size (H, W)
    """
    img = imread(img_path)

    if resize_size:
        w, h = img.shape[0], img.shape[1]
        scale_range = 0.5
        img = resize_img(img, (int(scale_range * w), int(scale_range * h)))  # resize 成特定大小
        # img = resize_img(img, (resize_size, resize_size))  # resize 成特定大小

    # infer 单个patch
    input_ = TF.to_tensor(Image.fromarray(img)).to(device).unsqueeze(0)
    prediction = model(input_)
    prediction = prediction.unsqueeze(0).detach().cpu().numpy()[..., 1].astype(np.float16)

    if resize_size:
        prediction = resize_mask(prediction, (1, w, h))  # 恢复到patch_size 大小

    return prediction


def save_pre(prediction, img_paths, output_dir='predictions'):
    """Save predictions to disk. 将预测(值在 0-1 之间)保存到磁盘。
    Args:
        prediction: model predictions of size (N, H, W)
        img_paths: list of paths to input images
        output_dir: path to output directory
    """
    if not osp.exists(output_dir):
        os.mkdir(output_dir)

    img_name = img_paths.name
    prediction = prediction.astype('uint8').transpose(1, 2, 0)
    pre = prediction * 255
    io.imsave(output_dir / img_name, pre)
    # 绘制紫色图片
    plt_dir = output_dir.parent / (output_dir.stem + "-plt")
    if not osp.exists(plt_dir):
        os.mkdir(plt_dir)
    plt.imsave( plt_dir / img_name, pre[:,:,0])
