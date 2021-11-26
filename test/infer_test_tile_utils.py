# -*- coding:utf-8 -*-
# @Time   : 2021/10/28 21:24
# @Author : 张清华
# @File   : test_tile.py
# @Note   :
"""
Inference module for window-based strategy.  基于窗口策略的推理模块。
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

# 图像open操作
def img_open(image,open_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (open_size, open_size))
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return image


def preprocess(device,*data):
    return [datum.to(device) for datum in data]

def postprocess(pred, target=None):
    # pred = pred.round()[:, 1, ...].long()
    pred = pred[:, 1, ...]
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    pred = pred.long()

    if target is not None:
        return pred, target.argmax(dim=1)
    return pred

def _get_top_left_coordinates(height, width, patch_size):
    """Calculate coordinates of top-left corners for patches. 计算补丁的左上角坐标"""

    n_h = math.ceil(height / patch_size)  # 用math.ceil 获得的是int,np.ceil是float
    n_w = math.ceil(width / patch_size)
    tops = np.linspace(0, height - patch_size, n_h, dtype=int) # 返回一个列表 array([  0, 133, 266])
    lefts = np.linspace(0, width - patch_size, n_w, dtype=int)

    return product(tops, lefts)


def divide_image_to_patches(img, patch_size):
    """
    Divide a large image (mask) to patches with (possibly overlapping) tile strategy.
    用(可能是重叠的)tile策略将一个大图像(mask)分割成patch。
    Args:
        img: input image of shape (H, W, 3)
        patch_size: target size of patches

    Returns:
        patches: patches of shape (N, patch_size, patch_size, 3)
    """

    assert len(img.shape) == 3 and img.shape[-1] == 3

    height, width, _ = img.shape
    coordinates = _get_top_left_coordinates(height, width, patch_size)

    patches = []

    for top, left in coordinates:
        patches.append(img[top:top + patch_size, left:left + patch_size])

    return np.array(patches).astype('uint8')


def combine_patches_to_image(patches, target_height, target_width):
    """Combine patches back to a single image (mask). 将patch组合成单一图像(mask)

    Args:
        patches: predicted patches of shape (N, H, W, C) or (N, H, W)
        target_height: target height of combined image
        target_width: target width of combined image

    Returns:
        combined: combined output of shape (H, W, C) or (H, W)
    """
    counter = 0
    patch_size = patches.shape[1]
    coordinates = _get_top_left_coordinates(
        target_height, target_width, patch_size)

    if len(patches.shape) == 3:  # channel dimension is missing
        patches = np.expand_dims(patches, -1)

    # The last channel is the number of overlapping patches for a given pixel, 最后一个通道是给定像素的重叠补丁数，
    # used for averaging predictions from multiple windows.                   用于平均来自多个窗口的预测
    combined = np.zeros((target_height, target_width, patches.shape[-1] + 1))

    for top, left in coordinates:
        patch = combined[top:top + patch_size, left:left + patch_size, :-1]
        overlaps = combined[top:top + patch_size, left:left + patch_size, -1:]
        patch = (patch * overlaps + patches[counter]) / (overlaps + 1)
        combined[top:top + patch_size, left:left + patch_size, :-1] = patch
        overlaps += 1.
        counter += 1

    return np.squeeze(combined[..., :-1])


def predict_bigimg(trainer, img_path, patch_size, resize_size=None,device='cpu'):
    """Predict on a single input image.  预测单一输入图像。
    Arguments:
        trainer: trainer for inference
        img_path: instance of `torch.utils.data.Dataset`
        patch_size: patch size when feeding into network
        device: target device
    Returns:
        predictions: list of model predictions of size (H, W)
    """
    img = imread(img_path)
    patches = divide_image_to_patches(img, patch_size)

    predictions = []

    for patch in tqdm(patches,ncols=50):
        if resize_size:
            patch = resize_img(patch, (resize_size, resize_size))  # resize 成特定大小

        # infer 单个patch
        input_ = TF.to_tensor(Image.fromarray(patch)).to(device).unsqueeze(0)
        # prediction = model(input_)
        input_, _ = trainer.preprocess(input_)  # w
        prediction = trainer.postprocess(trainer.model(input_))
        prediction = prediction.detach().cpu().numpy().astype('uint8')

        if resize_size:
            prediction = resize_mask(prediction, (1,patch_size, patch_size))  # 恢复到patch_size 大小

        predictions.append(prediction[..., np.newaxis])

    predictions = np.concatenate(predictions)

    return combine_patches_to_image(predictions, img.shape[0], img.shape[1])

def pixel_predict_bigimg(model, img_path, patch_size, resize_size=None,device='cpu'):
    """
        WESUP像素级推测
    Returns:
        predictions: list of model predictions of size (H, W)
    """
    img = imread(img_path)
    patches = divide_image_to_patches(img, patch_size)
    predictions = []

    # for patch in tqdm(patches,ncols=70):
    for patch in patches:
        if resize_size:
            patch = resize_img(patch, (resize_size, resize_size))  # resize 成特定大小

        # infer 单个patch
        input_ = TF.to_tensor(Image.fromarray(patch)).to(device).unsqueeze(0)
        prediction = model(input_)
        prediction = prediction.unsqueeze(0).detach().cpu().numpy()[..., 1]
        # prediction = prediction.detach().cpu().numpy().astype('uint8')

        if resize_size:
            prediction = resize_mask(prediction, (1,patch_size, patch_size))  # 恢复到patch_size 大小

        # predictions.append(np.expand_dims(prediction, 0))
        predictions.append(prediction[..., np.newaxis])

    predictions = np.concatenate(predictions)

    return combine_patches_to_image(predictions, img.shape[0], img.shape[1])



def save_pre(predictions, img_paths, output_dir='predictions'):
    """Save predictions to disk. 将预测(值在 0-1 之间)保存到磁盘。

    Args:
        predictions: model predictions of size (N, H, W)
        img_paths: list of paths to input images
        output_dir: path to output directory
    """
    print(f'\nSaving prediction to {output_dir} ...')

    if not osp.exists(output_dir):
        os.mkdir(output_dir)

    img_name = osp.basename(img_paths)
    # print(img_name)
    pred = predictions.astype('uint8')
    Image.fromarray(pred * 255).save(osp.join(output_dir, img_name))

def save_post(predictions, img_paths, output_dir='predictions'):
    """ 将后处理结果(值在 0-255 之间)保存到磁盘。
    """
    if not osp.exists(output_dir): os.mkdir(output_dir)

    img_name = osp.basename(img_paths)
    pred = predictions.astype('uint8')
    Image.fromarray(pred).save(osp.join(output_dir, img_name))


def save_predictions(predictions, img_paths, output_dir='predictions'):
    """Save predictions to disk. 将预测保存到磁盘。
    Args:
        predictions: model predictions of size (N, H, W)
        img_paths: list of paths to input images
        output_dir: path to output directory
    """

    print(f'\nSaving prediction to {output_dir} ...')

    if not osp.exists(output_dir):
        os.mkdir(output_dir)

    for pred, img_path in tqdm(zip(predictions, img_paths), total=len(predictions)):
        img_name = osp.basename(img_path)
        print(img_name)
        pred = pred.astype('uint8')
        Image.fromarray(pred * 255).save(osp.join(output_dir, img_name))


from skimage.transform import resize
def resize_img(img, target_size):
    img = resize(img, target_size, order=1, anti_aliasing=False)
    return (img * 255).astype('uint8')

def resize_mask(mask, target_size):
    mask = (mask * 255).astype('uint8')
    mask = resize(mask, target_size, order=1, anti_aliasing=False)
    mask = (mask * 255).astype('uint8')
    mask[ mask < 127] = 0
    mask[ mask > 127 ] = 1
    return mask


def pred_postprocess(pred, threshold=10000):
    # 执行像素级后处理时候很慢，后续改进一下; 以改进：fast_pred_postprocess
    regions = measure.label(pred)
    for region_idx in range(regions.max() + 1):
        region_mask = regions == region_idx
        if region_mask.sum() < threshold:
            pred[region_mask] = 0

    revert_regions = measure.label(255 - pred)
    for region_idx in range(revert_regions.max() + 1):
        region_mask = revert_regions == region_idx
        if region_mask.sum() < threshold:
            pred[region_mask] = 255
    return pred


def fast_pred_postprocess(pred, threshold=10000):
    # 先进行open操作,去掉零星点
    if pred.size < 4000 * 4000:
        threshold = pred.size * 0.00008
        pred = img_open(pred, open_size=3)
    elif pred.size <10000*10000:
        threshold = pred.size * 0.00004
        pred = img_open(pred, open_size=6)
    else:
        pred = img_open(pred,open_size = 30)
        regions = measure.label(pred)
        # 规避图像大小过大 域过多的情况
        # return pred
        if( regions.max() > 50000):
            return pred
        threshold = pred.size*0.000005
        print("超过10000，但是区域少于5w")

    #  快速后处理。
    regions = measure.label(pred)
    props = measure.regionprops(regions)
    resMatrix = np.zeros(regions.shape).astype(np.uint8)
    for i in range(0, len(props)):
        if props[i].area > threshold:
            tmp = (regions == i + 1).astype(np.uint8)
            resMatrix += tmp  # 组合所有符合条件的连通域
    resMatrix *= 255

    #     去掉包含的杂点
    revert_regions = measure.label(255 - resMatrix)
    revert_props = measure.regionprops(revert_regions)
    new_resMatrix = np.zeros(regions.shape).astype(np.uint8)
    for i in range(0, len(revert_props)):
        if revert_props[i].area < threshold:
            tmp = (revert_regions == i + 1).astype(np.uint8)
            new_resMatrix += tmp  # 组合所有符合条件的连通域
    new_resMatrix *= 255
    return resMatrix + new_resMatrix

