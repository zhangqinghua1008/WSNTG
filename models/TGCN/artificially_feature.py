# -*- coding:utf-8 -*-
# @Time   : 2021/11/17 19:58
# @Author : 张清华
# @File   : artificially_feature.py
# @Note   :  手工特征获取

import cv2
import time
import numpy as np
import torch
import torchvision.transforms.functional as TF
from pathlib import Path
from skimage import io
import matplotlib.pyplot as plt

smooth = 1e-5

# 求H(i)
def Hi(hist, img):
    hi = hist / (img.shape[0] * img.shape[1])
    return hi

# ---------------------------------- 一整张图的情况
# =================== 颜色特征
def colour_feature(gray):
    hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 255])
    h_gray = Hi(hist_gray, gray)

    # gray_mean:均值, gray_var:标准差
    gray_mean, gray_var = cv2.meanStdDev(gray)
    gray_mean, gray_var = (gray_mean+smooth), (gray_var+smooth)
    # peak
    gray_peak = 0
    for i, h_i in enumerate(h_gray):
        gray_peak += (i - gray_mean) ** 4 * h_i
    gray_peak = gray_peak / (gray_var ** 4) - 3

    # energy
    gray_energy = np.array([np.sum(h_gray ** 2)+smooth])

    # entropy
    gray_entropy = 0
    for i, h_i in enumerate(h_gray):
        if (h_i > 0):
            gray_entropy += h_i * np.log2(h_i)
    gray_entropy = -gray_entropy + smooth

    all_colour = [gray_mean[0]/256, gray_var[0]/256, gray_peak[0], gray_energy, gray_entropy]
    all_colour = np.hstack(all_colour)

    return all_colour


# ==================== 纹理特征
def coarseness(image, kmax):
    image = np.array(image)
    w = image.shape[0]
    h = image.shape[1]
    kmax = kmax if (np.power(2,kmax) < w) else int(np.log(w) / np.log(2))
    kmax = kmax if (np.power(2,kmax) < h) else int(np.log(h) / np.log(2))
    average_gray = np.zeros([kmax,w,h])
    horizon = np.zeros([kmax,w,h])
    vertical = np.zeros([kmax,w,h])
    Sbest = np.zeros([w,h])

    for k in range(kmax):
        window = np.power(2,k)
        for wi in range(w)[window:(w-window)]:
            for hi in range(h)[window:(h-window)]:
                average_gray[k][wi][hi] = np.sum(image[wi-window:wi+window, hi-window:hi+window])
        for wi in range(w)[window:(w-window-1)]:
            for hi in range(h)[window:(h-window-1)]:
                horizon[k][wi][hi] = average_gray[k][wi+window][hi] - average_gray[k][wi-window][hi]
                vertical[k][wi][hi] = average_gray[k][wi][hi+window] - average_gray[k][wi][hi-window]
        horizon[k] = horizon[k] * (1.0 / np.power(2, 2*(k+1)))
        vertical[k] = horizon[k] * (1.0 / np.power(2, 2*(k+1)))

    for wi in range(w):
        for hi in range(h):
            h_max = np.max(horizon[:,wi,hi])
            h_max_index = np.argmax(horizon[:,wi,hi])
            v_max = np.max(vertical[:,wi,hi])
            v_max_index = np.argmax(vertical[:,wi,hi])
            index = h_max_index if (h_max > v_max) else v_max_index
            Sbest[wi][hi] = np.power(2,index)

    fcrs = np.mean(Sbest)
    return fcrs

# 纹理特征
def texture_features(img_gray):
    # 论文中用的这个
    # fcrs = coarseness(img_gray, 5)
    fcrs = 1.1
    return np.array( [fcrs] )


# ==================== 形态学特征
def morphological_features(img_gray):
    # 阈值化img
    ret, thresh = cv2.threshold(img_gray, 125, 255, cv2.THRESH_BINARY)
    # contours:存储轮廓信息
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # 平均面积
    post_cnt = [1]
    post_area = [1]
    post_lenth = [1]
    for cnt in contours:
        cnt_area = cv2.contourArea(cnt)  # 计算轮廓面积
        # 筛选掉过小的轮廓
        if (cnt_area < 50):
            continue

        cnt_lenth = cv2.arcLength(cnt, True)  # 计算周长（true代表轮廓是闭合的）

        post_cnt.append(cnt)
        post_area.append(cnt_area)
        post_lenth.append(cnt_lenth)

    # print("筛选掉后剩余轮廓数：", len(post_area), "平均面积：", np.mean(post_area), "平均周长：", np.mean(post_lenth))
    return np.array( [len(post_area),np.mean(post_area),np.mean(post_lenth)] )


#手工特征
def art_features(_img):
    if torch.is_tensor(_img):
        img = _img.clone()
        img = img.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
        img = (img * 255).astype('uint8')
    else:
        img = _img

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    colours = colour_feature(gray)
    # textures = texture_features(gray)
    # morphologicals = morphological_features(gray)

    all = np.hstack( (colours) )  # 【1，9】
    all = np.vstack(all)        # [9,1]
    all[np.isnan(all)] = smooth # 排除nan值
    return all

# ----------------------------------------------------------------------
from skimage.transform import resize
from skimage.segmentation import slic
def resize_img(img, target_size):
    img = resize(img, target_size, order=1, anti_aliasing=False)
    return (img * 255).astype('uint8')


def segment(img):
    sp_area = 200
    segments = slic(
        img.squeeze().cpu().numpy().transpose(1, 2, 0),
        n_segments=int(img.size(-2) * img.size(-1) / sp_area),
        compactness=40,
    )
    segments = torch.as_tensor(
        segments, dtype=torch.long, device='cuda')

    return segments


# 输入进来的都是Tersor格式
# sp_maps ： Tersor
def sp_art(img_tersor,sp_maps_torch):
    if torch.is_tensor(img_tersor):
        img = img_tersor.clone()
        img = img.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
        img = (img * 255).astype('uint8')
    else:
        img = img_tersor

    if torch.is_tensor(sp_maps_torch):
        sp_maps = sp_maps_torch.detach().cpu().numpy()
    else:
        sp_maps = sp_maps_torch

    all_sp_features = []
    sp_num = np.max(sp_maps) + 1
    for sp_index in range(int(sp_num)):
        sp_img = np.zeros_like(img)
        sp_img[sp_maps == sp_index] = img[sp_maps == sp_index]

        # plt.imshow(sp_img),plt.show()

        # 切割超像素
        m, n = np.where(sp_maps == sp_index)
        sp_img = sp_img[min(m):max(m) + 1, min(n):max(n) + 1]

        # plt.imshow(sp_img),plt.show()

        sp_img = (sp_img).astype('uint8')  # sp_img = (sp_img * 255).astype('uint8')
        sp_features = art_features(sp_img)
        all_sp_features.append(sp_features)

    return  np.array( all_sp_features )


def sp_(img):
    sp_segment = segment(img)
    sp_num = torch.max(sp_segment) + 1
    print("超像素个数：", sp_num)

    sp_idx_list = sp_segment.unique()
    sp_maps = sp_segment == sp_idx_list[:, None, None]
    sp_maps = sp_maps.squeeze().float()  # size: (S_N,W,H)
    print(sp_maps.shape)

    sp_maps = sp_maps.view(sp_num, 270, 270).argmax(dim=0)  # size:(H,W)

    all_sp_features = sp_art(img,sp_maps)
    return  all_sp_features


if __name__ == '__main__':
    start = time.time()

    dir = Path(r"D:\组会内容\data\Digestpath2019\MedT\train\only_mask\train_800\train\img")
    print(dir)

    all = []
    for index,file in enumerate(dir.iterdir()):
        print(index,file)
        img = io.imread(file)
        img = resize_img(img,target_size=(270,270))
        img = TF.to_tensor(img).unsqueeze(0).cuda()

        features = art_features(img)
        features = torch.from_numpy(features)

        # -----------超像素测试
        sp_f = sp_(img)
        np.set_printoptions(suppress=True, precision=4) # 保留4位
        print("mean:",np.mean(sp_f,axis=0))
        print("max ：",np.max(sp_f,axis=0))
        print("min ：",np.min(sp_f,axis=0))

        sp_f = torch.from_numpy(sp_f).cuda().to(torch.float32)
        sp_f = torch.sigmoid(sp_f)
        print("mean:", torch.mean(sp_f, axis=0))
        print("max ：", torch.max(sp_f, axis=0))
        print("min ：", torch.min(sp_f, axis=0))
        if index>20:
            break
        # break
    # np.set_printoptions(precision=4)  # 保留4位
    # print("mean:",np.mean(all,axis=1))
    # print("max ：",np.max(all,axis=1))
    # print("min ：",np.min(all,axis=1))
    #
    print("花费时间：", time.time() - start)
