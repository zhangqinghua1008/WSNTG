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

smooth = 1e-5

# 求H(i)
def Hi(hist, img):
    hi = hist / (img.shape[0] * img.shape[1])
    return hi

# =================== 颜色特征
def colour_feature(gray):
    hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 255])
    h_gray = Hi(hist_gray, gray)

    # gray_mean:均值, gray_var:标准差
    gray_mean, gray_var = cv2.meanStdDev(gray)
    gray_mean, gray_var = gray_mean+smooth, gray_var+smooth
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

    all_colour = [gray_mean[0], gray_var[0], gray_peak[0], gray_energy, gray_entropy]
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
    textures = texture_features(gray)
    morphologicals = morphological_features(gray)

    all = np.hstack( (colours,textures,morphologicals) )
    all = np.vstack(all)
    all[np.isnan(all)] = smooth # 排除nan值
    return all


if __name__ == '__main__':
    start = time.time()

    dir = Path(r"D:\组会内容\data\Digestpath2019\MedT\train\only_mask\train_800\train\img")
    print(dir)
    for index,file in enumerate(dir.iterdir()):
        print(index,file)
        img = io.imread(file)
        img = np.zeros_like(img)
        img = TF.to_tensor(img).unsqueeze(0).cuda()

        features = art_features(img)
        print(features)

        features = torch.from_numpy(features)
        print(features.shape)

        # if index>50:
        break

    print("花费时间：", time.time() - start)
