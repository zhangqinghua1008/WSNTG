# -*- coding:utf-8 -*-
# @Time   : 2021/12/1 17:11
# @Author : 张清华
# @File   : geneerate_patches_DP2019.py
# @Note   :

"""
    把Digestpath数据集中的WSI 图像 切分成pacth。
"""

import sys
from pathlib import Path
from matplotlib.pyplot import cla
import numpy as np
from tqdm import tqdm
import skimage.io as io
from joblib import Parallel, delayed
from PIL import ImageFile
from skimage.transform import resize
# ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image
import shutil
import math
import random
from itertools import product
import math


def resize_img(img, target_size):
    img = resize(img, target_size, order=1, anti_aliasing=False)
    return (img * 255).astype('uint8')

# 移动文件
def move_fire(ori_path, save_path, suffix=""):
    img_dir = Path(ori_path) / "img"
    label_dir = Path(ori_path) / "labelcol"

    save_img_dir = Path(save_path) / ("img" + suffix)
    save_img_dir.mkdir(exist_ok=True)
    save_label_dir = Path(save_path) / ("labelcol" + suffix)
    save_label_dir.mkdir(exist_ok=True)
    # 转移img
    for fire in sorted(img_dir.iterdir()):
        img_path = save_img_dir / fire.name
        shutil.copyfile(fire, img_path)  # 拷贝文件

    # 转移label
    for fire in sorted(label_dir.iterdir()):
        img_path = save_label_dir / fire.name
        shutil.copyfile(fire, img_path)  # 拷贝文件

# 获取保存地址
def save_dir(output):
    output = Path(output)
    output.mkdir(exist_ok=True)
    target_img_dir = output / 'img'
    target_mask_dir = output / 'labelcol'
    target_img_dir.mkdir(exist_ok=True)
    target_mask_dir.mkdir(exist_ok=True)
    return target_img_dir, target_mask_dir

def _get_top_left_coordinates(height, width, patch_size):
    """计算补丁的左上角坐标"""
    n_h = math.ceil(height / patch_size)  # 用math.ceil 获得的是int,np.ceil是float
    n_w = math.ceil(width / patch_size)
    tops = np.linspace(0, height - patch_size, n_h, dtype=int)  # 返回一个列表 array([  0, 133, 266])
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

def process_img_and_mask_all(img_path, mask_path, target_img_dir, target_mask_dir, need_filter=False, n_patches=12):
    '''
        在pos类中 挑选全部的前景patch
        # need_filter : 需要筛选（也就是选取有意义patch）
    '''
    Image.MAX_IMAGE_PIXELS = None
    img = io.imread(img_path)
    mask = io.imread(mask_path)
    h, w = img.shape[:2]
    name = img_path.name
    suffix = img_path.suffix  # 后缀

    coordinates = _get_top_left_coordinates(h, w, patch_size)

    n = 0
    flag = 3
    for top, left in coordinates:
        is_save = 0
        img_patch = img[top:top + patch_size, left:left + patch_size]
        mask_patch = mask[top:top + patch_size, left:left + patch_size]
        # 去掉空白的patch
        if img_patch.mean() > 225:
            continue

        # 选定mask区域
        if mask.mean() > 15 and mask.mean() < 255:
            # is_save = 1  # 是mask就保存
            is_save = random.randint(0,2)%2 + 1  # 是mask：2/3 的概率
        else:
            is_save = random.randint(0,2)      # 不是mask：1/3 的概率

        # 可以保存
        if is_save ==1:
            patch_name = name.replace(suffix, f'_{n}.png')
            io.imsave(str(target_img_dir / patch_name), img_patch, check_contrast=False)
            io.imsave(str(target_mask_dir / patch_name), mask_patch, check_contrast=False)
        n += 1


def process_img_and_mask(img_path, mask_path, target_img_dir, target_mask_dir, need_filter=False, n_patches=12):
    '''
        在pos类中 随机挑选patch
        # need_filter : 需要筛选（也就是选取有意义patch）
    '''
    Image.MAX_IMAGE_PIXELS = None
    img = io.imread(img_path)
    mask = io.imread(mask_path)
    h, w = img.shape[:2]
    name = img_path.name
    suffix = img_path.suffix  # 后缀

    select_num = round( max(h//patch_size,w//patch_size)*2.2 ) # 被挑选的数量  -> *2=3533 张  *2.2 -> 3883
    # select_num = math.ceil( pow((h/patch_size)*(w/patch_size),0.45)) # 被挑选的数量 -> 1085张
    n_patches = max(n_patches, select_num)

    for n in range(n_patches):
        if need_filter:
            if n<=n_patches*0.7: # 0.75 选mask就行
                # 筛选前景
                count = 0
                while count <= 3:
                    rand_i = int(np.random.randint(0, h - patch_size - 0))
                    rand_j = int(np.random.randint(0, w - patch_size - 0))
                    img_patch = img[rand_i:rand_i + patch_size, rand_j:rand_j + patch_size]
                    mask_patch = mask[rand_i:rand_i + patch_size, rand_j:rand_j + patch_size]
                    if img_patch.mean()>231:  # 筛选前景.选到前景就再来一次
                        continue
                    # 找到有mask的
                    if mask_patch.mean() > 13 and mask_patch.mean() < 255:
                        break
                    count += 1
            else:  # 剩下0.25 选前景就行
                while True:
                    rand_i = int(np.random.randint(0, h - patch_size - 0))
                    rand_j = int(np.random.randint(0, w - patch_size - 0))
                    img_patch = img[rand_i:rand_i + patch_size, rand_j:rand_j + patch_size]
                    mask_patch = mask[rand_i:rand_i + patch_size, rand_j:rand_j + patch_size]
                    if img_patch.mean()<231:  # 筛选前景.选到前景就再来一次
                        break
        else:
            rand_i = int(np.random.randint(100, h - patch_size - 100))
            rand_j = int(np.random.randint(100, w - patch_size - 100))
            img_patch = img[rand_i:rand_i + patch_size, rand_j:rand_j + patch_size]
            mask_patch = mask[rand_i:rand_i + patch_size, rand_j:rand_j + patch_size]

        patch_name = name.replace(suffix, f'_{n}.png')
        io.imsave(str(target_img_dir / patch_name), img_patch, check_contrast=False)
        io.imsave(str(target_mask_dir / patch_name), mask_patch, check_contrast=False)


def process_img_and_mask_neg(img_path, target_img_dir, target_mask_dir, n_patches=2, need_filter=True):
    '''
         在neg类中 随机挑选patch
         # need_filter : 需要筛选（也就是选取有意义patch）
     '''
    Image.MAX_IMAGE_PIXELS = None
    img = io.imread(img_path)
    h, w = img.shape[:2]
    name = img_path.name
    suffix = img_path.suffix  # 后缀

    # select_num = int( (h/patch_size)*(w/patch_size))//4 # 被挑选的数量
    # select_num = math.ceil ( max(h//patch_size,w//patch_size) ) # 被挑选的数量  -> 1261 张
    select_num = round( max(h//patch_size,w//patch_size) ) # 被挑选的数量  -> 1261 张
    # select_num = math.ceil( pow((h/patch_size)*(w/patch_size),0.45)) # 被挑选的数量 -> 1085张
    n_patches = max(n_patches, select_num)

    if (h + 20) < patch_size or (w + 20) < patch_size:
        return

    mask = np.zeros((patch_size, patch_size)).astype('uint8')

    for n in range(n_patches):
        if need_filter:
            while True:
                rand_i = int(np.random.randint(0, h - patch_size))
                rand_j = int(np.random.randint(0, w - patch_size))
                img_patch = img[rand_i:rand_i + patch_size, rand_j:rand_j + patch_size]
                if img_patch.mean() < 230:
                    break
        else:
            rand_i = int(np.random.randint(50, h - patch_size))
            rand_j = int(np.random.randint(50, w - patch_size))
            img_patch = img[rand_i:rand_i + patch_size, rand_j:rand_j + patch_size]

        # patch_name = "negative_"+name.replace(suffix, f'_{n}{suffix}')
        patch_name = "negative_" + name.replace(suffix, f'_{n}.png')
        io.imsave(str(target_img_dir / patch_name), img_patch, check_contrast=False)
        io.imsave(str(target_mask_dir / patch_name), mask, check_contrast=False)


# 切分主要运行函数
def run(dataset_path, output, class_type="pos", need_filter=False):
    dataset_path = Path(dataset_path)  # 样本地址
    img_dir = dataset_path / 'images'
    mask_dir = dataset_path / 'masks'

    if not dataset_path.exists() or not img_dir.exists() or not mask_dir.exists():
        print("数据地址不存在")
        sys.exit()

    img_paths = sorted(img_dir.iterdir())
    mask_paths = sorted(mask_dir.iterdir())

    # 获得保存地址
    target_img_dir, target_mask_dir = save_dir(output)

    print('\nSplitting into patches 分裂成patch ...')

    # 并行计算
    executor = Parallel(n_jobs=12)
    # POS
    if class_type == "pos":
        executor(delayed(process_img_and_mask)(img_path, mask_path, target_img_dir, target_mask_dir, n_patches=5,
                                               need_filter=need_filter)
                 for img_path, mask_path in tqdm(zip(img_paths, mask_paths), total=len(img_paths)))
    elif class_type == "neg":
        executor(delayed(process_img_and_mask_neg)(img_path, target_img_dir, target_mask_dir, n_patches=2)
                 for img_path in tqdm(img_paths, total=len(img_paths)))

# 查看pos里面patch 有mask的patch有多少个
def count_mask(mask_dir):
    count = 0
    mask_dir = Path(mask_dir)
    mask_paths = sorted(mask_dir.iterdir())
    for mask_path in tqdm(mask_paths):
        mask = io.imread(mask_path)
        if mask.mean() > 14 and mask.mean() < 255:
        # if mask.mean() > 15 and mask.mean() < 255:
            count+=1
    print(count)
    print("mask所占比例：",count/len(mask_paths))


if __name__ == '__main__':
    patch_size = 800  # 切分patch的大小

    # ---------------------------  训练集
    # pos_dataset_path = r"D:\组会内容\data\Digestpath2019\Colonoscopy_tissue_segment_dataset\pos"  # 训练集 WSI 地址
    # pos_output =Path(r"D:\组会内容\data\Digestpath2019\MedT\train\all_foreground\patch_800\train/pos_800")     # 训练集patch输出地址
    #
    # neg_dataset_path = r"D:\组会内容\data\Digestpath2019\Colonoscopy_tissue_segment_dataset\neg"
    # neg_output =Path(r"D:\组会内容\data\Digestpath2019\MedT\train\all_foreground\patch_800\train/neg_800")     # 输出地址
    # ----------------------------

    # --------------------------- val集
    pos_dataset_path = r"D:\组会内容\data\Digestpath2019\tissue-testset-final\pos"
    pos_output = Path(r"D:\组会内容\data\Digestpath2019\MedT\train\all_foreground\patch_800\val/pos_800")  # 输出地址
    #
    neg_dataset_path = r"D:\组会内容\data\Digestpath2019\tissue-testset-final\neg"
    neg_output =Path(r"D:\组会内容\data\Digestpath2019\MedT\train\all_foreground\patch_800\val/neg_800")  # 输出地址
    # ---------------------------


    run(pos_dataset_path, pos_output, class_type="pos", need_filter=True)  # 处理pos样本
    run(neg_dataset_path, neg_output, class_type="neg", need_filter=True)  # 处理neg样本


    # print("合并输出___")
    save_output = pos_output.parent # 合并输出地址
    move_fire(ori_path = pos_output,save_path = save_output,suffix="")  # 转移pos
    move_fire(ori_path = neg_output,save_path = save_output,suffix="")  # 转移neg

    # 计算pos中mask所占比例
    mask_dir= pos_output / "labelcol"
    count_mask(mask_dir)




