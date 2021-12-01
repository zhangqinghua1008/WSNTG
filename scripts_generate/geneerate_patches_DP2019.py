# -*- coding:utf-8 -*-
# @Time   : 2021/12/1 17:11
# @Author : 张清华
# @File   : geneerate_patches_DP2019.py
# @Note   :


"""
    把Digestpath数据集切分成pacth
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


def resize_img(img, target_size):
    img = resize(img, target_size, order=1, anti_aliasing=False)
    return (img * 255).astype('uint8')

def process_img_and_mask(img_path, mask_path, target_img_dir, target_mask_dir, need_filter=False, n_patches=12):
    '''
        随机挑选patch
        # need_filter : 需要筛选（也就是选取有意义patch）
    '''
    Image.MAX_IMAGE_PIXELS = None
    img = io.imread(img_path)
    mask = io.imread(mask_path)
    h, w = img.shape[:2]
    name = img_path.name
    suffix = img_path.suffix  # 后缀

    for n in range(n_patches):
        if need_filter:
            # 筛选前景
            count = 0
            while count <= 4:
                rand_i = int(np.random.randint(0, h - patch_size - 0))
                rand_j = int(np.random.randint(0, w - patch_size - 0))
                img_patch = img[rand_i:rand_i + patch_size, rand_j:rand_j + patch_size]
                mask_patch = mask[rand_i:rand_i + patch_size, rand_j:rand_j + patch_size]
                # if img_patch.mean()<233:
                if mask_patch.mean() > 13 and mask_patch.mean() < 255:
                    break
                count += 1
            if count > 4:  # 找了次还没找到，说明难找
                continue
        else:
            rand_i = int(np.random.randint(100, h - patch_size - 100))
            rand_j = int(np.random.randint(100, w - patch_size - 100))
            img_patch = img[rand_i:rand_i + patch_size, rand_j:rand_j + patch_size]
            mask_patch = mask[rand_i:rand_i + patch_size, rand_j:rand_j + patch_size]

        patch_name = name.replace(suffix, f'_{n}.png')
        io.imsave(str(target_img_dir / patch_name), img_patch, check_contrast=False)
        io.imsave(str(target_mask_dir / patch_name), mask_patch, check_contrast=False)


def process_img_and_mask_neg(img_path, target_img_dir, target_mask_dir, n_patches=6, need_filter=False):
    Image.MAX_IMAGE_PIXELS = None
    img = io.imread(img_path)
    h, w = img.shape[:2]
    name = img_path.name
    suffix = img_path.suffix  # 后缀

    if (h + 50) < patch_size or (w + 50) < patch_size:
        return

    mask = np.zeros((patch_size, patch_size)).astype('uint8')

    for n in range(n_patches):
        if need_filter:
            while True:
                rand_i = int(np.random.randint(0, h - patch_size))
                rand_j = int(np.random.randint(0, w - patch_size))
                img_patch = img[rand_i:rand_i + patch_size, rand_j:rand_j + patch_size]
                if img_patch.mean() < 233:
                    break
        else:
            rand_i = int(np.random.randint(50, h - patch_size))
            rand_j = int(np.random.randint(50, w - patch_size))
            img_patch = img[rand_i:rand_i + patch_size, rand_j:rand_j + patch_size]

        # patch_name = "negative_"+name.replace(suffix, f'_{n}{suffix}')
        patch_name = "negative_" + name.replace(suffix, f'_{n}.png')
        io.imsave(str(target_img_dir / patch_name), img_patch, check_contrast=False)
        io.imsave(str(target_mask_dir / patch_name), mask, check_contrast=False)


def save_dir(output):
    output = Path(output)
    output.mkdir(exist_ok=True)
    target_img_dir = output / 'img'
    target_mask_dir = output / 'labelcol'
    target_img_dir.mkdir(exist_ok=True)
    target_mask_dir.mkdir(exist_ok=True)
    return target_img_dir, target_mask_dir


# 切分主要运行函数
def fun(dataset_path, output, class_type="pos", need_filter=False):
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

    print('\nSplitting into patches 分裂成补丁 ...')
    # for img_path, mask_path in tqdm(zip(img_paths, mask_paths), total=len(img_paths)):
    # process_img_and_mask(img_path, mask_path,target_img_dir,target_mask_dir,n_patches=10)

    # 并行计算
    executor = Parallel(n_jobs=12)
    # POS
    if class_type == "pos":
        executor(delayed(process_img_and_mask)(img_path, mask_path, target_img_dir, target_mask_dir, n_patches=9,
                                               need_filter=need_filter)
                 for img_path, mask_path in tqdm(zip(img_paths, mask_paths), total=len(img_paths)))
    elif class_type == "neg":
        executor(delayed(process_img_and_mask_neg)(img_path, target_img_dir, target_mask_dir, n_patches=4,
                                                   need_filter=need_filter)
                 for img_path in tqdm(img_paths, total=len(img_paths)))


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


# pos_dataset_path = r"D:\组会内容\data\Digestpath2019\Colonoscopy_tissue_segment_dataset\pos"
# pos_output =r"D:\组会内容\data\Digestpath2019\MedT\train\only_mask\train_800/pos_800"     # 输出地址

# neg_dataset_path = r"D:\组会内容\data\Digestpath2019\Colonoscopy_tissue_segment_dataset\neg"
# neg_output =r"D:\组会内容\data\Digestpath2019\MedT\train\only_mask\train_800/neg_800"     # 输出地址

# 测试集
pos_dataset_path = r"D:\组会内容\data\Digestpath2019\tissue-testset-final\pos"
pos_output = r"D:\组会内容\data\Digestpath2019\MedT\val\only_mask/pos_800"  # 输出地址

neg_dataset_path = r"D:\组会内容\data\Digestpath2019\tissue-testset-final\neg"
neg_output = r"D:\组会内容\data\Digestpath2019\MedT\val\only_mask/neg_800"  # 输出地址

# save_output =r"D:\组会内容\data\Digestpath2019\MedT\train\only_mask\train_800"     # 合并输出地址

patch_size = 800  # 切分patch的大小
# patch_resize = 256                                 # 切分完后patch,再resize成多大;

fun(pos_dataset_path, pos_output, class_type="pos", need_filter=True)
fun(neg_dataset_path, neg_output, class_type="neg", need_filter=True)

# print("合并输出___")
# move_fire(ori_path = pos_output,save_path = save_output,suffix="")  # 转移pos
# move_fire(ori_path = neg_output,save_path = save_output,suffix="")  # 转移neg


# print('\nSplitting into patches 分裂成补丁 ...')
# executor = Parallel(n_jobs=8)
# executor(delayed(process_img_and_mask)(img_path, mask_path)
#          for img_path, mask_path in tqdm(zip(img_paths, mask_paths), total=len(img_paths)))


# executor(delayed(process_img_and_mask_neg)(img_path)
#          for img_path in tqdm(img_paths, total=len(img_paths)))
# for img_path in tqdm(img_paths, total=len(img_paths)):
#     process_img_and_mask_neg(img_path)
