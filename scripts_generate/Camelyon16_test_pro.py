# -*- coding:utf-8 -*-
# @Time   : 2022/3/15 10:05
# @Author : 张清华
# @File   : Camelyon16_test_pro.py
# @Note   :
# 消除警告
import warnings
warnings.filterwarnings("ignore")

import os
import skimage.io as io
import time
import sys

import random
from pathlib import Path
import numpy as np
from tqdm import tqdm
from PIL import Image
import openslide
import matplotlib.pyplot as plt

Image.MAX_IMAGE_PIXELS = None




# 生成slide的缩略图（thumbnail）
def generate_thumbnail():
    img_folder = Path(r"F:\data2\Camelyon16\testing\images")
    output_folder = Path(r"F:\data2\Camelyon16\testing\image_thumbnail")  # 放在c盘

    level = 2

    img_files = img_folder.iterdir()
    for img_file in tqdm(img_files):

        img_stem = img_file.stem   # test_001

        slide = openslide.OpenSlide( str(img_file) )  # 读入WSI
        down_size = slide.level_dimensions[level]

        print(down_size)
        slide_thumbnail = slide.get_thumbnail(down_size)

        slide_thumbnail.save( output_folder / (img_stem + ".png"))


"""
    根据原图文件生成全黑mask.
"""
def generate_blackMask(img_dir=""):

    if not os.path.exists(img_dir):
        print('没找到该文件夹')
        sys.exit(1)

    #存放点 mask的地址
    root_dir = os.path.dirname(img_dir)     # G:/dataG/CAMELYON16/training/patches_level2
    folder_name = os.path.basename(img_dir)  # normal_patches_from_normal_slides
    Mask_dir = os.path.join(root_dir, folder_name + '_mask')
    if not os.path.exists(Mask_dir):
        os.mkdir(Mask_dir)

    print('Generating mask ...')
    #对某一个图像的mask 进行处理
    for fname in tqdm(os.listdir(img_dir) ):
        slide = io.imread(img_dir+"/"+fname)
        basename = os.path.splitext(fname)[0]

        mask = np.zeros((slide.shape[0],slide.shape[1]))
        mask = mask.astype(np.uint8)

        #保存图像中
        io.imsave(os.path.join(Mask_dir, f'{basename}.png'),mask)

def count_tumor():
    data_dir = Path(r"G:\dataG\CAMELYON16\training\patches_level2_Tumor_4000\train\tumor_patches_from_tumor_slides_mask")
    data_dir = Path(r"G:\dataG\CAMELYON16\training\patches_level2_Tumor_4000\tumor_patches_from_tumor_slides_mask_changed")

    file_count = 0
    count = 0

    for file in data_dir.iterdir():
        img = io.imread(file)
        print( file.name, np.unique(img) )
        if( np.max(img) < 1):
            count+=1
        file_count+=1
    print("文件总数：",file_count, "    mask数量",count)

def change_mask():
    '''
        将mask中的【0，1，255】 -> [0 , 255]
    '''
    mask_dir = Path(r"G:\dataG\CAMELYON16\training\patches_level2_Tumor_3000\tumor_patches_from_tumor_slides_mask")
    save_dir = mask_dir.parent / "tumor_patches_from_tumor_slides_mask_changed"
    save_dir.mkdir(exist_ok=True)

    for file in tqdm(mask_dir.iterdir()):
        file_name = file.name

        mask = io.imread(file)
        if 1 in np.unique(mask):
            mask[ mask==1 ] = 255
        io.imsave( save_dir / file_name ,mask)


# 随机移动tumor图像，花费val集合使用
def copyAndremove_img_tumor(path_dir = ""):
    '''
        移动图像，花费val集合使用
    '''
    path_dir = Path(path_dir)

    tumor_img_dir = path_dir / "tumor_patches_from_tumor_slide"
    tumor_mask_dir = path_dir / "tumor_patches_from_tumor_slides_mask"
    tumor_img_save_dir = tumor_img_dir.parent / ( tumor_img_dir.name + "_val")
    tumor_img_save_dir.mkdir(exist_ok=True)
    tumor_mask_save_dir = tumor_mask_dir.parent / (tumor_mask_dir.name + "_val")
    tumor_mask_save_dir.mkdir(exist_ok=True)

    # normal
    normal_img_dir = path_dir / "normal_patches_from_tumor_slides"
    normal_mask_dir = path_dir / "normal_patches_from_tumor_slides_mask"
    normal_img_save_dir = normal_img_dir.parent / (normal_img_dir.name + "_val")
    normal_img_save_dir.mkdir(exist_ok=True)
    normal_mask_save_dir = normal_mask_dir.parent / (normal_mask_dir.name + "_val")
    normal_mask_save_dir.mkdir(exist_ok=True)

    for i in range(0,1):
        # num = random.randint(70, 111)  # 紫色
        num = random.randint(0, 70)  # 红色
        # num = random.randint(0, 112)  # 随机

        print(num)

        # =======================tumor
        # move img
        for file in tumor_img_dir.glob("tumor_" + str(num).zfill(3) + "*"):
            # print(file)
            img = io.imread(file)
            io.imsave( tumor_img_save_dir / file.name , img )
            file.unlink(file)  # 删除掉原来的

        # move mask
        for file in tumor_mask_dir.glob("tumor_" + str(num).zfill(3) + "*"):
            mask = io.imread(file)
            io.imsave(tumor_mask_save_dir / file.name, mask)
            file.unlink(file)  # 删除掉原来的

        # ======================= normal
        for file in normal_img_dir.glob("tumor_" + str(num).zfill(3) + "*"):
            # print(file)
            img = io.imread(file)
            io.imsave(normal_img_save_dir / file.name, img)
            file.unlink(file)  # 删除掉原来的

        for file in normal_mask_dir.glob("tumor_" + str(num).zfill(3) + "*"):
            mask = io.imread(file)
            io.imsave(normal_mask_save_dir / file.name, mask)
            file.unlink(file)  # 删除掉原来的


def copyAndremove_img_normal(img_dir = ""):
    '''
        移动图像，花费val集合使用
    '''
    mask_dir = Path(img_dir + "_mask")
    img_dir = Path(img_dir)

    img_save_dir = img_dir.parent / ( img_dir.name + "_val")
    img_save_dir.mkdir(exist_ok=True)

    mask_save_dir = mask_dir.parent / (mask_dir.name + "_val")
    mask_save_dir.mkdir(exist_ok=True)

    for i in range(0,25):
        # num = random.randint(70, 111)  # 紫色
        # num = random.randint(0, 70)  # 红色
        num = random.randint(0, 161)  # 随机

        print(num)

        # move img
        for file in img_dir.glob("normal_" + str(num).zfill(3) + "*"):
            # print(file)
            img = io.imread(file)
            io.imsave( img_save_dir / file.name , img )
            file.unlink(file)  # 删除掉原来的

        # move mask
        for file in mask_dir.glob("normal_" + str(num).zfill(3) + "*"):
            mask = io.imread(file)
            io.imsave(mask_save_dir / file.name, mask)
            file.unlink(file)  # 删除掉原来的



def fun(img_name = "tumor_110_59571_22008_T"):
    img_data_dir = Path(r"G:\dataG\CAMELYON16\training\patches_level2_Tumor_4000\train\tumor_patches_from_tumor_slide")
    mask_data_dir = Path(r"G:\dataG\CAMELYON16\training\patches_level2_Tumor_4000\train"
                    r"\tumor_patches_from_tumor_slides_mask/")
    img_dir = img_data_dir / ( img_name + ".png")
    mask_dir = mask_data_dir / (  img_name + "_mask.png")
    img = io.imread(img_dir)
    mask = io.imread(mask_dir)


    plt.subplot(1,2,1)
    plt.imshow(img)

    plt.subplot(1,2,2)
    plt.imshow(mask)

    plt.show()

def fun2():
    test_dir = Path(r"G:/dataG/test_code/False/")
    save_dir = Path(r"G:\dataG\test_code\True")
    for file in test_dir.iterdir():
        img = io.imread(file)
        io.imsave(save_dir/file.name,img)
        file.unlink()


if __name__ == '__main__':
    generate_thumbnail()

    # 改变mask中为1的值
    #  change_mask()

    # 生成normal_patches_from_normal_slides的mask
    # generate_blackMask(img_dir=r"G:\dataG\CAMELYON16\training\patches_level2_Tumor_3000\train\normal_patches_from_normal_slides")

    # 生成normal_patches_from_tumor_slides的mask
    # generate_blackMask(img_dir=r"G:\dataG\CAMELYON16\training\patches_level2_Tumor_3000\train\normal_patches_from_tumor_slides")

    # 移动normal
    # copyAndremove_img_normal(r"G:\dataG\CAMELYON16\training\patches_level2_Tumor_3000\train\normal_patches_from_normal_slides")

    # 挑选出val集的img和mask
    # copyAndremove_img_tumor(r"G:\dataG\CAMELYON16\training\patches_level2_Tumor_3000\train")
