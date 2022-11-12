"""
Script for generating area information, used in CWDS-MIL.
生成区域信息的脚本，在CWDS-MIL中使用。
"""

import argparse
import sys
import os

import pandas as pd
from tqdm import tqdm
from skimage.io import imread


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Area information generator. 区域信息生成器。')
    parser.add_argument('--root_dir',default=r"D:\组会内容\data\CRAG\CRAG\train",
                        help='Path to data root directory with mask-level annotation. 带掩码级注释的数据根目录。')
    args = parser.parse_args()

    mask_dir = os.path.join(args.root_dir, 'masks')  # mask 地址
    if not os.path.exists(mask_dir):
        print('Cannot generate area information without masks. 不能生成没有遮罩的区域信息')
        sys.exit(1)

    area_info = pd.DataFrame(columns=['img', 'area'])

    pbar = tqdm(enumerate(sorted(os.listdir(mask_dir))), total=len(os.listdir(mask_dir)))
    for idx, img_name in pbar:
        img_path = os.path.join(mask_dir, img_name)
        img = imread(img_path)
        print("max:===", img.max())
        if img.max()>1:
            img[img <= 127] = 0   # 将掩码的像素从[0,255]转换为[0,1]
            img[img > 127] = 1
        area_info.loc[idx] = [img_name, img.mean()]

    output_path = os.path.join(args.root_dir, 'area.csv')
    area_info.to_csv(output_path)
    print(f'Area information saved to {output_path}.')
