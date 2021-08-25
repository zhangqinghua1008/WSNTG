"""
Script for visualizing point annotation.  可视化点注释的脚本。
"""
import argparse
import csv
import os
import cv2
from tqdm import tqdm
from skimage.io import imread, imsave
from skimage.segmentation import mark_boundaries
from joblib import Parallel, delayed

COLORS = (
    (0, 255, 0),
    (255, 0, 0),
)

parser = argparse.ArgumentParser()
parser.add_argument('--point_root', default=r'G:\py_code\pycharm_Code\WESUP——修改版本\data_glas\train\points-3e-05',
                    help='Path to point labels directory')
parser.add_argument('-r', '--radius', type=int, default=5, help='圆半径')
parser.add_argument('-o', '--output',
                    help='存储可视化结果的输出路径')
args = parser.parse_args()

output_dir = args.output or os.path.join(args.point_root, 'point_viz')

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

img_dir = os.path.join(os.path.dirname(args.point_root), 'images')
mask_dir = os.path.join(os.path.dirname(args.point_root), 'masks')
print(f'___Generating dot annotation visualizaion to {output_dir} ...')


def para_func(img_name):
    basename = os.path.splitext(img_name)[0]
    img = imread(os.path.join(img_dir, img_name))
    mask = imread(os.path.join(mask_dir, img_name)) if os.path.exists(mask_dir) else None

    # handle PNG files with alpha channel
    if img.shape[-1] == 4:
        img = img[..., :3]

    # mark boundaries if mask is present
    if mask is not None:
        img = (mark_boundaries(img, mask, mode='thick') * 255).astype('uint8')

    csvfile = open(os.path.join(args.point_root, f'{basename}.csv'))
    csvreader = csv.reader(csvfile)

    for point in csvreader:
        print(point)
        point = [int(d) for d in point]
        cv2.circle(img, (point[0], point[1]), args.radius, COLORS[point[2]], -1)

    imsave(os.path.join(output_dir, img_name), img, check_contrast=False)
    csvfile.close()

# n_jobs = os.cpu_count() 使用电脑cpu核数个数的任务 进行并行运算
# tqdm 进度条
Parallel(n_jobs=os.cpu_count())(delayed(para_func)(img_name) for img_name in tqdm(os.listdir(img_dir)) )
