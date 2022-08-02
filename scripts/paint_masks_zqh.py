import argparse
import glob
import os
import os.path as osp
from pathlib import Path
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from skimage.io import imread, imsave
from skimage.measure import label
from joblib import Parallel, delayed
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
'''
    分割结果的彩图绘制在 scripts/paint_masks.py
'''

if __name__ == '__main__':
    # pred_path = Path( r"D:\组会内容\实验报告\MedT\records\20220102-2109-PM_tgcn\results_0044\all_test\pos\_post")
    pred_path = Path( r"D:\组会内容\实验报告\MedT\records\20220102-2109-PM_tgcn\results_0044\all_test\pos\_post")
    model = "tgcn"
    output_dir = pred_path.parent / 'painting_gnuplot'
    output_dir.mkdir(exist_ok=True)

    print('Reading predictions  ...')

    executor = Parallel(os.cpu_count())
    # pred_list = executor(delayed(imread)(mask_path) for mask_path in pred_masks)

    for pred in tqdm( pred_path.glob("*.png") ):
        img = imread(pred)
        if img.size < 10000*10000:
            pred_name = f'{pred.stem}_{model}.png'
            pred = label(img)
            plt.imsave( output_dir / pred_name ,pred,cmap = 'gnuplot')

    print('Done')
