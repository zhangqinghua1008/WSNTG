"""
Script for generating point annotation. 生成点注释的脚本。
"""
import argparse
import csv
import os
import sys
import numpy as np
from tqdm import tqdm
from PIL import Image
from skimage.measure import label
from joblib import Parallel, delayed

def _sample_within_region(region_mask, class_label, num_samples=1):
    '''
        在某个区域内找到点
    :param region_mask: 该区域某一个实例的mask图，对象为二维list，里面的值为bool
    :param class_label: 类标签，int型
    :param num_samples:  找到几个point
    :return: 返回point (x,y,label)
    '''
    #返回该mask内所有的点坐标
    xs, ys = np.where(region_mask)   # numpy.where() 函数返回输入数组中满足给定条件的元素的索引。

    # 如果只取一个点
    if num_samples == 1:
        # 求中心点坐标   round():返回指定数字的四舍五入值。
        x_center, y_center = int(xs.mean().round()), int(ys.mean().round())

        retry = 0
        while True:
            # deviate from the center within a circle within radius 5  偏离半径为5的圆的中心
            # random.randint(-5:6) 随机生成[-5,6) 之间的整数
            x = x_center + np.random.randint(-5, 6)
            y = y_center + np.random.randint(-5, 6)

            # if the center point is inside the region, return it  如果中心点在区域内，返回它
            try:
                if region_mask[x, y]:
                    return np.c_[x, y, class_label]  # np.c_ : 是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等。
            except IndexError:
                pass
            finally:
                retry += 1

            if retry > 5:
                break

    selected_indexes = np.random.permutation(len(xs))[:num_samples]  # 对0-len(xs) 之间的序列进行随机排序,并去前num_samples个
    xs, ys = xs[selected_indexes], ys[selected_indexes]

    return np.c_[xs, ys, np.full_like(xs, class_label)]  # np.c_ : 是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等。


#生成点，返回一个二维list，每一行有y,x,类别
def _generate_points(mask, point_ratio=1e-4): # point_ratio 是点比例
    points = []

    # loop over all class labels  循环所有类标签
    # (from 0 to n_classes, where 0 is background) 从0到n_classes，其中0是background
    for class_label in np.unique(mask):
        class_mask = mask == class_label  # class_mask 返回一个和mask大小相同的array对象，里面的值= True/False
        if class_label == 0:
            # if background, randomly sample some points  如果背景，随机抽取一些点
            points.append(
                _sample_within_region(
                    class_mask, class_label,
                    num_samples=int(class_mask.sum() * point_ratio)
                )
            )
        else:
            class_mask = label(class_mask)  # label方法将图像的每个类别的每个联通区域使用不同的值标记出来，不属于这个类的会被标为0
            # 求出区域的个数
            region_indexes = np.unique(class_mask)
            # 把0去掉，因为0是背景，而不是这个类的数量
            region_indexes = region_indexes[np.nonzero(region_indexes)]

            # iterate over all instances of this class  从第一个实例到n ， 遍历该类的所有实例
            for idx in np.unique(region_indexes):
                region_mask = class_mask == idx  #找到该实例的一个True/False图
                num_samples = max(1, int(region_mask.sum() * point_ratio))  # 从这个实例中 选取几个点标签， 通常是一个

                points.append(
                    _sample_within_region(
                        region_mask, class_label, num_samples=num_samples )
                )

    return np.concatenate(points)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Dot annotation generator. 点注释生成器。')
    parser.add_argument('-r','--root_dir', type=str, default=r"G:\py_code\pycharm_Code\WESUP-TGCN\data_glas\train",
                        help='带掩码级注释的数据根目录的路径')
    parser.add_argument('-p', '--point-ratio', type=float, default=3e-5,  # 参数用于控制已标记像素的百分比。default=1e-4
                        help='Percentage of labeled objects (regions) for each class')  # 每个类标记对象(区域)的百分比
    args = parser.parse_args()     # 最后调用parse_args()方法进行解析；

    if not os.path.exists(args.root_dir):
        print('数据地址不存在')
        sys.exit(1)

    mask_dir = os.path.join(args.root_dir, 'masks')
    if not os.path.exists(mask_dir):
        print('没有masks无法生成点注释。')
        sys.exit(1)

    #存放点标签的地址
    label_dir = os.path.join(args.root_dir, f'points-{str(args.point_ratio)}')
    if not os.path.exists(label_dir):
        os.mkdir(label_dir)

    print('正在产生点注释 ...')

    #对某一个图像的mask 进行处理
    def para_func(fname):
        basename = os.path.splitext(fname)[0]
        mask = np.array(Image.open(os.path.join(mask_dir, fname)))  # 读取 mask 并转换成array
        #生成点
        points = _generate_points(mask, point_ratio=args.point_ratio)

        # conform to the xy format 应该是 把二维的坐标 转换一下
        # 此时points是一个二维list，每一行都代表一个点（有三列），把一二列的数值交换
        points[:, [0, 1]] = points[:, [1, 0]]

        #保存到mask中
        with open(os.path.join(label_dir, f'{basename}.csv'), 'w', newline="") as fp:
            writer = csv.writer(fp)
            writer.writerows(points)

        return len(points)

    executor = Parallel(n_jobs=os.cpu_count())  # 并行化计算
    # 对mask_dir中的每个图像mask,进行para_func函数.
    points_nums = executor(    delayed(para_func)(fname) for fname in tqdm(os.listdir(mask_dir) )   )
    print(f'Average number of points: {np.mean(points_nums)}.')
