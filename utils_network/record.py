"""
Utilities for recording multiple runs of experiments. 用于记录多次试验运行的实用程序。
"""

import glob
import json
from datetime import datetime
from pathlib import Path
from shutil import copyfile, copytree, rmtree
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torchvision import utils as vutils

matplotlib.use('Agg')


def chk_mkdir(dir_path):
    """ Creates folders if they do not exist. """
    if not dir_path.exists():
        dir_path.mkdir()


# 主备记录路径
def prepare_record_dir(model_type):
    """Create new record directory and return its path. 创建新的记录目录并返回其路径 """

    data_set = "records"  # records (DP)  | records_SICAPV2 \ records_16
    # if model_type == 'tgcn' or model_type == 'wesup':
    #     record_root = Path('D:/组会内容/实验报告/MedT') / data_set  # 存放recoder的地址
    # else:
    record_root = Path('D:/组会内容/实验报告/MedT') / data_set / '0对比算法'  # 存放recoder的地址

    if not record_root.exists():
        record_root.mkdir()

    record_dir = record_root / (datetime.now().strftime('%Y%m%d-%H%M-%p') + "_" + model_type)  # %H 24小时制

    if not record_dir.exists():
        record_dir.mkdir()

    checkpoint_dir = record_dir / 'checkpoints'
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir()

    return record_dir


def save_params(record_dir, params):
    """Save experiment parameters to record directory.
        保存实验参数到记录目录,保存为一个json文件  """
    params_dir = record_dir / 'params'

    if not params_dir.exists():
        params_dir.mkdir()

    num_of_runs = len(list(params_dir.iterdir()))

    with open(params_dir / f'{num_of_runs}.json', 'w') as fp:
        json.dump(params, fp, indent=4)


def copy_source_files(record_dir):
    """Copy all source scripts to record directory for reproduction.
        复制所有源脚本到记录目录进行复制
    """
    source_dir = record_dir / 'source'
    if source_dir.exists():
        rmtree(source_dir)
    source_dir.mkdir()

    for source_file in glob.glob('*.py'):
        copyfile(source_file, source_dir / source_file)

    copytree('utils_network', source_dir / 'utils_network')
    copytree('models', source_dir / 'models')
    copytree('scripts', source_dir / 'scripts')


def plot_learning_curves(history_path):
    """Read history csv file and plot learning curves.
        读取历史csv文件，绘制学习曲线
    """
    history = pd.read_csv(history_path)
    record_dir = history_path.parent
    curves_dir = record_dir / 'curves'

    if not curves_dir.exists():
        curves_dir.mkdir()

    for key in history.columns:
        if key.startswith('val_'):
            if key.replace('val_', '') not in history.columns:
                # plot metrics computed only on validation phase  plot度量,仅有验证阶段
                plt.figure(dpi=200)
                plt.title('Model ' + key.replace('val_', ''))
                plt.plot(history[key])
                plt.ylabel(key.replace('val_', '').capitalize())
                plt.xlabel('Epoch')
                plt.grid(True)
                plt.savefig(curves_dir / f'{key}.png')
            continue

        plt.figure(dpi=200)
        try:
            plt.plot(history[key])
            plt.plot(history['val_' + key])
        except KeyError:
            pass

        plt.title('Model ' + key)
        plt.ylabel(key.capitalize())
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'])
        plt.grid(True)
        plt.savefig(curves_dir / f'{key}.png')
        plt.close()


def save_preAndMask(record_dir, pred, target, index):
    """ 保存模型pre 和 对应的mask  """
    img_dir = record_dir / 'pre_output'

    if not img_dir.exists():
        img_dir.mkdir()

    pred_save = pred[0].clone().detach().to(torch.device('cpu')).float()
    target_save = target[0].clone().detach().to(torch.device('cpu')).float()
    vutils.save_image(pred_save, Path.joinpath(img_dir, str(index) + ".png"))
    vutils.save_image(target_save, Path.joinpath(img_dir, str(index) + "_mask.png"))
