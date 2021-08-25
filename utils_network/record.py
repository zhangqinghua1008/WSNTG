"""
Utilities for recording multiple runs of experiments. 用于记录多次试验运行的实用程序。
"""

import glob
import json
import os
from datetime import datetime
from pathlib import Path
from shutil import copyfile, copytree, rmtree

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use('Agg')

# 主备记录路径
def prepare_record_dir():
    """Create new record directory and return its path. 创建新的记录目录并返回其路径 """

    # record_root = Path.home() / 'records'  # 系统用户目录 （cz源代码）
    record_root = Path.cwd() / 'records'  # path.cwd() = 运行原始py文件的当前目录（不是该文件的原始目录）  .parent 父目录
    if os.environ.get('RECORD_ROOT'):
        record_root = Path(os.environ.get('RECORD_ROOT')).expanduser()

    if not record_root.exists():
        record_root.mkdir()

    record_dir = record_root / datetime.now().strftime('%Y%m%d-%I%M-%p')

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
        复制所有源脚本到记录目录进行复制 """

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
                # plot metrics computed only on validation phase
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
