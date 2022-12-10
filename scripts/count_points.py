import argparse
import os
import glob
import numpy as np
import pandas as pd

# 计算点标签平均数量

parser = argparse.ArgumentParser()
parser.add_argument('--points_path', default=r'G:\py_code\pycharm_Code\WESUP-TGCN\data_glas\train\points',
                    help='Path to point annotations')
args = parser.parse_args()

print(np.mean([len(pd.read_csv(csv_file)) for csv_file in glob.glob(os.path.join(args.points_path, '*.csv'))]))
