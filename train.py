"""
Training module.
"""

import logging
from shutil import rmtree

import fire

from models import initialize_trainer
from utils_network.metrics import accuracy
from utils_network.metrics import dice
import  torch

def fit(model='tgcn', **kwargs):
    # Initialize logger. 初始化日志记录器。
    logger = logging.getLogger('Train')         # 初始化变量，声明logger对象
    logger.setLevel(logging.DEBUG)              # 指定日志文件的输出级别
    logger.addHandler(logging.StreamHandler())  # 添加一个Handler. Handler基于日志级别对日志进行分发，如设置为WARNING级别的Handler只会处理WARNING及以上级别的日志。

    trainer = initialize_trainer(model, logger=logger, **kwargs)

    metrics_fun = [accuracy, dice]   # 度量函数

    try:
        dataset_path = r"/\\data_glas"
        #执行训练， BaseTrainer类下的train方法
        trainer.train(dataset_path, metrics=metrics_fun, **kwargs)
    finally:
        print("Train End")

if __name__ == '__main__':
    fire.Fire(fit)
