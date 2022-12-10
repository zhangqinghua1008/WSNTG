"""
Training module.
"""
import logging
import fire

from models import initialize_trainer
from utils_network.metrics import accuracy, dice, iou_score, dice_coef
import torch
import numpy as np
import random


# 指定生成随机数的种子，从而每次生成的随机数都是相同的，通过设定随机数种子的好处是，使模型初始化的可学习参数相同，从而使每次的运行结果可以复现。
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def fit(model='fcn', **kwargs):  # model  = tgcn \ test \ wesup \ testunet \ cdws \ sizeloss \   \ fcn \ unext \ WSGNet
    print("model: ", model)
    # Initialize logger. 初始化日志记录器。
    logger = logging.getLogger('Train')  # 初始化变量，声明logger对象
    logger.setLevel(logging.DEBUG)  # 指定日志文件的输出级别
    logger.addHandler(logging.StreamHandler())  # 添加一个Handler. Handler基于日志级别对日志进行分发，如设置为WARNING级别的Handler只会处理WARNING及以上级别的日志。

    trainer = initialize_trainer(model, logger=logger, **kwargs)

    metrics_fun = [accuracy, dice, iou_score]  # 度量函数  , detection_f1, object_dice  //object_hausdorff有点慢

    try:
        # dataset_path = r"D:\组会内容\data\GlaS\data_glas_Label"  # glas
        dataset_path = r"D:\组会内容\data\CRAG\CRAG"  # CRAG
        # dataset_path = r"D:\组会内容\data\LUSC"  # LUSC
        # 执行训练， BaseTrainer类下的train方法
        trainer.train(dataset_path, model, metrics=metrics_fun, **kwargs)
    finally:
        print("Train End")


if __name__ == '__main__':
    fire.Fire(fit)
