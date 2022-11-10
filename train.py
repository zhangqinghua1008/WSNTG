"""
Training module.
"""
import logging
from shutil import rmtree
import fire

from models import initialize_trainer
from utils_network.metrics import accuracy, dice, iou_score,dice_coef
from utils_network.metrics import detection_f1,object_dice,object_hausdorff
import  torch
import numpy as np
import random

# tgcn  /  test_backbone
# 指定生成随机数的种子，从而每次生成的随机数都是相同的，通过设定随机数种子的好处是，使模型初始化的可学习参数相同，从而使每次的运行结果可以复现。
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# model : 选择训练哪个模型
def fit(model='fcn', **kwargs):   # model  = tgcn \ test \ wesup \ testunet \ cdws \ sizeloss \ unet \ fcn \ UNeXt
    setup_seed(123)

    # Initialize logger. 初始化日志记录器。
    logger = logging.getLogger('Train')         # 初始化变量，声明logger对象
    logger.setLevel(logging.DEBUG)              # 指定日志文件的输出级别
    logger.addHandler(logging.StreamHandler())  # 添加一个Handler. Handler基于日志级别对日志进行分发，如设置为WARNING级别的Handler只会处理WARNING及以上级别的日志。

    # 初始化训练模型
    trainer = initialize_trainer(model, logger=logger, **kwargs)

    # 设置选取什么样的度量函数
    metrics_fun = [accuracy, dice, dice_coef, iou_score]  # 度量函数  , detection_f1, object_dice

    try:
        # 数据集地址
        # dataset_path = r"G://py_code//pycharm_Code//WESUP-TGCN//data_glas"
        # dataset_path = r"D://组会内容//data//HoVer_ConSep//consep_cut_512"
        # dataset_path = r"D://组会内容//data//HoVer_ConSep//test_debug"
        # dataset_path = r"D:\组会内容\data\Digestpath2019\MedT\fast_test_model\train_256"
        # dataset_path = r"D:\组会内容\data\Digestpath2019\MedT\train\only_mask\train_800"  # 2801张patch DP2019
        # dataset_path = r"D:\组会内容\data\Digestpath2019\MedT\train\all_foreground\patch_800"  # 5144张patch DP2019
        # dataset_path = r"C:\Zdata\all_foreground\patch_800"  # 5144张patch (放在c盘)
        # dataset_path = r"D:\组会内容\data\SICAPV2\res\patch3"  # SICAPV2
        # dataset_path = r"G:\dataG\CAMELYON16\training\patches_level2_Tumor_4000_new"  # CAMELYON16
        # dataset_path = r"G:\dataG\CAMELYON16\training\patches_level2_Tumor_3000"  # CAMELYON16
        # dataset_path = r"D:\组会内容\data\ISIC2018\isic512DataImagesMask"  # isic512
        dataset_path = r"D:\组会内容\data\GlaS\data_glas_Label"  # glas
        #执行训练， BaseTrainer类下的train方法
        trainer.train(dataset_path, model, metrics=metrics_fun, **kwargs)
    finally:
        print("Train End")

if __name__ == '__main__':
    fire.Fire(fit)
