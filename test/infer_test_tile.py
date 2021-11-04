import argparse
import lib
import torch
from infer_test_tile_utils import *
from models import initialize_trainer
from performance_metrics import *
"""
    Inference module for window-based strategy.  基于窗口滑动的推理模块。
"""

def infer(model,data_dir,patch_size,device,output_dir=None):
    data_dir = Path(data_dir)
    img_paths = list((data_dir / 'img').iterdir())
    print(f'Predicting {len(img_paths)} images from {data_dir} ...')

    for img_path in tqdm(img_paths,ncols=50):
        pre = predict_bigimg(model, img_path, patch_size, device=device)

        if output_dir is not None:
            save_pre(pre, img_path, output_dir)


def main():
    data_dir = r"D:\组会内容\data\Digestpath2019\MedT\test/"
    checkpoint = r"D:\组会内容\实验报告\MedT\records\20211103-1026-PM\checkpoints/ckpt.0061.pth"
    model_type = 'wesup'
    patch_size = 256

    output_dir = r"D:\组会内容\实验报告\MedT\records\Digestpath_WSI_results_WESUP/temp"

    # 加载模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = initialize_trainer(model_type, device=device)
    if checkpoint is not None:
        trainer.load_checkpoint(checkpoint)

    infer(trainer,data_dir,patch_size = patch_size ,output_dir=output_dir,device=device)

    # 测指标
    pre_dir = output_dir
    lable_dir = r"D:\组会内容\data\Digestpath2019\MedT\test\labelcol"
    performance_metrics(pre_dir,lable_dir)

main()