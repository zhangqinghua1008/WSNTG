import argparse
import lib
import torch
from infer_test_tile_utils import *
from models import initialize_trainer
from performance_metrics import *
from models.wesup import WESUPPixelInference
"""
    Inference module for window-based strategy.  基于窗口滑动的推理模块。
"""

def infer(trainer,data_dir,patch_size,resize_size=None,device='cuda',output_dir=None):
    data_dir = Path(data_dir)
    img_paths = list((data_dir / 'img').iterdir())
    print(f'Predicting {len(img_paths)} images from {data_dir} ...')

    trainer.model.eval()
    for img_path in tqdm(img_paths,ncols=50):
        pre = predict_bigimg(trainer, img_path, patch_size, resize_size=resize_size,device=device)

        if output_dir is not None:
            save_pre(pre, img_path, output_dir +"/_pre")
            fast_pred_postprocess(pre, pre.size*0.001)  # 对预测出来的图片进行后处理,并保存
            save_pre(pre, img_path, output_dir + "/_post")

# 像素级别推理
def pixel_infer(model,data_dir,patch_size,resize_size=None,device='cuda',output_dir=None):
    data_dir = Path(data_dir)
    img_paths = list((data_dir / 'img').iterdir())
    print(f'像素级 Predicting {len(img_paths)} images from {data_dir} ...')

    with torch.no_grad():
        for img_path in tqdm(img_paths):
            pre = pixel_predict_bigimg(model, img_path, patch_size, resize_size=resize_size,device=device)

            if output_dir is not None:
                save_pre(pre, img_path, output_dir +"/_pre")
                # pred_postprocess(pre, pre.size*0.001)  # 对预测出来的图片进行后处理,并保存
                # save_pre(pre, img_path, output_dir + "/_post")


def run():
    data_dir = r"D:\组会内容\data\Digestpath2019\MedT\test/"
    checkpoint = r"D:\组会内容\实验报告\MedT\records\20211106-1339-PM\checkpoints/ckpt.0054.pth"
    model_type = 'wesup'
    patch_size = 800
    resize_size = 280  # None

    output_dir = r"D:\组会内容\实验报告\MedT\records\Digestpath_WSI_results_WESUP/temp"

    # 加载模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = initialize_trainer(model_type, device=device)
    if checkpoint is not None:
        trainer.load_checkpoint(checkpoint)

    infer(trainer,data_dir,patch_size = patch_size ,output_dir=output_dir,resize_size=resize_size,device=device)

    # 测指标
    pre_dir = output_dir + "/_post"
    lable_dir = r"D:\组会内容\data\Digestpath2019\MedT\test\labelcol"
    performance_metrics(pre_dir,lable_dir)

def pixel_run():
    data_dir = r"D:\组会内容\data\Digestpath2019\MedT\test/"
    checkpoint = r"D:\组会内容\实验报告\MedT\records\20211106-1339-PM\checkpoints/ckpt.0054.pth"
    patch_size = 800
    resize_size = 280  # None

    output_dir = r"D:\组会内容\实验报告\MedT\records\Digestpath_WSI_results_WESUP/temp"

    # 加载模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = WESUPPixelInference().to(device)
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint)['model_state_dict'])

    pixel_infer(model,data_dir,patch_size = patch_size ,output_dir=output_dir,resize_size=resize_size,device=device)

    # 测指标
    pre_dir = output_dir + "/_pre"
    # lable_dir = r"D:\组会内容\data\Digestpath2019\MedT\test\labelcol"
    lable_dir = r"D:\组会内容\data\Digestpath2019\MedT\test\all_test\labelcol"
    performance_metrics(pre_dir,lable_dir)
pixel_run()


# run()
