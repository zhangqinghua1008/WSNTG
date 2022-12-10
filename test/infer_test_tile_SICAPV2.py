import argparse
import lib
import torch
from pathlib import Path
from infer_test_tile_utils import *
from models import initialize_trainer
from performance_metrics import *
from models.wesup import WESUPPixelInference
from models.WSNTG.wsntg import WSNTGPixelInference
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
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
            save_pre(pre, img_path, output_dir / "_pre")

# 像素级别推理
def pixel_infer(model,data_dir,patch_size,resize_size=None,device='cuda',output_dir=None):
    data_dir = Path(data_dir)
    img_paths = list((data_dir / 'img').iterdir())
    print(f'像素级 Predicting {len(img_paths)} images from {data_dir} ...')

    with torch.no_grad():
        for img_path in tqdm(img_paths):
            pre = pixel_predict_bigimg(model, img_path, patch_size, resize_size=resize_size,device=device)

            if output_dir is not None:
                save_pre(pre, img_path, output_dir / "_pre")

def infer_run_SIC(model_type = None,checkpoint = None):
    begin = time.time()
    patch_size = 512
    resize_size = 256  # None

    output_dir = checkpoint.parent.parent / 'results'
    output_dir.mkdir(exist_ok=True)

    # 10张图快速测试
    data_dir = Path(r"D:\组会内容\data\SICAPV2\res\patch3\test/")
    output_dir = output_dir
    output_dir.mkdir(exist_ok=True)

    # 加载模型并推理
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = initialize_trainer(model_type, device=device)
    if checkpoint is not None:
        trainer.load_checkpoint(checkpoint)

    # 推理
    phase_data_dir = data_dir
    phase_output_dir = output_dir
    phase_output_dir.mkdir(exist_ok=True)

    infer(trainer,phase_data_dir,patch_size = patch_size ,output_dir=phase_output_dir,resize_size=resize_size,device=device)
    print("\ncheckpoint:", checkpoint, "\n")
    # 评价模型输出
    phase_lable_dir = phase_data_dir / "labelcol"  # GT 地址
    evaluate_img(modelPre_dir=phase_output_dir, gt_lable_dir=phase_lable_dir, need_post=True)
    print("时间：", time.time() - begin)


# 执行像素级别推理
def infer_pixel_run_SIC(model_type = 'tgcn',checkpoint = None):
    '''
        model_type : wesup / tgcn
    '''
    begin = time.time()
    patch_size = 512
    resize_size = 256  # None

    output_dir = checkpoint.parent.parent / ('results_'+checkpoint.stem[-4:])
    output_dir.mkdir(exist_ok=True)

    data_dir = Path(r"D:\组会内容\data\SICAPV2\res\patch3\test/") # 测试集地址
    output_dir = output_dir
    output_dir.mkdir(exist_ok=True)

    # 加载模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if model_type == 'wesup':
        model = WESUPPixelInference().to(device)
    elif model_type == 'tgcn':
        model = WSNTGPixelInference().to(device)
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint)['model_state_dict'])

    # 推理
    phase_data_dir = data_dir
    phase_output_dir = output_dir
    phase_output_dir.mkdir(exist_ok=True)
    pixel_infer(model, phase_data_dir, patch_size=patch_size, output_dir=phase_output_dir, resize_size=resize_size,
                device=device)

    print("\ncheckpoint:", checkpoint, "\n")
    # 评价模型输出
    phase_lable_dir = phase_data_dir / "labelcol"  # GT 地址
    evaluate_img(modelPre_dir=phase_output_dir, gt_lable_dir=phase_lable_dir, need_post=True)
    print("时间：", time.time() - begin)


if __name__ == '__main__':

    model_type = "yamu"     # sizeloss || unet / fcn / cdws / sizeloss / tgcn /wesup /yamu

    ckpts = ["ckpt.0100.pth","ckpt.0106.pth"]
    for ckpt in ckpts:
        checkpoint = Path(r"D:\组会内容\实验报告\MedT\records_SICAPV2\0对比算法\20220622-1634-PM_yamu")

        checkpoint =  checkpoint / "checkpoints" / ckpt
        if model_type == "tgcn" or model_type == "wesup":
            # 执行像素级别推理
            infer_pixel_run_SIC(model_type=model_type,checkpoint=checkpoint)
        else:
            # 正常模型推理
            infer_run_SIC(model_type=model_type,checkpoint=checkpoint)
