import argparse
import lib
import torch
from pathlib import Path
from infer_test_tile_utils import *
from models import initialize_trainer
from performance_metrics import *
from models.wesup import WESUPPixelInference
from models.TGCN.tgcn import TGCNPixelInference
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


# 执行模型正常推理
def infer_run(model_type = None,checkpoint = None,
              test_model = "fast",phases = ["pos"]):
    begin = time.time()
    patch_size = 800
    resize_size = 400  # None

    output_dir = checkpoint.parent.parent / 'results'
    output_dir.mkdir(exist_ok=True)

    # 10张图快速测试
    if test_model == 'fast':
        data_dir = Path(r"D:\组会内容\data\Digestpath2019\MedT\test\fast_test/")
        output_dir = output_dir / "fast_test"
        output_dir.mkdir(exist_ok=True)
    # 完整90张测试图片
    elif test_model == 'all':
        data_dir = Path(r"D:\组会内容\data\Digestpath2019\MedT\test\all_test/")
        output_dir = output_dir / "all_test"
        output_dir.mkdir(exist_ok=True)

    # 加载模型并推理
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = initialize_trainer(model_type, device=device)
    if checkpoint is not None:
        trainer.load_checkpoint(checkpoint)

    # 推理
    for phase in phases:
        print("\n++++++++++++++++++++ 当前正在处理：", phase)
        phase_data_dir = data_dir / phase
        phase_output_dir = output_dir / phase
        phase_output_dir.mkdir(exist_ok=True)

        infer(trainer,phase_data_dir,patch_size = patch_size ,output_dir=phase_output_dir,resize_size=resize_size,device=device)
        print("\ncheckpoint:", checkpoint, "\n")
        # 评价模型输出
        phase_lable_dir = phase_data_dir / "labelcol"  # GT 地址
        evaluate_img(modelPre_dir=phase_output_dir, gt_lable_dir=phase_lable_dir, need_post=True)
        print("时间：", time.time() - begin)

# 执行像素级别推理
def infer_pixel_run(model_type = 'tgcn',checkpoint = None,
                    test_model = "fast",phases = ["pos"]):
    '''
        model_type : wesup / tgcn
        test_model : fast || all
        phases = ["pos","neg"]  # 测试类型    ["pos","neg"] || ["neg"] || ["pos"]
    '''
    begin = time.time()
    patch_size = 800
    resize_size = 280  # None

    output_dir = checkpoint.parent.parent / ('results_'+checkpoint.stem[-4:])
    output_dir.mkdir(exist_ok=True)
    # 10张图快速测试
    if test_model =='fast':
        data_dir = Path(r"D:\组会内容\data\Digestpath2019\MedT\test\fast_test/")
        output_dir = output_dir / "fast_test"
        output_dir.mkdir(exist_ok=True)
    # 完整90张测试图片
    elif test_model == 'all':
        data_dir = Path(r"D:\组会内容\data\Digestpath2019\MedT\test\all_test/")
        output_dir = output_dir / "all_test"
        output_dir.mkdir(exist_ok=True)

    # 加载模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if model_type=='wesup':
        model = WESUPPixelInference().to(device)
    elif model_type=='tgcn':
        model = TGCNPixelInference().to(device)
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint)['model_state_dict'])

    # 推理
    for phase in phases:
        print("\n++++++++++++++++++++ 当前正在处理：",phase)
        phase_data_dir = data_dir / phase
        phase_output_dir = output_dir / phase
        phase_output_dir.mkdir(exist_ok=True)
        pixel_infer(model, phase_data_dir, patch_size=patch_size, output_dir=phase_output_dir, resize_size=resize_size,
                    device=device)

        print("\ncheckpoint:", checkpoint, "\n")
        # 评价模型输出
        phase_lable_dir = phase_data_dir / "labelcol"  # GT 地址
        evaluate_img(modelPre_dir=phase_output_dir, gt_lable_dir=phase_lable_dir, need_post=True)
        print("时间：",time.time()-begin)


if __name__ == '__main__':

    model_type = "unet"     # sizeloss || unet / fcn / cdws
    test_model = "all"     # fast || all
    phases = ["pos","neg"]  # 测试类型    ["pos","neg"] || ["neg"] || ["pos"]

    ckpts = ["ckpt.0071.pth","ckpt.0073.pth"]
    for ckpt in ckpts:
        checkpoint = Path(r"D:\组会内容\实验报告\MedT\records_SICAPV2\0对比算法\20220112-1731-PM_unet") / "checkpoints" / ckpt

        if model_type == "tgcn" or model_type == "wesup":
            # 执行像素级别推理
            infer_pixel_run(model_type=model_type,checkpoint=checkpoint,
                            test_model=test_model,phases=phases)
        else:
            # 正常模型推理
            infer_run(model_type=model_type,checkpoint=checkpoint,
                            test_model=test_model,phases=phases)
