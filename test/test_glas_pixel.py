import os
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from PIL import Image
from skimage.io import imread, imsave
import torch.nn.functional as F
import torchvision.transforms.functional as TF

# ========= infer 推理
from utils_network.metrics import *
from TGCN.tgcn import TGCNPixelInference

'''
    TGCN, 像素级别推理，并根据推理出的图片，和原图进行比较得出评价指标.
'''

# =========

def pixel_infer(checkpoint,data_root, scales):
    print('Infer ++++++++++++ \n')
    data_dir = data_root / phase
    infer_output_dir = pred_root / phase

    # 加载模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TGCNPixelInference().to(device)
    model.load_state_dict(torch.load(
        checkpoint, map_location=device)['model_state_dict'])

    img_paths = list((data_dir / 'images').iterdir())

    with torch.no_grad():
        for img_path in tqdm(img_paths):
            img = TF.to_tensor(Image.open(img_path)).to(device).unsqueeze(0)
            preds = []

            for scale in scales:
                target_size = (int(img.size(2) * scale), int(img.size(3) * scale))
                pred = model(F.interpolate(img, scale_factor=scale,
                                           mode='bilinear', align_corners=True))
                pred = pred[..., 1].unsqueeze(0).unsqueeze(0)
                pred = F.interpolate(pred, size=img.size()[-2:],
                                     align_corners=True, mode='bilinear')
                preds.append(pred.squeeze())

            fused_pred = sum(preds) / len(preds)
            imsave(infer_output_dir / img_path.name.replace('.jpg', '.png'),
                   fused_pred.round().cpu().numpy().astype('uint8') * 255,
                   check_contrast=False)

# ========= evaluate 评估

def postprocess(pred):
    regions = label(pred)
    for region_idx in range(regions.max() + 1):
        region_mask = regions == region_idx
        if region_mask.sum() < 2000:
            pred[region_mask] = 0

    revert_regions = label(1 - pred)
    for region_idx in range(revert_regions.max() + 1):
        region_mask = revert_regions == region_idx
        if region_mask.sum() < 2000:
            pred[region_mask] = 1

    return pred

def compute_metrics(predictions, gts, pred_paths):
    iterable = list(zip(predictions, gts))

    executor_ac = Parallel(n_jobs=os.cpu_count())  # 并行化计算
    accuracies = executor_ac(delayed(accuracy)(pred, gt) for pred, gt in iterable)
    print('Accuracy:', np.mean(accuracies))

    executor_di = Parallel(n_jobs=os.cpu_count())
    dices = executor_di(delayed(dice)(pred, gt) for pred, gt in iterable)
    print('Dice:', np.mean(dices))

    executor_f1 = Parallel(n_jobs=os.cpu_count())
    detection_f1s = executor_f1(delayed(detection_f1)(pred, gt) for pred, gt in iterable)
    print('Detection F1:', np.mean(detection_f1s))

    executor_od = Parallel(n_jobs=os.cpu_count())
    object_dices = executor_od(delayed(object_dice)(pred, gt) for pred, gt in iterable)
    print('Object Dice:', np.mean(object_dices))

    # executor_of = Parallel(n_jobs=os.cpu_count())
    # object_hausdorffs = executor_of(delayed(object_hausdorff)(pred, gt) for pred, gt in iterable)
    # print('Object Hausdorff:', np.mean(object_hausdorffs))

    df = pd.DataFrame()
    df['detection_f1'] = detection_f1s
    df['object_dice'] = object_dices
    # df['object_hausdorff'] = object_hausdorffs
    df.index = [pred_path.name for pred_path in pred_paths]

    return df


if __name__ == '__main__':
    # infer 参数
    is_infer = True  # 是否infer,只有第一次test时infer
    model_type = 'tgcn'
    checkpoint = Path(r"G:\py_code\pycharm_Code\WESUP-TGCN\records\20210902-1106-PM\checkpoints\ckpt.0300.pth")
    glas_root = Path(r'..\data_glas')  # 数据地址

    infer_scales = (0.5,)   # 默认(0.5,)  ps:0.8/0.7的时候越差,

    # 根据推理出的图片进行比较
    output_dir = checkpoint.parent / 'output'
    if not output_dir.exists(): output_dir.mkdir()
    output = output_dir / f'results-pixel-{str(infer_scales)}'
    if not output.exists():
        output.mkdir()

    pred_root =  output / 'infer'             # 保存infer的图
    new_pred_root = output / 'evaluate-new'   #保存后处理后的预测图
    if not pred_root.exists():
        pred_root.mkdir()
        (pred_root / 'testA').mkdir()
        (pred_root / 'testB').mkdir()
    if not new_pred_root.exists():
        new_pred_root.mkdir()
        (new_pred_root / 'testA').mkdir()
        (new_pred_root / 'testB').mkdir()

    phases = ['testA','testB']

    for phase in phases:
        print('\n', phase, ' =========== ========== =========')

        # infer ---------
        if is_infer:
            pixel_infer(checkpoint,data_root = glas_root ,scales = infer_scales)

        # 评估 -----
        print('Evaluate ++++++++++++\n')
        print('Reading predictions and gts ...')
        pred_paths = sorted((pred_root / phase ).glob('*.bmp'))
        # predictions = executor(delayed(postprocess)(imread(str(pred_path)) / 255) for pred_path in pred_paths)
        predictions = []
        for pred_path in pred_paths:
            predictions.append(postprocess(imread(str(pred_path)) / 255) )

        executor = Parallel(n_jobs=os.cpu_count())  # 并行化计算
        gt_paths = sorted((glas_root / phase / 'masks').glob('*.bmp'))
        gts = executor(delayed(imread)(gt_path) for gt_path in gt_paths)

        print(f'保存后处理后的预测 ... {phase} 的测试结果：')
        for pred, pred_path in zip(predictions, pred_paths):
            imsave(new_pred_root / phase / pred_path.name, (pred * 255).astype('uint8'))

        metrics = compute_metrics(predictions, gts, pred_paths)
        metrics.to_csv(pred_root / (phase+'.csv'))


