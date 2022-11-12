from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed
from skimage.io import imread, imsave
import os
import os.path as osp
import torchvision.transforms.functional as TF
from tqdm import tqdm
from PIL import Image
from skimage.transform import resize

# ========= infer 推理
import infer
from data import FullSegmentationDataset
from models import initialize_trainer
from utils_network.metrics import *

'''
    推理，并根据推理出的图片，和原图进行比较得出评价指标.
'''

# ========= evaluate 评估

def infer(trainer, dataset, output_dir=None, num_workers=4, device='cpu'):
    """Making inference on a directory of images with given model checkpoint.
        对给定模型检查点的图像目录进行推理。    """

    trainer.model.eval()

    predictions = predict(trainer, dataset, num_workers=num_workers, device=device)

    if output_dir is not None:
        save_predictions(predictions, dataset, output_dir)

    return predictions


def predict(trainer, dataset, num_workers=4, device='cpu'):
    # 加载数据集
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=num_workers)

    print(f'\nPredicting {len(dataset)} images ...')
    predictions = []
    for data in tqdm(dataloader, total=len(dataset)):
        img = data[0].to(device)

        # infer 单个 img
        img = trainer.preprocess(img)  # w
        img = img[0]
        prediction = trainer.postprocess(trainer.model(img))

        # UneXt 专属
        prediction = torch.sigmoid(prediction).squeeze(0).detach().cpu().numpy()
        prediction[prediction >= 0.5] = 1
        prediction[prediction < 0.5] = 0
        predictions.append(prediction[0])

    return predictions


def save_predictions(predictions, dataset, output_dir='predictionds'):
    """Save predictions to disk.  将预测保存到磁盘。
    Args:
        predictions: model predictions of size (N, H, W)
        dataset: dataset for prediction, used for naming the prediction output
        output_dir: path to output directory
    """
    print(f'\nSaving prediction to {output_dir} ...')

    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir()

    for pred, img_path, mask_path in tqdm(zip(predictions, dataset.img_paths, dataset.mask_paths),
                                          total=len(predictions)):
        pred = pred.astype('uint8')
        Image.fromarray(pred * 255).save(output_dir / f'{img_path.stem}.png')  # 保存预测
        # if mask_path is not None: # 保存本来mask
        #     mask = imread(str(mask_path))
        # Image.fromarray(mask * 255).save(output_dir / f'{mask_path.stem}.png')


def compute_metrics(predictions, gts, pred_paths):
    for i in range(len(gts)):
        gts[i] = resize(gts[i], (512,512), order=1, anti_aliasing=False)

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

    df = pd.DataFrame()
    df['detection_f1'] = detection_f1s
    df['object_dice'] = object_dices
    df.index = [pred_path.name for pred_path in pred_paths]

    return df


if __name__ == '__main__':
    # infer 参数
    is_infer = True  # 是否infer,只有第一次test时infer
    model_type = 'unext'
    checkpoint = Path(r"E:\records\0对比算法\20220906-1641-PM_unext\checkpoints\ckpt.0200.pth")
    glas_root = Path(r'D:\组会内容\data\GlaS\data_glas')  # 数据地址

    # 文件夹
    output = checkpoint.parent.parent / f'{checkpoint.name}.output'
    if not output.exists():
        output.mkdir()
    output = output / f'results_Glas'
    if not output.exists():
        output.mkdir()

    pred_root = output / 'infer'  # 保存infer的图
    if not pred_root.exists():
        pred_root.mkdir()
        (pred_root / 'testA').mkdir()
        (pred_root / 'testB').mkdir()

    phases = ['testB']
    for phase in phases:
        print('\n', phase, ' =========== ========== =========')

        # infer ---------
        if is_infer:
            data_dir = glas_root / phase
            infer_output_dir = pred_root / phase

            # 加载模型
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            trainer = initialize_trainer(model_type, device=device)
            if checkpoint is not None:
                trainer.load_checkpoint(checkpoint)
            # 加载数据集
            dataset = FullSegmentationDataset(data_dir, train=False, target_size=(512, 512))
            infer(trainer, dataset, infer_output_dir, num_workers=4, device=device)

        # 评估 -----
        print('Evaluate ++++++++++++\n')
        pred_paths = sorted((pred_root / phase).glob('*.png'))
        predictions = []
        for pred_path in pred_paths:
            predictions.append(imread(str(pred_path)) / 255)
            # predictions.append(postprocess(imread(str(pred_path)) / 255) )

        executor = Parallel(n_jobs=os.cpu_count())  # 并行化计算
        gt_paths = sorted((glas_root / phase / 'masks').glob('*.bmp'))
        # 加载数据集
        # dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=4)
        gts = executor(delayed(imread)(gt_path) for gt_path in gt_paths)

        metrics = compute_metrics(predictions, gts, pred_paths)
        metrics.to_csv(pred_root / (phase + '.csv'))
