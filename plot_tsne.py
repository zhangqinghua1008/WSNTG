import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
from pathlib import Path
from models import WESUP,WESUPTrainer
from models.TGCN.tgcn import TGCN, TGCNTrainer
from models import FSTGN, FSTGNTrainer
from utils_network.data import SegmentationDataset
from sklearn import preprocessing
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
'''
    TSNE散点图在 plot_tsne.py
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rescale-factor', type=float, default=0.5, help='Rescale factor for input samples')
    args = parser.parse_args()

    model_type = "tgcn"  #  FSTGN | tgcn
    checkpoint = Path(r"D:\组会内容\实验报告\MedT\records\20220103-1939-PM_tgcn\checkpoints")  / "ckpt.0045.pth"

    indexs = [101]  # [a, b) c是步长
    indexs = np.arange(1, 500, 5)  # [a, b) c是步长

    # DP2019
    dataset_path = r"D:\组会内容\data\Digestpath2019\MedT\train\all_foreground\patch_800\train"
    rescale_factor = 0.35

    # SICAPV2
    # dataset_path = r"D:\组会内容\data\SICAPV2\res\patch3\train"
    # rescale_factor = 0.45  # SICAPV2

    # 16
    # dataset_path = r"G:\dataG\CAMELYON16\training\patches_level2_Tumor_3000/train"
    # rescale_factor = 0.45  # 16
    # indexs = np.arange(3500, 4500, 10)  # [a, b) c是步长


    save_path = checkpoint.parent.parent / "train_TSNE"
    save_path.mkdir(exist_ok=True)

    # before training
    print('preparing before training ...')

    for index in indexs:
        if model_type == 'FSTGN':
            model = FSTGN()
            train = FSTGNTrainer(model, sp_area=25)
        elif  model_type == 'tgcn':
            model = TGCN()
            train = TGCNTrainer(model, sp_area=25)

        ckpt = torch.load(checkpoint)
        dataset = SegmentationDataset(dataset_path, rescale_factor=rescale_factor)

        # tsne = TSNE(),,early_exaggeration=100
        tsne = TSNE(n_components=2 ,random_state=42)

        img, mask = dataset[index]
        (img, sp_maps), (pixel_mask, sp_labels) = train.preprocess(img, mask)

        # sp_labels = sp_labels.argmax(dim=1).numpy()
        sp_labels = sp_labels.argmax(dim=1)

        # 前景比例不合适不画图
        if sp_labels.sum().item() < sp_labels.shape[0] * 0.3 or sp_labels.sum().item() > sp_labels.shape[0] * 0.7:
            continue

        pred = model((img.unsqueeze(0), sp_maps))
        before_features = model.sp_features.detach().cpu().numpy()
        before_x2d = tsne.fit_transform(before_features)
        # 归一化处理
        # scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        # before_x2d = scaler.fit_transform(before_x2d)

        # after training
        print('preparing after training ...')
        model.load_state_dict(ckpt['model_state_dict'])
        pred = model((img.unsqueeze(0), sp_maps))
        after_features = model.sp_features.detach().cpu().numpy()
        after_x2d = tsne.fit_transform(after_features)
        # 归一化处理
        # scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        # after_x2d = scaler.fit_transform(after_x2d)


        # plotting
        # plt.figure(figsize=(12, 12))
        fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(13, 6))
        sp_labels = sp_labels.cpu().numpy()
        # plt.scatter(x,y,c= '颜色可选',marker= '点的样式', cmap= '颜色变化',alpha=“透明度”, linewidths=“线宽”,s= '点的大小')
        ax1.scatter(before_x2d[:, 0], before_x2d[:, 1], c=sp_labels, alpha=0.5)
        ax2.scatter(after_x2d[:, 0], after_x2d[:, 1], c=sp_labels, alpha=0.5)
        plt.savefig(save_path / (str(index) + ".png"))
