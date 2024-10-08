# -*- coding:utf-8 -*-
# @Time   : 2021/10/9 17:06
# @Author : 张清华
# @File   : testnet.py
# @Note   :  将back-bone换成预训练的resnet18形式尝试

import os
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.segmentation import slic
from torchvision import models

from utils_network import empty_tensor
from utils_network import is_empty_tensor
from utils_network.data import PointSupervisionDataset
from utils_network.data import SegmentationDataset
from base import BaseConfig, BaseTrainer

import segmentation_models_pytorch as smp
from model_utils.summary import summary
from net_structure import vgg_conv,vgg_one_conv,vgg_classifier

def _preprocess_superpixels(segments, mask=None, epsilon=1e-7):
    """Segment superpixels of a given image and return segment maps and their labels.
        给定图像的分段超像素，并return 分段映射(segment maps)及其标签。
    Args:
        segments: slic segments tensor with shape (H, W)   (H,W)大小的slic分割张量,每个像素一个超像素的编号（slic segments tensor）
        mask (optional): annotation mask tensor with shape (C, H, W). Each pixel is a one-hot
            encoded label vector. If this vector is all zeros, then its class is unknown.
                形状为（C，H，W）的注释掩码张量。每个像素是一个热编码标签向量。如果这个向量都是零，那么它的类是未知的。
    Returns:
        sp_maps: superpixel maps with shape (N, H, W)
        sp_labels: superpixel labels with shape (N_l, C), where N_l is the number of labeled samples.

        sp_maps: 具有形状(N, H, W)的超像素映射
        sp_labels: 具有形状(N_l, C)的超像素标签，其中N_l为已标记样本的数量。
    """

    # ordering of superpixels  获取所有的超像素标签（就是1...n，共n个超像素）
    sp_idx_list = segments.unique()

    if mask is not None and not is_empty_tensor(mask):
        def compute_superpixel_label(sp_idx):
            sp_mask = (mask * (segments == sp_idx).long()).float()
            return sp_mask.sum(dim=(1, 2)) / (sp_mask.sum() + epsilon)   # epsilon 避免出现0的情况

        # compute labels for each superpixel  计算每个超像素的标签-》 [0,1]之间的一个值
        sp_labels = torch.cat([
            compute_superpixel_label(sp_idx).unsqueeze(0)
            for sp_idx in range(segments.max() + 1)
        ])  # sp_labels shape: [sp_N, 2] ,sp_N 是超像素个数, 代表超像素为每一类的概率

        # move labeled superpixels to the front of `sp_idx_list` 将带标签的超像素移到' sp_idx_list '的前面
        labeled_sps = (sp_labels.sum(dim=-1) > 0).nonzero(as_tuple=False).flatten()
        unlabeled_sps = (sp_labels.sum(dim=-1) == 0).nonzero(as_tuple=False).flatten()
        sp_idx_list = torch.cat([labeled_sps, unlabeled_sps])

        # quantize superpixel labels (e.g., from (0.7, 0.3) to (1.0, 0.0))
        # 量化超像素标签(例如，从(0.7,0.3)到(1.0,0.0))
        sp_labels = sp_labels[labeled_sps]
        sp_labels = (sp_labels == sp_labels.max(
            dim=-1, keepdim=True)[0]).float()
    else:  # no supervision provided  没有提供监督
        sp_labels = empty_tensor().to(segments.device)

    # stacking normalized superpixel segment maps 叠加归一化超像素段映射
    sp_maps = segments == sp_idx_list[:, None, None]
    sp_maps = sp_maps.squeeze().float()

    # make sure each superpixel map sums to one  确保每个超像素映射和为1
    sp_maps = sp_maps / sp_maps.sum(dim=(1, 2), keepdim=True)

    return sp_maps, sp_labels


def _cross_entropy(y_hat, y_true, class_weights=None, epsilon=1e-7):
    """Semi-supervised cross entropy loss function. 半监督交叉熵损失函数。

    Args:
        y_hat: prediction tensor with size (N, C), where C is the number of classes size:(N, C)的预测张量，其中C是类的数量
        y_true: label tensor with size (N, C). A sample won't be counted into loss size:(N, C)的标注张量。样本不会计入损失
            if its label is all zeros. 如果它的标签全是0。
        class_weights: class weights tensor with size (C,)  具有大小(C，)的类权张量
        epsilon: numerical stability term   数值稳定的术语

    Returns:
        cross_entropy: cross entropy loss computed only on samples with labels  交叉熵损失仅在有标记的样本上计算
    """

    device = y_hat.device

    # clamp all elements to prevent numerical overflow/underflow  夹紧所有元件以防止数值溢出/下溢
    y_hat = torch.clamp(y_hat, min=epsilon, max=(1 - epsilon))

    # number of samples with labels  带标签的样品数量
    labeled_samples = torch.sum(y_true.sum(dim=1) > 0).float()

    if labeled_samples.item() == 0:
        return torch.tensor(0.).to(device)

    ce = -y_true * torch.log(y_hat)

    if class_weights is not None:
        ce = ce * class_weights.unsqueeze(0).float()

    return torch.sum(ce) / labeled_samples


def _label_propagate(features, y_l, threshold=0.95):
    """Perform random walk based label propagation with similarity graph.
       利用相似图进行基于随机游走的标签传播。

    Arguments:
        features: features of size (N, D), where N is the number of superpixels
            and D is the dimension of input features
        y_l: label tensor of size (N, C), where C is the number of classes
        threshold: similarity threshold for label propagation

        features: 特征的大小(N, D)，其中N是超像素的数量,D是输入特征的维数
        y_l: 大小为(N_label, C)的标记张量，其中C为类的数目
        threshold: 标签传播的相似阈值
    Returns:
        pseudo_labels: propagated label tensor of size (N, C)
        pseudo_labels:  大小(N, C)的传播标号张量
    """

    # disable gradient computation  禁用梯度计算
    features = features.detach()   # torch.Size([256, 32])
    y_l = y_l.detach()             # torch.Size([23, 2])

    # number of labeled and unlabeled samples  贴有标签和未贴有标签的样品数量
    n_l = y_l.size(0)
    n_u = features.size(0) - n_l

    # feature affinity matrix  特征关联矩阵
    # features - features.unsqueeze(1): size(N,N,D) torch.Size([256, 256, 32])
    # W.size: (N,N), 即torch.Size([256, 256])
    W = torch.exp(-torch.einsum('ijk,ijk->ij',    # 爱因斯坦求和 （einsum）
                                features - features.unsqueeze(1),
                                features - features.unsqueeze(1)))

    # sub-matrix of W containing similarities between labeled and unlabeled samples
    # W的子矩阵包含标记和未标记样品之间的相似性  ,W_ul: torch.Size([233, 23])  . 代表233个未标记样本和 23个已经标记样本之间的相似度
    W_ul = W[n_l:, :n_l]

    # src_indexes is the respective labeled sample index
    # max_similarity是每个未标记样本的最大相似度,Src_indexes是各自标记的样本索引(就是每个样本对于最大的已标记样本的下标)
    max_similarities, src_indexes = W_ul.max(dim=1)

    # initialize y_u with zeros  用零初始化y_u,torch.Size([233, 2])
    # y_u : 未标记像本的y
    y_u = torch.zeros(n_u, y_l.size(1)).to(y_l.device)

    # only propagate labels if maximum similarity is above the threshold
    # 仅在最大相似度超过阈值时传播标签 propagated_samples: torch.Size([233])
    propagated_samples = max_similarities > threshold
    y_u[propagated_samples] = y_l[src_indexes[propagated_samples]]

    return y_u

def load_model_weights(model, weights):
    model_dict = model.state_dict()
    weights = {k: v for k, v in weights.items() if k in model_dict}
    if weights == {}:
        print('No weight could be loaded..')
    model_dict.update(weights)
    model.load_state_dict(model_dict)

    return model


class TEST_RESNET18Config(BaseConfig):
    """Configuration for WESUP model. 为WESUP模型配置 """
    net_info = "这是: TEST_RESNET18"

    # Rescale factor to subsample input images. 重新缩放因子的子样本输入图像。
    rescale_factor = 0.5

    # multi-scale range for training  多尺度范围训练
    multiscale_range = (1, 1)

    # Number of target classes.
    n_classes = 2

    # Class weights for cross-entropy loss function.  交叉熵损失函数的类权值
    class_weights = (3, 1)

    # Superpixel parameters.
    sp_area = 200   # 50 / 200
    sp_compactness = 40

    # Weight for label-propagated samples when computing loss function 计算损时 标签传播样本的权重
    propagate_threshold = 0.8

    # Weight for label-propagated samples when computing loss function 计算损失时 标签传播样本的权重
    propagate_weight = 0.5

    # Optimization parameters.
    momentum = 0.9
    weight_decay = 0.001

    # Whether to freeze backbone. 是否冻结骨干网
    freeze_backbone = False

    # Training configurations.
    batch_size = 1
    epochs = 200

    lr = 1e-3  # 5e-4


class TEST_RESNET18(nn.Module):
    """Weakly supervised histopathology image segmentation with sparse point annotations."""

    def __init__(self, n_classes=2, D=32, **kwargs):
        """Initialize a WESUP model.
        Kwargs:
            n_classes: number of target classes (default to 2)
            D: output dimension of superpixel features 超像素特征的输出维度
        Returns:
            model: a new WESUP model
        """
        super().__init__()
        self.kwargs = kwargs

        resnet = models.__dict__['resnet18'](pretrained=False)
        state_dict = torch.load('G:/py_code/pycharm_Code/WESUP-TGCN/pretrained/checkpoint/tenpercent_resnet18.ckpt')[
            'state_dict']
        for key in list(state_dict.keys()):
            state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)
        self.backbone = load_model_weights(resnet, state_dict)
        self.backbone.fc = torch.nn.Sequential()

        # sum of channels of all feature maps  所有特征图的通道之和
        self.fm_channels_sum = 0

        for name, layer in self.backbone.named_modules():
            if isinstance(layer, torch.nn.Conv2d):
                layer.register_forward_hook(self._hook_fn)  # 对需要的层注册hook，后续该层执行完forword后去执行_hook_fn函数
                setattr(self, f'side_conv{self.fm_channels_sum}',   # setattr(object, name, value) 函数指定对象的指定属性的值
                                nn.Conv2d(layer.out_channels, layer.out_channels // 2, 1) )      # 指定side_conv{int} = Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1))
                self.fm_channels_sum += layer.out_channels // 2

        # fully-connected layers for dimensionality reduction  全连接层降维
        self.fc_layers = nn.Sequential(
            nn.Linear(self.fm_channels_sum, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, D),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(D, self.kwargs.get('n_classes', 2)),
            nn.Softmax(dim=1)
        )

        self.vgg_conv = vgg_conv()
        self.one_conv = vgg_one_conv()
        self.vgg_classifier = vgg_classifier()

        # store conv feature maps 存储conv特性映射
        self.feature_maps = None

        # spatial size of first feature map 第一个特征图的空间大小
        self.fm_size = None

        # label propagation input features 标签传播输入特性
        self.sp_features = None

        # superpixel predictions (tracked to compute loss) 超像素预测(跟踪以计算损失
        self.sp_pred = None

    # hook（）函数是register_forward_hook()函数必须提供的参数
    # hook函数需要三个参数，这三个参数是系统传给hook函数的，自己不能修改这三个参数：
    # hook函数负责将获取的输入输出添加到feature列表中
    def _hook_fn(self, _, input_, output):
        if self.feature_maps is None:
            # self.fm_size = (input_[0].size(2), input_[0].size(3))
            side_conv_name = 'side_conv0'
        else:
            side_conv_name = f'side_conv{self.feature_maps.size(0)}'

        output = getattr(self, side_conv_name)(output.clone())
        output = F.interpolate(output, self.fm_size,
                               mode='bilinear', align_corners=True)

        if self.feature_maps is None:
            self.feature_maps = output.squeeze()
        else:
            self.feature_maps = torch.cat(
                (self.feature_maps, output.squeeze()))

    def forward(self, x):
        """Running a forward pass.
        Args:
            x: a tuple containing input tensor of size (1, C, H, W) and
                stacked superpixel maps with size (N, H, W)
            x : 一个元组，包含大小为(1,C, H, W)输入张量, 具有大小(N, H, W)的堆叠超像素映射

        Returns:
            pred: prediction with size (1, H, W)
        """
        # 测试时：
        # sp_maps = torch.rand((101, x.size(2), x.size(3))).cuda()
        # 训练时
        x, sp_maps = x

        n_superpixels, height, width = sp_maps.size()
        self.fm_size = (height, width)

        # extract conv feature maps and flatten 提取卷积特征图并flatten
        self.feature_maps = None
        _ = self.backbone(x)
        x = self.feature_maps  # size:(fm_channels_sum,H,W)
        x = x.view(x.size(0), -1)

        # calculate features for each superpixel 计算每个超像素的特征
        sp_maps = sp_maps.view(sp_maps.size(0), -1)
        x = torch.mm(sp_maps, x.t())

        # reduce superpixel feature dimensions with fully connected layers
        # 利用全连通层降低超像素特征维数
        x = self.fc_layers(x)
        self.sp_features = x

        # 降维后加VGG层
        # [num,128] -> [num,1,128,128]
        # x = x.view((n_superpixels, 32, 32))
        # x = x.unsqueeze(dim=1)
        #
        # x = self.vgg_conv(x)
        # x = self.one_conv(x)  # 用1*1卷积
        # x = x.view(x.size(0), -1)
        # self.sp_pred = self.vgg_classifier(x)
        # ++++++++++++++++++++++++++++++++++++++++++++++

        # classify each superpixel  每个superpixel分类
        self.sp_pred = self.classifier(x)

        # flatten sp_maps to one channel  将sp_maps Flatten到一个通道
        sp_maps = sp_maps.view(n_superpixels, height, width).argmax(dim=0)

        # initialize prediction mask  初始化预测mask
        pred = torch.zeros(height, width, self.sp_pred.size(1))
        pred = pred.to(sp_maps.device)

        for sp_idx in range(sp_maps.max().item() + 1):
            pred[sp_maps == sp_idx] = self.sp_pred[sp_idx]

        return pred.unsqueeze(0)[..., 1]


class TEST_RESNET18Trainer(BaseTrainer):
    """Trainer for TESTNET."""

    def __init__(self, model, **kwargs):
        """Initialize a WESUPTrainer instance.

        Kwargs:
            rescale_factor: rescale factor to subsample input images 重新缩放因子的子样本输入图像
            multiscale_range: multi-scale range for training 多尺度范围的训练
            class_weights: class weights for cross-entropy loss function 交叉熵损失函数的权重
            sp_area: area of each superpixel 每个超像素的面积
            sp_compactness: compactness parameter of SLIC SLIC紧实度参数
            enable_propagation: whether to enable label propagation 是否启用标签传播
            propagate_threshold: threshold for label propagation 标号传播阈值
            propagate_weight: weight for label-propagated samples in loss function 损失函数中标签传播样本的权重
            momentum: SGD momentum
            weight_decay: weight decay for optimizer
            freeze_backbone: whether to freeze backbone

        Returns:
            trainer: a new WESUPTrainer instance
        """

        config = TEST_RESNET18Config()
        if config.freeze_backbone:
            # 冻结主干网络，默认不冻结
            for param in model.backbone.parameters():
                param.requires_grad = False
        kwargs = {**config.to_dict(), **kwargs}  # 把设置和命令行参数全都放到 kwargs下
        super().__init__(model, **kwargs)        # 执行 BaseTrainer的 init方法

        # cross-entropy loss function 叉损失函数
        self.xentropy = partial(_cross_entropy)  # 偏函数，这里没附加参数，所以类似换个名称而已

    def get_default_dataset(self, root_dir, train=True, proportion=1.0):
        if train:
            if os.path.exists(os.path.join(root_dir, 'points')):
                return PointSupervisionDataset(root_dir, proportion=proportion,
                                              multiscale_range=self.kwargs.get('multiscale_range'))
            return SegmentationDataset(root_dir, proportion=proportion,
                                       multiscale_range=self.kwargs.get('multiscale_range'))
        return SegmentationDataset(root_dir, rescale_factor=self.kwargs.get('rescale_factor'), train=False)

    def get_default_optimizer(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.kwargs.get('lr'),
            betas=(self.kwargs.get('momentum'), 0.999),
            weight_decay=self.kwargs.get('weight_decay'), )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=10, factor=0.5, min_lr=1e-5, verbose=True)

        return optimizer, None

    # 预处理
    def preprocess(self, *data):
        data = [datum.to(self.device) for datum in data]
        if len(data) == 3:
            img, pixel_mask, point_mask = data
        elif len(data) == 2:
            img, pixel_mask = data
            point_mask = empty_tensor()
        elif len(data) == 1:
            img, = data
            point_mask = empty_tensor()
            pixel_mask = empty_tensor()
        else:
            raise ValueError('Invalid input data for TESTNET')

        # SLIC 超像素分割
        segments = slic(
            img.squeeze().cpu().numpy().transpose(1, 2, 0),
            n_segments=int(img.size(-2) * img.size(-1) /    # n_segments: 分割输出图像中标签的(近似)数目。
                           self.kwargs.get('sp_area')),
            compactness=self.kwargs.get('sp_compactness'),
        )
        segments = torch.as_tensor(
            segments, dtype=torch.long, device=self.device)

        if point_mask is not None and not is_empty_tensor(point_mask):
            mask = point_mask.squeeze()
        elif pixel_mask is not None and not is_empty_tensor(pixel_mask):
            mask = pixel_mask.squeeze()
        else:
            mask = None

        sp_maps, sp_labels = _preprocess_superpixels(
            segments, mask, epsilon=self.kwargs.get('epsilon'))

        # img_size(1,3(3通道),W,H),sp_maps_size(SP_Number,W,H)
        # pixel_mask (1,C(分类数),W,H),  sp_labels(SP_Number,C)
        return (img, sp_maps), (pixel_mask, sp_labels)

    # 计算TESTNET的损失
    def compute_loss(self, pred, target, metrics=None):
        '''
            target=(pixel_mask,sp_labels)
            pixel_mask (1,C(分类数),W,H),  sp_labels(SP_Number,C)  '''
        _, sp_labels = target

        # sp_features = self.model.sp_features  # self.model = TGCN,之前标签传递要用到。现在用不到。
        sp_pred = self.model.sp_pred

        if sp_pred is None:
            raise RuntimeError('You must run a forward pass before computing loss. 在计算损失之前，必须进行前向传递')

        # 超像素的总数
        total_num = sp_pred.size(0)

        # number of labeled superpixels 标记的超像素数
        labeled_num = sp_labels.size(0)

        if labeled_num < total_num: # weakly-supervised mode  weakly-supervised模式(点注释模式)
            loss = self.xentropy(sp_pred[:labeled_num], sp_labels)   # 只取已标注的超像素进行loss计算
            if metrics is not None and isinstance(metrics, dict):
                metrics['labeled_sp_ratio'] = labeled_num / total_num    # 已标注的超像素比例
        else:  # fully-supervised mode 全监督模式
            loss = self.xentropy(sp_pred, sp_labels)

        # TGCN 正则化损失
        # if self.kwargs.get('is_gcn'):
        #     gcn_out = self.model.gcn_output
        #     tgcn_loss = smoothness_reg(gcn_out, self.model.adj_list )
        #     regloss = self.kwargs.get('gcn_smooth_reg_weight') * tgcn_loss
        #     self.logger.info( '正则化损失：',regloss.item())
        #     loss += regloss

        # clear outdated superpixel prediction 清除过时的超像素预测
        self.model.sp_pred = None

        # clear outdated superpixel prediction 清除过时的超像素预测
        self.model.gcn_output = None
        self.model.adj_list = None

        return loss

    def postprocess(self, pred, target=None):
        pred = pred.round().long()
        if target is not None:
            return pred, target[0].argmax(dim=1)
        return pred

    def post_epoch_hook(self, epoch):
        if self.scheduler is not None:
            labeled_loss = np.mean(self.tracker.history['loss'])

            # only adjust learning rate according to loss of labeled examples
            # 仅根据标记样例的丢失情况来调整学习率
            if 'propagate_loss' in self.tracker.history:
                labeled_loss -= np.mean(self.tracker.history['propagate_loss'])

            self.scheduler.step(labeled_loss)


if __name__ == '__main__':
    model = TEST_RESNET18().cuda()

    summary( model, (3,512,512) )