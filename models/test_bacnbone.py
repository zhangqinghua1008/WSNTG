# -*- coding:utf-8 -*-
# @Time   : 2021/8/28 22:41
# @Author : 张清华
# @File   : test_bacnbone.py
# @Note   :  将back-bone换成u-net形式尝试

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.segmentation import slic
from torchvision import models

from models.base import BaseConfig, BaseTrainer
import segmentation_models_pytorch as smp
from model_utils.summary import summary

def load_model_weights(model, weights):
    model_dict = model.state_dict()
    weights = {k: v for k, v in weights.items() if k in model_dict}
    if weights == {}:
        print('No weight could be loaded..')
    model_dict.update(weights)
    model.load_state_dict(model_dict)

    return model



class Test_Backbone(nn.Module):
    def __init__(self, n_classes=2, D=32, **kwargs):
        """Initialize a TGCN model.
        Kwargs:
            n_classes: number of target classes (default to 2)
            D: output dimension of superpixel features 超像素特征的输出维度
        Returns:
            model: a new WESUP model
        """
        super().__init__()
        self.kwargs = kwargs
        # self.vgg16 = models.vgg16(pretrained=True).features

        resnet = models.__dict__['resnet18'](pretrained=False)
        state_dict = torch.load('G:/py_code/pycharm_Code/WESUP-TGCN/pretrained/checkpoint/tenpercent_resnet18.ckpt')['state_dict']
        for key in list(state_dict.keys()):
            state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)
        self.backbone = load_model_weights(resnet, state_dict)
        self.backbone.fc = torch.nn.Sequential()

        # sum of channels of all feature maps  所有特征图的通道之和
        self.fm_channels_sum = 0

        # 此时fm_channels_sum = 2112 ，VGG16总的特征维度为4224，因为每次做了sude_conv卷积除2，所以才是2112

        # unet 特征提取==================
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
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, D),
            nn.ReLU()
        )

        # final softmax classifier
        self.classifier = nn.Sequential(
            nn.Linear(D, 2),
            nn.Softmax(dim=1)
        )

        # store conv feature maps 存储conv特性映射
        self.feature_maps = None

        # spatial size of first feature map 第一个特征图的空间大小
        self.fm_size = None

        # label propagation input features 标签传播输入特性. (超像素特征)
        self.sp_features = None

        # superpixel predictions (tracked to compute loss) 超像素预测(跟踪以计算损失
        self.sp_pred = None

    # _hook_fn 函数是register_forward_hook()函数必须提供的参数. 当某一层被注册了hook之后,就会执行这个函数
    def _hook_fn(self, _, input_, output):
        '''
            用于处理feature的hook函数必须包含三个参数[module, fea_in, fea_out]，参数的名字可以自己起，但其意义是固定的。
            第一个参数表示torch里的一个子module，比如Linear,Conv2d等，第二个参数是该module的输入，其类型是tuple；
            第三个参数是该module的输出，其类型是tensor。注意输入和输出的类型是不一样的，切记。
            此函数：hook函数负责将获取的输入输出添加到feature列表中
        '''
        if self.feature_maps is None:
            # self.fm_size = (input_[0].size(2), input_[0].size(3))
            side_conv_name = 'side_conv0'
        else:
            side_conv_name = f'side_conv{self.feature_maps.size(0)}'

        output = getattr(self, side_conv_name)(output.clone())  # 获得我们定义的side_conv 的输出。（相当于我们给他降维了）
        output = F.interpolate(output, self.fm_size,            # 双线性插值
                               mode='bilinear', align_corners=True)

        # 将获取的输入输出添加到feature列表中
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
            x : 一个元组，包含大小为(1,C, H, W)输入张量, 和具有大小(N, H, W)的堆叠超像素映射
        Returns:
            pred: prediction with size (1, H, W)
        """
        # x, sp_maps = x
        sp_maps = torch.rand((101, x.size(2), x.size(3))).cuda()
        n_superpixels, height, width = sp_maps.size()  # n_superpixels:超像素个数, height,width 高,宽
        self.fm_size = (height, width)

        # extract conv feature maps and flatten 提取卷积特征图并flatten
        self.feature_maps = None
        _ = self.backbone(x)
        x = self.feature_maps[-33:-1,:,:]      # size:(fm_channels_sum,H,W)
        x = x.view(x.size(0), -1)  # flatten,得到size(fm_channels_sum,H*W)

        # calculate features for each superpixel 计算每个超像素的特征
        sp_maps = sp_maps.view(sp_maps.size(0), -1)  # flatten,得到size(N,H*W)
        x = torch.mm(sp_maps, x.t())                 # 矩阵相乘, 得到size(N,fm_channels_sum)  ps: x.t()转置

        # 利用全连通层降低超像素特征维数
        # x = self.fc_layers(x)       # reduce,得到size(N ,D),D=32

        self.sp_features = x       # 得到超像素特征:size(N ,D),D=32

        # classify each superpixel  每个superpixel分类
        self.sp_pred = self.classifier(x)  # Size: [N, 2]

        # 将sp_maps Flatten到一个通道  size:(Number,H*W) - > size:(Nubmer,H,W) ->argmax -> size:(H,W)
        sp_maps = sp_maps.view(n_superpixels, height, width).argmax(dim=0)  # size:(H,W) 此时map就是带有超像素id的原始图

        # initialize prediction mask  初始化预测mask
        pred = torch.zeros(height, width, self.sp_pred.size(1)) # size:(H,W,C（类别）)
        pred = pred.to(sp_maps.device)

        for sp_idx in range(sp_maps.max().item() + 1):
            pred[sp_maps == sp_idx] = self.sp_pred[sp_idx]  # self.sp_pred[sp_idx]: （C,）第idx个超像素的预测C类的结果

        out_z = pred.unsqueeze(0)[..., 1]  # size:(1,H,W), 获得c=1的概率 ;; 先增加一个维度，然后只取最后一位
        return out_z


if __name__ == '__main__':
    model = Test_Backbone().cuda()

    summary( model, (3,512,512) )

    fm_channels_sum = 0
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Conv2d) and layer.out_channels>2:
            print("fm_channels_sum: ",fm_channels_sum,end=" ")
            fm_channels_sum += layer.out_channels // 2
            print("layer.out_channels:",layer.out_channels,"  ->  fm_channels:",fm_channels_sum)
    print(fm_channels_sum)