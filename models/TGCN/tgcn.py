from functools import partial
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.segmentation import slic
from torchvision import models

from utils_network import empty_tensor
from utils_network import is_empty_tensor
from utils_network.data import SegmentationDataset
from utils_network.data import PointSupervisionDataset
from models.base import BaseConfig, BaseTrainer

from .gcn_layers import AdaptiveGraphConvolution
from .gcn_layers import AdaptiveGraphRecursiveConvolution

from .artificially_feature import art_features
from .handler import _preprocess_superpixels,_cross_entropy
from .handler import _adj,art_adj,smoothness_reg,load_model_weights
from .handler import tgcn_propagation


class TGCNConfig(BaseConfig):
    """Configuration for TGCN model. 为TGCN型配置 """

    # Rescale factor to subsample input images. 重新缩放因子的子样本输入图像。
    # rescale_factor = 0.35   #zqh DP2019
    rescale_factor = 0.5   #SICAPV2

    # multi-scale range for training  多尺度范围训练
    # multiscale_range = (0.25, 0.45)  # DP2019
    multiscale_range = (0.4, 0.6)  # DP2019

    # Number of target classes.
    n_classes = 2

    # Class weights for cross-entropy loss function.  交叉熵损失函数的类权值
    class_weights = (2, 1)  # default = (3,1)

    # Superpixel parameters.
    sp_area = 300   # 50 / 200
    sp_compactness = 40

    # Optimization parameters.
    momentum = 0.9
    weight_decay = 0.001

    # Whether to freeze backbone. 是否冻结骨干网
    freeze_backbone = False

    # Training configurations.
    batch_size = 1
    epochs = 200

    lr = 8e-4  # 6e-4
    # lr = 1e-3  # 6e-4



    is_gcn = True
    is_propagated = False
    # Weight for TGCN  when computing loss function 计算损失时 tgcn的正则化loss权重
    # 'reg_scalar', 1e-05, 'Weight of smoothness regularizer.')  # 平滑权值正则化器
    # gcn_smooth_reg_weight = 1e-05   # 平滑度调整
    gcn_smooth_reg_weight = 1e-02   # 平滑度调整


class TGCN(nn.Module):
    """Weakly supervised histopathology image segmentation with sparse point annotations."""

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
        self.backbone = models.vgg16(pretrained=True).features

        # sum of channels of all feature maps  所有特征图的通道之和
        self.fm_channels_sum = 0
        # side convolution layers after each conv feature map 每个conv特征图后的边卷积层
        for layer in self.backbone:
            if isinstance(layer, nn.Conv2d):  # isinstance() 函数来判断一个对象是否是一个已知的类型
                layer.register_forward_hook(self._hook_fn)  # 对需要的层注册hook，后续该层执行完forword后去执行_hook_fn函数
                setattr(self, f'side_conv{self.fm_channels_sum}',   # setattr(object, name, value) 函数指定对象的指定属性的值
                        nn.Conv2d(layer.out_channels, layer.out_channels // 2, 1) )   # 指定side_conv{int} = Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1))
                self.fm_channels_sum += layer.out_channels // 2
        # 此时fm_channels_sum = 2112，VGG16总的特征维度为4224，因为每次做了sude_conv卷积除2，所以才是2112

        # 对 low 和 height进行 降维
        self.fc_layers_low = nn.Sequential(
            nn.Linear(self.fm_channels_sum//2, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 12),
            # nn.BatchNorm1d(12),
            nn.ReLU()
        )

        self.fc_layers_hight = nn.Sequential(
            nn.Linear(self.fm_channels_sum//2, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 20),
            # nn.BatchNorm1d(20),
            nn.ReLU( )
        )

        # final softmax classifier
        self.classifier = nn.Sequential(
            nn.Linear(D, 2),
            nn.Softmax(dim=1)
        )

        # ============================= 图网络初始化
        #  层次化特征图
        self.sp_features_bn = nn.BatchNorm1d(32)
        self.gcn_nfeat = D
        self.gcn_nhid = D*2  # 隐藏层层数
        self.adj_len = 2     # adj长度(也就是有几个图)
        self.gcn_out_dim = 2
        self.gc1 = AdaptiveGraphConvolution(in_features_dim  = self.gcn_nfeat,
                                            out_features_dim = self.gcn_nhid,
                                            len = self.adj_len, bias=True)
        self.gc2 = AdaptiveGraphRecursiveConvolution(in_features_dim=self.gcn_nhid,
                                                     net_input_dim=self.gcn_nfeat,
                                                     out_features_dim=self.gcn_out_dim,
                                                     len = self.adj_len, bias=True)
        self.dropout = 0.1      # 图网络dropout 比例
        self.gcn_output = None  # 图网络output
        self.adj_list = None

        # ------ ---------- ----手工特征图
        self.artificial_features = None
        self.art_gcn_nfeat = 1
        self.art_gcn_nhid = 4   # 隐藏层层数
        self.art_adj_len = 1  # adj长度(也就是有几个图)
        self.art_gcn_out_dim = 2
        self.art_gc1 = AdaptiveGraphConvolution(in_features_dim=self.art_gcn_nfeat,
                                            out_features_dim=self.art_gcn_nhid,
                                            len=self.art_adj_len, bias=True)
        self.art_gc2 = AdaptiveGraphRecursiveConvolution(in_features_dim=self.art_gcn_nhid,
                                                     net_input_dim=self.art_gcn_nfeat,
                                                     out_features_dim=self.art_gcn_out_dim,
                                                     len=self.adj_len, bias=True)
        self.art_gcn_output = None  # 图网络output
        self.art_adj_list = None
        # ============================= =============================

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
            self.fm_size = (input_[0].size(2), input_[0].size(3))
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
            x : 一个元组，包含大小为(1,C, H, W)输入张量, 和 具有大小(N, H, W)的堆叠超像素映射

        Returns:
            pred: prediction with size (1, H, W)
        """
        x, sp_maps = x
        n_superpixels, height, width = sp_maps.size()  # n_superpixels:超像素个数, height,width 高,宽

        # 构建全局手工特征
        if self.kwargs.get('is_gcn'):
            art_feas = art_features(x)
            art_feas = torch.from_numpy(art_feas).cuda().to(torch.float32)
            art_feas = torch.sigmoid(art_feas)
            self.artificial_features = art_feas
            self.artificial_features.requires_grad = True

            # --------------- 全局手工特征 构建多图
            art_adj_list = art_adj(self.artificial_features)  # 获得手工特征adj列表
            self.art_adj_list = art_adj_list

            art_g1 = F.relu(self.art_gc1(self.artificial_features, art_adj_list))
            art_g1 = F.dropout(art_g1, 0, training=self.training)
            art_g2 = self.art_gc2(art_g1, self.artificial_features, art_adj_list)
            # art_gcn_out = F.softmax(art_g2, dim=1)
            art_gcn_out = F.sigmoid(art_g2)
            self.art_gcn_output = art_gcn_out

        # extract conv feature maps and flatten 提取卷积特征图并flatten
        self.feature_maps = None
        _ = self.backbone(x)
        x = self.feature_maps      # size:(fm_channels_sum,H,W)
        x = x.view(x.size(0), -1)  # flatten,得到size(fm_channels_sum,H*W)

        # calculate features for each superpixel 计算每个超像素的特征
        sp_maps = sp_maps.view(sp_maps.size(0), -1)  # flatten,得到size(N,H*W)
        x = torch.mm(sp_maps, x.t())                 # 矩阵相乘, 得到size(N,fm_channels_sum)  ps: x.t()转置

        # 利用全连通层降低超像素特征维数 reduce superpixel feature dimensions with fully connected layers
        # x = self.fc_layers(x)       # reduce,得到size(N ,D),D=32
        # self.sp_features = x  # 得到超像素特征

        # ======= 把层次化特征分开 分成low / height
        low_fetures = x[:,:self.fm_channels_sum//2]
        height_fetures = x[:, self.fm_channels_sum//2: ]
        # 利用全连通层降低超像素特征维数
        x_low = self.fc_layers_low(low_fetures)  # reduce,得到size(N ,12)
        x_height = self.fc_layers_hight(height_fetures)  # reduce,得到size(N ,20)
        x = torch.cat((x_low, x_height), dim=1)
        self.sp_features = x  # 得到超像素特征

        # TGCN =============
        if self.kwargs.get('is_gcn'):
            # --------------- 层次化特征 构建多图
            # self.sp_features = self.sp_features_bn(self.sp_features)  # 先对超像素特征进行批标准化
            adj_list = _adj(self.sp_features)    # 获得adj列表  sp_features:(N,32) -> adj.size: (N,N) -> adj_list:[adj1,adj2]
            self.adj_list = adj_list

            # gcnx = F.dropout(self.sp_features , self.dropout, training=self.training)
            g1 = F.relu( self.gc1(self.sp_features, adj_list) )
            g1 = F.dropout(g1, self.dropout, training=self.training)
            g2 = self.gc2(g1, self.sp_features, adj_list)
            gcn_out = F.softmax(g2, dim=1)
            # gcn_out = torch.nn.functional.sigmoid(g2)
            self.gcn_output = gcn_out

        # ========================

        # classify each superpixel  每个superpixel分类
        self.sp_pred = self.classifier(x)  # Size: [N, 2]

        # flatten sp_maps to one channel  将sp_maps Flatten到一个通道
        sp_maps = sp_maps.view(n_superpixels, height, width).argmax(dim=0)  # size:(H,W)

        # initialize prediction mask  初始化预测mask
        pred = torch.zeros(height, width, self.sp_pred.size(1)) # size:(H,W,C（类别）)
        pred = pred.to(sp_maps.device)

        for sp_idx in range(sp_maps.max().item() + 1):
            pred[sp_maps == sp_idx] = self.sp_pred[sp_idx]  # self.sp_pred[sp_idx]: （C,）第idx个超像素的预测C类的结果

        out_z = pred.unsqueeze(0)[..., 1]  # size:(1,H,W), 获得c=1的概率
        return out_z


class TGCNTrainer(BaseTrainer):
    """Trainer for TGCN."""

    def __init__(self, model, **kwargs):
        """Initialize a WESUPTrainer instance.
        Kwargs:
            rescale_factor: rescale factor to subsample input images 重新缩放因子的子样本输入图像
            multiscale_range: multi-scale range for training 多尺度范围的训练
            class_weights: class weights for cross-entropy loss function 交叉熵损失函数的权重
            sp_area: area of each superpixel 每个超像素的面积
            sp_compactness: compactness parameter of SLIC SLIC紧实度参数
            enable_propagation: whether to enable label propagation 是否启用标签传播
            momentum: SGD momentum
            weight_decay: weight decay for optimizer
            freeze_backbone: whether to freeze backbone
        Returns:
            trainer: a new WESUPTrainer instance
        """
        config = TGCNConfig()
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
                # return Digest2019PointDataset(root_dir, proportion=proportion,
                #                               multiscale_range=self.kwargs.get('multiscale_range'))
            return SegmentationDataset(root_dir, proportion=proportion,
                                       multiscale_range=self.kwargs.get('multiscale_range'))
        return SegmentationDataset(root_dir, rescale_factor=self.kwargs.get('rescale_factor'), train=False)

    def get_default_optimizer(self):
        # optimizer = torch.optim.Adam(
        #     filter(lambda p: p.requires_grad, self.model.parameters()),
        #     lr=self.kwargs.get('lr'),
        #     betas = (self.kwargs.get('momentum'),0.999),
        #     weight_decay=self.kwargs.get('weight_decay'), )
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.kwargs.get('lr'),  # 6e-4,
            momentum=self.kwargs.get('momentum'),
            weight_decay=self.kwargs.get('weight_decay'),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=7, factor=0.5, min_lr=1e-5, verbose=True)

        return optimizer, scheduler

    # 预处理,包含超像素分割等  img(1(batch_size),3,W,H)   point_mask/pixel_mask: (1,2(类别),W,H)
    def preprocess(self, *data):
        data = [datum.to(self.device) for datum in data]  # 放到CUDA中
        if len(data) == 3:   # 点标注信息时
            img, pixel_mask, point_mask = data
        elif len(data) == 2:
            img, pixel_mask = data
            point_mask = empty_tensor()
        elif len(data) == 1:
            img, = data
            point_mask = empty_tensor()
            pixel_mask = empty_tensor()
        else:
            raise ValueError('Invalid input data for WESUP')

        # SLIC 超像素分割, 此时进来的image已经缩小过了 不是原图大小了。
        segments = slic(
            img.squeeze().cpu().numpy().transpose(1, 2, 0),
            n_segments=int(img.size(-2) * img.size(-1) /    # n_segments: 分割输出图像中标签的(近似)数目。
                           self.kwargs.get('sp_area')),
            compactness=self.kwargs.get('sp_compactness'),
        )
        segments = torch.as_tensor(
            segments, dtype=torch.long, device=self.device)

        if point_mask is not None and not is_empty_tensor(point_mask):   # 如果点标签存在,点标注就是 mask, 即为点标注模式
            mask = point_mask.squeeze()
        elif pixel_mask is not None and not is_empty_tensor(pixel_mask): # 无点标签,则像素mask标注就是 mask, 即为全监督模式
            mask = pixel_mask.squeeze()
        else:
            mask = None

        # 超像素预处理
        sp_maps, sp_labels = _preprocess_superpixels(
            segments, mask, epsilon=self.kwargs.get('epsilon'))

        # img_size(1,3(3通道),W,H),sp_maps_size(SP_Number,W,H)
        # pixel_mask (1,C(分类数),W,H),  sp_labels(SP_Number,C)
        return (img, sp_maps), (pixel_mask, sp_labels)

    # 计算TGCN的损失
    def compute_loss(self, pred, target, metrics=None):
        '''
            target=(pixel_mask,sp_labels)
            pixel_mask (1,C(分类数),W,H),  sp_labels(SP_Number,C)  '''
        _, sp_labels = target  # _ 为像素级标签

        # sp_features = self.model.sp_features  # self.model = TGCN,之前标签传递要用到。现在用不到。
        sp_pred = self.model.sp_pred

        if sp_pred is None:
            raise RuntimeError('You must run a forward pass before computing loss. 在计算损失之前，必须进行前向传递')

        # 超像素的总数seed
        total_num = sp_pred.size(0)

        # number of labeled superpixels 标记的超像素数
        labeled_num = sp_labels.size(0)

        if labeled_num < total_num: # weakly-supervised mode  weakly-supervised模式(点注释模式)
            loss = self.xentropy(sp_pred[:labeled_num], sp_labels,
                                 class_weights=torch.Tensor(self.kwargs.get('class_weights')).to("cuda"))   # 只取已标注的超像素进行loss计算
            if metrics is not None and isinstance(metrics, dict):
                metrics['labeled_sp_ratio'] = labeled_num / total_num    # 已标注的超像素比例
        else:  # fully-supervised mode 全监督模式
            loss = self.xentropy(sp_pred, sp_labels)

        regloss = 0
        #=================== TGCN 正则化损失
        if self.kwargs.get('is_gcn'):
            # 超像素特征多图损失
            gcn_out = self.model.gcn_output
            # 通过有标签的样本进行计算tgcn_loss
            tgcn_loss_label = 0.3 * self.xentropy(gcn_out[:labeled_num], sp_labels)  # 只取已标注的超像素进行loss计算
            tgcn_loss_label = tgcn_loss_label.clamp(0.00001, 0.3)  # 限制范围
            loss += tgcn_loss_label
            metrics['gcn_loss'] = tgcn_loss_label.item()

            tgcn_propagate_loss = 0
            if self.kwargs.get('is_propagated'):
                # 然后通过TGCN进行标签扩散，对backbone进行微调
                tgcn_propagated_labels = tgcn_propagation(gcn_out,labeled_num)
                tgcn_propagate_loss = 1 * self.xentropy(sp_pred[labeled_num:], tgcn_propagated_labels)  # 只取已标注的超像素进行loss计算
                tgcn_propagate_loss = tgcn_propagate_loss.clamp(0.001, 0.4)  # 限制范围
                loss += tgcn_propagate_loss

                metrics['tgcn_propagate_number'] = tgcn_propagated_labels.sum().item()
                metrics['tgcn_propagate_loss'] = tgcn_propagate_loss.item()
            else:
                metrics['tgcn_propagate_number'] = 0
                metrics['tgcn_propagate_loss'] = 0

            # 手工特征单独构图损失
            art_gcn_out = self.model.art_gcn_output
            art_gcn_loss = smoothness_reg(art_gcn_out, self.model.art_adj_list)
            art_regloss = 0.1 * art_gcn_loss
            art_regloss = art_regloss.clamp(0.001,0.3)  # 限制范围
            loss += art_regloss
            metrics['art_regloss'] = art_regloss.item()

        # clear outdated superpixel prediction 清除过时的超像素预测
        self.model.sp_pred = None

        # clear outdated superpixel prediction 清除过时的图
        self.model.gcn_output = None
        self.model.adj_list = None

        self.model.art_gcn_output = None
        self.model.art_adj_list = None

        return loss

    def postprocess(self, pred, target=None):
        pred = pred.round().long()  # round(): 返回一个新张量，将pred张量每个元素舍入到最近的整数
        if target is not None:
            pixel_mask, _ = target
            return pred, pixel_mask.argmax(dim=1)
        return pred

    def post_epoch_hook(self, epoch):
        if self.scheduler is not None:
            labeled_loss = np.mean(self.tracker.history['loss'])

            # only adjust learning rate according to loss of labeled examples
            # 仅根据标记样例的丢失情况来调整学习率
            if 'propagate_loss' in self.tracker.history:
                labeled_loss -= np.mean(self.tracker.history['propagate_loss'])

            self.scheduler.step(labeled_loss)


class TGCNPixelInference(TGCN):
    """Weakly supervised histopathology image segmentation with sparse point annotations."""

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
        self.backbone = models.vgg16(pretrained=True).features

        # sum of channels of all feature maps
        self.fm_channels_sum = 0

        # side convolution layers after each conv feature map
        for layer in self.backbone:
            if isinstance(layer, nn.Conv2d):
                layer.register_forward_hook(self._hook_fn)
                setattr(self, f'side_conv{self.fm_channels_sum}',
                        nn.Conv2d(layer.out_channels, layer.out_channels // 2, 1))
                self.fm_channels_sum += layer.out_channels // 2
        # 此时fm_channels_sum = 2112，VGG16总的特征维度为4224，因为每次做了sude_conv卷积除2，所以才是2112

        # fully-connected layers for dimensionality reduction 全连接层降维
        self.fc_layers_low = nn.Sequential(
            nn.Linear(self.fm_channels_sum//2, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 12),
            # nn.BatchNorm1d(12),
            nn.ReLU()
        )

        self.fc_layers_hight = nn.Sequential(
            nn.Linear(self.fm_channels_sum//2, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 20),
            # nn.BatchNorm1d(20),
            nn.ReLU( )
        )


        # final softmax classifier
        self.classifier = nn.Sequential(
            nn.Linear(D, 2),
            nn.Softmax(dim=1)
        )

        # ============================= 图网络初始化
        #  层次化特征图
        self.sp_features_bn = nn.BatchNorm1d(32)
        self.gcn_nfeat = D
        self.gcn_nhid = D*2  # 隐藏层层数
        self.adj_len = 2     # adj长度(也就是有几个图)
        self.gcn_out_dim = 2

        self.gc1 = AdaptiveGraphConvolution(in_features_dim  = self.gcn_nfeat,
                                            out_features_dim = self.gcn_nhid,
                                            len = self.adj_len, bias=True)
        self.gc2 = AdaptiveGraphRecursiveConvolution(in_features_dim=self.gcn_nhid,
                                                     net_input_dim=self.gcn_nfeat,
                                                     out_features_dim=self.gcn_out_dim,
                                                     len = self.adj_len, bias=True)
        self.dropout = 0.1      # 图网络dropout 比例
        self.gcn_output = None  # 图网络output
        self.adj_list = None

        # ------ ---------- ----手工特征图
        self.artificial_features = None
        self.art_gcn_nfeat = 1
        self.art_gcn_nhid = 4  # 隐藏层层数
        self.art_adj_len = 1  # adj长度(也就是有几个图)
        self.art_gcn_out_dim = 2
        # self.art_gc1 = AdaptiveGraphConvolution(in_features_dim=self.art_gcn_nfeat,
        #                                     out_features_dim=self.art_gcn_nhid,
        #                                     len=self.art_adj_len, bias=True)
        # self.art_gc2 = AdaptiveGraphRecursiveConvolution(in_features_dim=self.art_gcn_nhid,
        #                                              net_input_dim=self.art_gcn_nfeat,
        #                                              out_features_dim=self.art_gcn_out_dim,
        #                                              len=self.adj_len, bias=True)
        self.art_gcn_output = None  # 图网络output
        self.art_adj_list = None
        # ============================= =============================

        # store conv feature maps 存储conv特性映射
        self.feature_maps = None

        # spatial size of first feature map 第一个特征图的空间大小
        self.fm_size = None

        # label propagation input features 标签传播输入特性. (超像素特征)
        self.sp_features = None

        # superpixel predictions (tracked to compute loss) 超像素预测(跟踪以计算损失
        self.sp_pred = None


    def _hook_fn(self, _, input_, output):
        if self.feature_maps is None:
            self.fm_size = (input_[0].size(2), input_[0].size(3))
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
            x: input image tensor of size (1, 3, H, W)
        Returns:
            pred: prediction with size (H, W, C)
        """
        height, width = x.size()[-2:]

        self.feature_maps = None
        _ = self.backbone(x)
        x = self.feature_maps
        x = x.view(x.size(0), -1)  # x -> torch.Size([2112, 78400]) ( 也就是【fm_channels_sum,W*H】)
        x = x.t()
        # x = self.fc_layers(x.t())

        # ======= 把层次化特征分开 分成low / height
        low_fetures = x[:, :self.fm_channels_sum // 2]
        height_fetures = x[:, self.fm_channels_sum // 2:]
        # 利用全连通层降低超像素特征维数 reduce superpixel feature dimensions with fully connected layers
        x_low = self.fc_layers_low(low_fetures)  # reduce,得到size(N ,D),D=32
        x_height = self.fc_layers_hight(height_fetures)  # reduce,得到size(N ,D),D=32
        x = torch.cat((x_low, x_height), dim=1)

        x = self.classifier(x)
        return x.view(height, width, -1)
