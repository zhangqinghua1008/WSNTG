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

    # mask存在，说明为点监督 或 全监督
    if mask is not None and not is_empty_tensor(mask):
        def compute_superpixel_label(sp_idx):
            sp_mask = (mask * (segments == sp_idx).long()).float()
            return sp_mask.sum(dim=(1, 2)) / (sp_mask.sum() + epsilon)   # epsilon 避免出现0的情况

        # compute labels for each superpixel  计算每个超像素的标签 -> [0,1]之间的一个值
        sp_labels = torch.cat([
            compute_superpixel_label(sp_idx).unsqueeze(0)
            for sp_idx in range(segments.max() + 1)
        ])  # sp_labels : [sp_N, 2] ,代表超像素为每一类的概率; 其中sp_N 是超像素个数;

        # move labeled superpixels to the front of `sp_idx_list` 将带标签的超像素移到' sp_idx_list '的前面
        labeled_sps = (sp_labels.sum(dim=-1) > 0).nonzero(as_tuple=False).flatten()  # 列表,包含已经标注的超像素的id
        unlabeled_sps = (sp_labels.sum(dim=-1) == 0).nonzero(as_tuple=False).flatten()
        sp_idx_list = torch.cat([labeled_sps, unlabeled_sps])

        # quantize superpixel labels (e.g., from (0.7, 0.3) to (1.0, 0.0))
        # 量化超像素标签(例如，从(0.7,0.3)到(1.0,0.0))
        sp_labels = sp_labels[labeled_sps]   # 拿到已标注的超像素标签
        sp_labels = (sp_labels == sp_labels.max(dim=-1, keepdim=True)[0]).float()

    else:  # no supervision provided  没有提供监督
        sp_labels = empty_tensor().to(segments.device)

    # stacking normalized superpixel segment maps 叠加归一化超像素段映射
    sp_maps = segments == sp_idx_list[:, None, None]
    sp_maps = sp_maps.squeeze().float()  # size: (S_N,W,H)

    # make sure each superpixel map sums to one  确保每个超像素映射和为1.  之前超像素每个像素标签都是1，此时所有像素标签sum为1
    sp_maps = sp_maps / sp_maps.sum(dim=(1, 2), keepdim=True)

    return sp_maps, sp_labels


def _cross_entropy(y_hat, y_true, class_weights=None, epsilon=1e-7):
    """Semi-supervised cross entropy loss function. 半监督交叉熵损失函数。
    Args:
        y_hat: prediction tensor with size (N, C), where C is the number of classes 预测张量,size:(N, C)，C是类的数量
        y_true: label tensor with size (N, C). A sample won't be counted into loss
            if its label is all zeros.    标注张量,size:(N, C)。如果它的标签全是0,样本不会计入损失
        class_weights: class weights tensor with size (C,)  具有大小(C，)的类权张量
        epsilon: numerical stability term   数值稳定的术语
    Returns:
        cross_entropy: cross entropy loss computed only on samples with labels  交叉熵损失仅对带有标签的样本进行计算
    """

    device = y_hat.device

    # clamp all elements to prevent numerical overflow/underflow  夹紧所有元件，防止数值溢出/下流
    y_hat = torch.clamp(y_hat, min=epsilon, max=(1 - epsilon))    # 将y_hat缩放到(0,1)之间 (本来就应该在0-1之间，这样是为了防止意外)

    # number of samples with labels  带标签的样品数量
    labeled_samples = torch.sum(y_true.sum(dim=1) > 0).float()

    if labeled_samples.item() == 0:    # 都是未标注的话,直接返回0
        return torch.tensor(0.).to(device)

    ce = -y_true * torch.log(y_hat)

    if class_weights is not None:   # 给不同的类,分配不同的权值,一般为None
        ce = ce * class_weights.unsqueeze(0).float()

    return torch.sum(ce) / labeled_samples

# =========== 生成邻接矩阵
# sp_features： （N,32）
def _adj(sp_features):
    # adj.size: (N,N)
    low_features = sp_features[:,:16]
    # feature affinity matrix  特征关联矩阵
    # features - features.unsqueeze(1): size(N,N,D)
    adj1 = torch.exp(-torch.einsum('ijk,ijk->ij',    # 爱因斯坦求和 （einsum）
                                low_features - low_features.unsqueeze(1),
                                low_features - low_features.unsqueeze(1)))
    high_features = sp_features[:,16:]
    adj2 = torch.exp(-torch.einsum('ijk,ijk->ij',  # 爱因斯坦求和 （einsum）
                                   high_features - high_features.unsqueeze(1),
                                   high_features - high_features.unsqueeze(1)))
    return [adj1,adj2]

# def smoothness_reg(gcn_out, adj_list, loss, reg_scalar):
def smoothness_reg(gcn_out, adj_list):
    loss = 0
    for i in range(len(adj_list)):
        # the first power of each graph in the list ...  maybe transform to Laplacian?
        # 列表中每个图的第一次方…或者转换成拉普拉斯式?
        # pre_trace_tensor= torch.matmul( torch.transpose(torch.matmul( adj_list[i][0],gcn_out)), gcn_out)
        # pre_trace_tensor = reg_scalar*pre_trace_tensor
        # pre_reg = torch.trace(pre_trace_tensor) / tf.cast(torch.shape(gcn_out)[0] * torch.shape(gcn_out)[1], 'float32')
        # pre_reg = torch.trace(pre_trace_tensor) / torch.FloatTensor( gcn_out.size[0] * gcn_out.size[1] )

        one = torch.matmul(gcn_out.t(), adj_list[i])
        res = torch.matmul(one, gcn_out)
        tr = torch.trace(res)
        pre_reg = tr / float(gcn_out.size()[0] * gcn_out.size()[1])
        loss+= pre_reg
    return loss


class TGCNConfig(BaseConfig):
    """Configuration for TGCN model. 为TGCN型配置 """

    # Rescale factor to subsample input images. 重新缩放因子的子样本输入图像。
    # rescale_factor = 0.5
    rescale_factor = 0.5   #zqh

    # multi-scale range for training  多尺度范围训练
    multiscale_range = (0.3, 0.4)

    # Number of target classes.
    n_classes = 2

    # Class weights for cross-entropy loss function.  交叉熵损失函数的类权值
    class_weights = (3, 1)  # default = (3,1)

    # Superpixel parameters.
    sp_area = 150   # 50 / 200
    sp_compactness = 40

    # Optimization parameters.
    momentum = 0.9
    weight_decay = 0.001

    # Whether to freeze backbone. 是否冻结骨干网
    freeze_backbone = False

    # Training configurations.
    batch_size = 1
    epochs = 200

    lr = 6e-4  # 6e-4

    is_gcn = True
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

        # ============================= 图网络初始化
        #  层次化特征图
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
        self.dropout = 0.8      # 图网络dropout 比例
        self.gcn_output = None  # 图网络output
        self.adj_list = None
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

        # extract conv feature maps and flatten 提取卷积特征图并flatten
        self.feature_maps = None
        _ = self.backbone(x)
        x = self.feature_maps      # size:(fm_channels_sum,H,W)
        x = x.view(x.size(0), -1)  # flatten,得到size(fm_channels_sum,H*W)

        # calculate features for each superpixel 计算每个超像素的特征
        sp_maps = sp_maps.view(sp_maps.size(0), -1)  # flatten,得到size(N,H*W)
        x = torch.mm(sp_maps, x.t())                 # 矩阵相乘, 得到size(N,fm_channels_sum)  ps: x.t()转置

        # 利用全连通层降低超像素特征维数 reduce superpixel feature dimensions with fully connected layers
        x = self.fc_layers(x)       # reduce,得到size(N ,D),D=32
        self.sp_features = x       # 得到超像素特征

        # TGCN =============
        if self.kwargs.get('is_gcn'):
            #  层次化特征 构建多图
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
            optimizer, 'min', patience=10, factor=0.5, min_lr=1e-5, verbose=True)

        return optimizer, None

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
        _, sp_labels = target

        # sp_features = self.model.sp_features  # self.model = TGCN,之前标签传递要用到。现在用不到。
        sp_pred = self.model.sp_pred

        if sp_pred is None:
            raise RuntimeError('You must run a forward pass before computing loss. 在计算损失之前，必须进行前向传递')

        # 超像素的总数seed
        total_num = sp_pred.size(0)

        # number of labeled superpixels 标记的超像素数
        labeled_num = sp_labels.size(0)

        if labeled_num < total_num: # weakly-supervised mode  weakly-supervised模式(点注释模式)
            loss = self.xentropy(sp_pred[:labeled_num], sp_labels)   # 只取已标注的超像素进行loss计算
            if metrics is not None and isinstance(metrics, dict):
                metrics['labeled_sp_ratio'] = labeled_num / total_num    # 已标注的超像素比例
        else:  # fully-supervised mode 全监督模式
            loss = self.xentropy(sp_pred, sp_labels)

        regloss = 0
        # TGCN 正则化损失
        if self.kwargs.get('is_gcn'):
            gcn_out = self.model.gcn_output
            tgcn_loss = smoothness_reg(gcn_out, self.model.adj_list )
            regloss = self.kwargs.get('gcn_smooth_reg_weight') * tgcn_loss
            # self.logger.info( '正则化损失：',regloss.item())
            loss += regloss

        # clear outdated superpixel prediction 清除过时的超像素预测
        self.model.sp_pred = None

        # clear outdated superpixel prediction 清除过时的超像素预测
        self.model.gcn_output = None
        self.model.adj_list = None

        return loss,regloss

    def postprocess(self, pred, target=None):
        pred = pred.round().long()  # round(): 返回一个新张量，将pred张量每个元素舍入到最近的整数
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


class TGCNPixelInference(TGCN):
    """Weakly supervised histopathology image segmentation with sparse point annotations."""

    def __init__(self, n_classes=2, D=32, **kwargs):
        """Initialize a TGCN model.
        Kwargs:
            n_classes: number of target classes (default to 2)
            D: output dimension of superpixel features
        Returns:
            model: a new TGCN model
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

        # fully-connected layers for dimensionality reduction 全连接层降维
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
            nn.Linear(D, self.kwargs.get('n_classes', 2)),
            nn.Softmax(dim=1)
        )

        # ============================= 图网络初始化
        self.gcn_nfeat = D
        self.gcn_nhid = D*2
        self.adj_len = 2
        self.gcn_out_dim = 2
        self.gc1 = AdaptiveGraphConvolution(in_features_dim  = self.gcn_nfeat,
                                            out_features_dim = self.gcn_nhid,
                                            len = self.adj_len, bias=True)
        self.gc2 = AdaptiveGraphRecursiveConvolution(in_features_dim=self.gcn_nhid,
                                                     net_input_dim=self.gcn_nfeat,
                                                     out_features_dim=self.gcn_out_dim,
                                                     len = self.adj_len, bias=True)
        self.dropout = 0.8      # 图网络dropout 比例
        self.gcn_output = None  # 图网络output
        self.adj_list = None
        # ============================= =============================

        # store conv feature maps 存储conv特性映射
        self.feature_maps = None

        # spatial size of first feature map 第一个特征图的空间大小
        self.fm_size = None

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
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x.t())

        x = self.classifier(x)
        return x.view(height, width, -1)
