# -*- coding:utf-8 -*-
# @Time   : 2021/11/19 12:14
# @Author : 张清华
# @File   : handler.py
# @Note   : TGCN 的处理函数

import numpy as np
import torch

from utils_network import empty_tensor
from utils_network import is_empty_tensor


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
        y_true: label tensor with size (N, C). A sample won't be counted into loss if its label is all zeros.
               标注张量,size:(N, C)。如果它的标签全是0,样本不会计入损失
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

# 手工特征邻接矩阵
def art_adj(art_features):
    # adj.size: (9,)
    # adj1 = torch.ones((9,9)).to(torch.float32).cuda()
    # return [adj1]

    # feature affinity matrix  特征关联矩阵
    # features - features.unsqueeze(1): size(N,N,D)
    adj1 = torch.exp(-torch.einsum('ijk,ijk->ij',    # 爱因斯坦求和 （einsum）
                                art_features - art_features.unsqueeze(1),
                                art_features - art_features.unsqueeze(1)))
    return [adj1]

# 拉普拉斯正则化
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


def tgcn_propagation(gcn_out,labeled_num, threshold=0.93):
    '''
        gcn_out: [N_sp,2]
        y_l : [N_label_sp,2]
    '''

    # number of labeled and unlabeled samples  贴有标签和未贴有标签的样品数量
    n_l = labeled_num  # 有标签的数量； n_label
    n_u = gcn_out.size(0) - n_l  # n_unlabel

    # initialize y_u with zeros  用零初始化y_u,torch.Size([233, 2])
    # y_u : 未标记像本的y
    y_u = torch.zeros(n_u, 2).to(gcn_out.device)
    for index,one_out in enumerate(gcn_out):
        # max_p,index = one_out.max(dim=0)
        # 大于阈值
        if index>=n_l and max(one_out)>threshold:
            y_u[index-n_l][one_out.argmax()] = 1
    return y_u

# 读取resnet 参数
def load_model_weights(model, weights):
    model_dict = model.state_dict()
    weights = {k: v for k, v in weights.items() if k in model_dict}
    if weights == {}:
        print('No weight could be loaded..')
    model_dict.update(weights)
    model.load_state_dict(model_dict)

    return model
