"""
Metrics calculation utilities.  指标计算工具
Code of `detection_f1`, `object_dice` and `object_hausdorff` are adapted from
https://warwick.ac.uk/fac/sci/dcs/research/tia/glascontest/evaluation/evaluation_metrics_v6.zip.
"""
import functools
import torch
import numpy as np
from scipy import stats
from scipy.spatial.distance import directed_hausdorff
from skimage.measure import label
from sklearn.metrics import confusion_matrix


def convert_to_numpy(func):
    """Decorator for converting each argument to numpy array.
        修饰器，用于将每个参数转换为numpy数组
    """

    @functools.wraps(func)
    def wrapper(*args):
        args = [
            arg.detach().cpu().numpy() if torch.is_tensor(arg) else np.array(arg)
            for arg in args
        ]
        return func(*args)

    return wrapper

# DI
def dice(S, G, epsilon=1e-7):
    """Dice index for segmentation evaluation.  用于分割评价的骰子指数。(DI)
    Arguments:
        S: segmentation mask with shape (B, H, W)
        G: ground truth mask with shape (B, H, W)
        epsilon: numerical stability term
    Returns:
        dice_score: segmentation dice score
    """

    if torch.is_tensor(S) and torch.is_tensor(G):
        S = S.unsqueeze(0) if len(S.size()) == 2 else S
        G = G.unsqueeze(0) if len(G.size()) == 2 else G
        S, G = S.float(), G.float()

        if G.sum() == 0:  # 是阴性样本 .  则反转
            S = (S < 0.5)
            G = torch.ones_like(G)

        dice_score = 2 * (G * S).sum(dim=(1, 2)) / (G.sum(dim=(1, 2)) + S.sum(dim=(1, 2)) + epsilon)
        return dice_score.mean().item()

    S, G = np.array(S), np.array(G)
    S = np.expand_dims(S, 0) if len(S.shape) == 2 else S
    G = np.expand_dims(G, 0) if len(G.shape) == 2 else G
    dice_score = 2 * (G * S).sum(axis=(1, 2)) / (G.sum(axis=(1, 2)) + S.sum(axis=(1, 2)) + epsilon)
    return dice_score.mean()

#OA
def accuracy(P, G):
    """Classification accuracy.

    Arguments:
        P: prediction with size (B, H, W)
        G: ground truth tensor with the same size as P
    Returns:
        accuracy: classification accuracy
    """

    if torch.is_tensor(P) and torch.is_tensor(G):
        return (P == G).float().mean().item()

    return (np.array(P) == np.array(G)).mean()

#F1
@convert_to_numpy
def detection_f1(S, G, overlap_threshold=0.5, epsilon=1e-7):
    """F1-score for object detection.  对象检测f1评分。

    The ground truth for each segmented object is the object in the manual annotation
    that has maximum overlap with that segmented object.

    A segmented glandular object that intersects with at least 50% of its ground truth
    will be considered as true positive, otherwise it will be considered as false positive.
    A ground truth glandular object that has no corresponding segmented object or has
    less than 50% of its area overlapped by its corresponding segmented object will be
    considered as false negative.

    每个分割对象的ground truth为人工标注的对象,它与被分割的物体有最大的重叠
    与至少 50% 的地面实况相交的分段腺体对象将被视为真阳性，否则将被视为假阳性。
    没有对应的分割对象或与对应的分割对象重叠的区域少于 50% 的真实腺体对象将被视为假阴性。
    See more on https://warwick.ac.uk/fac/sci/dcs/research/tia/glascontest/evaluation/.

    Arguments:
        S: segmentation mask with shape (H, W)
        G: ground truth mask with the same shape as S
        overlap_threshold: overlap threshold for counting true positives
        epsilon: numerical stability term
    Returns:
        f1: detection F1 score
    """

    S, G = label(S), label(G)
    num_S, num_G = S.max(), G.max()

    if num_S == 0 and num_G == 0:
        return 1
    elif num_S == 0 or num_G == 0:
        return 0

    # matrix for identifying corresponding ground truth object in G for each segmented object in S
    # (the 1st col contains labels of segmented objects, the 2nd col contains corresponding ground
    # truth objects and the 3rd col is the true positive flags)
    tp_table = np.zeros((num_S, 3))
    tp_table[:, 0] = np.arange(1, num_S + 1)

    for seg_idx in range(num_S):
        intersect = G[S == tp_table[seg_idx, 0]]
        intersect = intersect[intersect > 0]
        if intersect.size > 0:
            tp_table[seg_idx, 1] = stats.mode(intersect)[0]

    for seg_idx in range(num_S):
        if tp_table[seg_idx, 1] != 0:
            seg_obj = S == tp_table[seg_idx, 0]
            gt_obj = G == tp_table[seg_idx, 1]
            overlap = seg_obj & gt_obj
            if overlap.sum() / gt_obj.sum() > overlap_threshold:
                tp_table[seg_idx, 2] = 1

    TP = np.sum(tp_table[:, 2] == 1)
    FP = np.sum(tp_table[:, 2] == 0)
    FN = num_G - TP

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    return (2*precision*recall) / (precision + recall + epsilon)

#OD
@convert_to_numpy
def object_dice(S, G):
    """Object-level Dice(OD)  index for segmentation evaluation.  用于分割评价的对象级骰子索引(OD)
    See more on https://warwick.ac.uk/fac/sci/dcs/research/tia/glascontest/evaluation/.
    Arguments:
        S: segmentation mask with shape (H, W)
        G: ground truth mask with shape (H, W)

    Returns:
        object_dice_score: segmentation object dice score
    """

    S, G = label(S), label(G)

    S_labels = np.unique(S)
    S_labels = S_labels[S_labels > 0]

    G_labels = np.unique(G)
    G_labels = G_labels[G_labels > 0]

    if len(S_labels) == 0 and len(G_labels) == 0:
        return 1
    elif len(S_labels) == 0 or len(G_labels) == 0:
        return 0

    S_obj_dice = 0
    S_total_area = (S > 0).sum()
    for seg_idx in S_labels:
        Si = S == seg_idx
        intersect = G[Si]
        intersect = intersect[intersect > 0]

        if intersect.size > 0:
            Gi = G == stats.mode(intersect)[0]
        else:
            Gi = np.zeros_like(G)

        omegai = Si.sum() / S_total_area
        S_obj_dice += omegai * dice(Si, Gi)

    G_obj_dice = 0
    G_total_area = (G > 0).sum()
    for gt_idx in G_labels:
        tilde_Gi = G == gt_idx
        intersect = S[tilde_Gi]
        intersect = intersect[intersect > 0]

        if intersect.size > 0:
            tilde_Si = S == stats.mode(intersect)[0]
        else:
            tilde_Si = np.zeros_like(S)

        tilde_omegai = tilde_Gi.sum() / G_total_area
        G_obj_dice += tilde_omegai * dice(tilde_Gi, tilde_Si)

    return (S_obj_dice + G_obj_dice) / 2


@convert_to_numpy
def hausdorff(S, G):
    """Symmetric hausdorff distance for shape similarity evaluation.
        形状相似度评价的对称hausdorff距离。
    Arguments:
        S: segmentation mask with shape (H, W)
        G: ground truth mask with shape (H, W)

    Returns:
        hausdorff_distance: hausdorff distance between S and G
    """

    if S.sum() == 0 and G.sum() == 0:
        return 0
    elif S.sum() == 0 or G.sum() == 0:
        return np.inf

    # coordinates of S
    Sc = np.column_stack(np.where(S > 0))

    # coordinates of G
    Gc = np.column_stack(np.where(G > 0))

    return max(directed_hausdorff(Sc, Gc)[0], directed_hausdorff(Gc, Sc)[0])

#OH
@convert_to_numpy
def object_hausdorff(S, G):
    """Object-level Hausdorff(OH) distance for shape similarity evaluation.
        形状相似度评价的目标级Hausdorff距离。
    See more on https://warwick.ac.uk/fac/sci/dcs/research/tia/glascontest/evaluation/.

    Arguments:
        S: segmentation mask with shape (H, W)
        G: ground truth mask with shape (H, W)

    Returns:
        object_hausdorff_distance: object hausdorff distance between S and G
    """

    S, G = label(S), label(G)

    S_total_area = (S > 0).sum()
    G_total_area = (G > 0).sum()

    S_labels = np.unique(S)
    S_labels = S_labels[S_labels > 0]

    G_labels = np.unique(G)
    G_labels = G_labels[G_labels > 0]

    # sum of all hausdorff distances
    hausdorff_sum = 0
    for seg_idx in S_labels:
        Si = S == seg_idx
        omegai = Si.sum() / S_total_area

        intersect = G[Si]
        intersect = intersect[intersect > 0]

        if intersect.size > 0:
            Gi = G == stats.mode(intersect)[0]
            hausdorff_sum += omegai * hausdorff(Si, Gi)
        elif len(G_labels) > 0:
            min_distance = min(hausdorff(Si, G == gt_idx) for gt_idx in G_labels)
            hausdorff_sum += omegai * min_distance

    # sum of all tilde hausdorff distances
    tilde_hausdorff_sum = 0
    for gt_idx in G_labels:
        tilde_Gi = G == gt_idx
        tilde_omegai = tilde_Gi.sum() / G_total_area

        intersect = S[tilde_Gi]
        intersect = intersect[intersect > 0]

        if intersect.size > 0:
            tilde_Si = S == stats.mode(intersect)[0]
            tilde_hausdorff_sum += tilde_omegai * hausdorff(tilde_Si, tilde_Gi)
        elif len(S_labels) > 0:
            min_distance = min(hausdorff(S == seg_idx, tilde_Gi) for seg_idx in S_labels)
            tilde_hausdorff_sum += tilde_omegai * min_distance

    return (hausdorff_sum + tilde_hausdorff_sum) / 2

# ---------------------

# IOU
def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        # output = torch.sigmoid(output).data.cpu().numpy()
        output = output.data.cpu().numpy()
        # output = torch.argmax(output, dim=1).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    # if target.sum()==0: # 是阴性样本 .  #反转
    #    output = np.int64(output<0.5)
    #    target = np.ones_like(target)
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)

    # dice = (2 * iou) / (iou + 1)
    # return iou, dice

# DICE
def dice_coef(output, target): #output为预测结果 target为真实结果
    '''
        2*| A∩B|  /  (|A|+|B|)
    '''
    smooth = 1e-5

    if torch.is_tensor(output):
        # output = torch.sigmoid(output).data.cpu().numpy()
        output = output.data.cpu().numpy()
        # output = torch.argmax(output, dim=1).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    if target.sum()==0: # 是阴性样本 .  则反转
       output = np.int64(output<0.5)
       target = np.ones_like(target)

    intersection = (output * target).sum()
    all = output.sum() + target.sum()

    return (2. * intersection + smooth) / (all + smooth)


# 其中TPR即为敏感度（sensitivity）,SE
def SE_sensitivity(output, target): #output为预测结果 target为真实结果
    # TPR：true positive rate，描述识别出的所有正例占所有正例的比例 计算公式为：TPR= TP/ (TP+ FN)
    smooth = 1e-5
    if torch.is_tensor(output):
        output = torch.argmax(output, dim=1).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    if target.sum() == 0:  # 是阴性样本 .  则反转
        output = np.int64(output < 0.5)
        target = np.ones_like(target)

     # TN|FP
     # FN|TP
    tn, fp, fn, tp = confusion_matrix(output, target)

    return (tp + smooth) / (tp + fn + smooth)

# TNR即为特异度（specificity）,
def SP_specificity(output, target): #output为预测结果 target为真实结果
    # sensitivity: TNR：true negative rate，描述识别出的负例占所有负例的比例
    #   计算公式为：TNR= TN / (FP + TN)
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.argmax(output, dim=1).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    if target.sum() == 0:  # 是阴性样本 .  则反转
        output = np.int64(output < 0.5)
        target = np.ones_like(target)

    # TN|FP
    # FN|TP
    tn, fp, fn, tp = confusion_matrix(output, target)
    return (tn + smooth) / (fp + tn + smooth)


# jaccard_index  https://github.com/assassint2017/MICCAI-LITS2017/blob/master/utilities/calculate_metrics.py
def jaccard_index(output, target):
    """
    :return: 杰卡德系数
    """
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.argmax(output, dim=1).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    if target.sum()==0: # 是阴性样本 .  则反转
       output = np.int64(output<0.5)
       target = np.ones_like(target)

    intersection = (output * target).sum()
    union = (output | target).sum()

    return intersection / union