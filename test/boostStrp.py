import numpy as np
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error, mean_absolute_error


def MAE(arrays):
    y_true, y_pred = np.array(arrays)[:, 0], np.array(arrays)[:, 1]
    return mean_absolute_error(y_true, y_pred)


def RMSE(arrays):
    y_true, y_pred = np.array(arrays)[:, 0], np.array(arrays)[:, 1]
    return mean_squared_error(y_true, y_pred, squared=False)


def bootstrap(arrays, n_samples, n_replicates=1000, confidence=0.95, evaluate_method=np.mean):
    """
    利用bootstrap抽样进行置信区间估计
    :param arrays:形状为 (N,) 或 (N, N) 的 array-like 序列; N为样本数量；若形状为（N,N），其第一维为ground_truth，第二维为预测值
    :param n_samples:每次bootstrap抽样时选择的样本数量 n_samples<=N
    :param n_replicates:bootstrap抽样的次数
    :param confidence:置信度
    :param evaluate_method:计算评价指标的方法，evaluate_method的维度与arrays的维度一致
    :return:（原数据评价指标值, 置信度区间下界, 置信度区间上界）
    """
    eval_res = evaluate_method(arrays)  # 原数据评价指标值
    delta_list = []  # 用于存储delta的列表，delta=抽样数据的评价指标值-原数据的评价指标值
    for i in range(n_replicates):  # 进行n_replicates次抽样
        replicate_arrays = resample(arrays, n_samples=n_samples, replace=1)  # 对arrays进行有放回抽样，共抽样n_samples个
        delta_list.append(evaluate_method(replicate_arrays) - eval_res)  # 计算delta 并将其存入列表
    delta_list.sort()  # 对存储delta列表按从小到大排序
    lower_bound = eval_res - delta_list[int((1.0 - (1.0 - confidence) / 2.0) * n_replicates)]  # 计算置信度区间下界
    upper_bound = eval_res - delta_list[int((1.0 - confidence) / 2.0 * n_replicates)]  # 计算置信度区间上界
    return eval_res*100, lower_bound*100, upper_bound*100


def oneByone(dataSet, dir,throsd = 0.95):
    names = ["ac", "iou", "dice_sc","Jaccard","id"]

    # DP2019
    metirs = np.load(dir)
    print( metirs.shape , len(metirs))
    for i in range( metirs.shape[1] ):
        name = names[i]
        arr = metirs[:,i]
        print(dataSet," " ,name , "  :  ", bootstrap(arr, metirs.shape[0], n_replicates=1000, confidence=throsd, evaluate_method=np.mean))

if __name__ == "__main__":
    # arrays = [30, 37, 36, 43, 42, 43, 43, 46, 41, 42]
    # throsd = 0.95

    #DP2019
    dp_full = "D:\组会内容\实验报告\实验数据汇总/DP2019_tgcn_full_metrics.npy"
    dp_point = "D:\组会内容\实验报告\实验数据汇总/DP2019_tgcn_point_metrics.npy"

    sicapv2_full = "D:\组会内容\实验报告\实验数据汇总/SICAPV2_tgcn_full_metrics.npy"
    sicapv2_point = "D:\组会内容\实验报告\实验数据汇总/SICAPV2_tgcn_full_metrics.npy"

    CAMELYON16_full = "D:\组会内容\实验报告\实验数据汇总/CAMELYON16_tgcn_full_metrics.npy"
    CAMELYON16_point = "D:\组会内容\实验报告\实验数据汇总/CAMELYON16_tgcn_point_metrics.npy"

    oneByone(dataSet = "CAMELYON16_point", dir=CAMELYON16_point)


    # dp_point = np.load("D:\组会内容\实验报告\实验数据汇总/DP2019_tgcn_point_metrics.npy")
    # print( "DP2019 point : ", bootstrap(dp_point, 10, n_replicates=100, confidence=throsd, evaluate_method=np.mean))



    # arrays2 = [[30, 29], [37, 39], [36, 32], [43, 45], [42, 38], [43, 48], [43, 48], [46, 42], [41, 41], [42, 44]]
    # print(bootstrap(arrays2, 10, n_replicates=1000, confidence=0.95, evaluate_method=MAE))
    # print(bootstrap(arrays2, 10, n_replicates=1000, confidence=0.95, evaluate_method=RMSE))
