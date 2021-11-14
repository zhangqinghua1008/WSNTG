import math
import numpy as np
import torch
from collections import OrderedDict

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
'''
    layers中主要定义了图数据实现卷积操作的层，类似于CNN中的卷积层，只是一个层而已。
    本节将分别通过属性定义、参数初始化、前向传播以及字符串表达四个方面对代码进一步解析。
'''

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    '''
        GraphConvolution作为一个类，定义其相关属性。
        主要定义了其两个输入:输入特征in_feature、输出特征out_feature，
        以及权重weight和偏移向量bias两个参数，同时调用了其参数初始化的方法。
    '''
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))   # weight要后面训练的，故用parameter定义
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))   # bias要后面训练的，故也用parameter定义
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    # 参数初始化
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))   # size()函数主要是用来统计矩阵元素个数，或矩阵某一维上的元素个数的函数  size（1）为行
        self.weight.data.uniform_(-stdv, stdv)       # uniform() ：随机生成下一个实数，在 [x, y] 范围内
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    #  前向传播，一般是 A ∗ X ∗ W 的计算方法，  由于A是一个sparse变量，
    #  因此其与X进行卷积的结果也是稀疏矩阵。
    def forward(self, input, adj):
        # torch.mm(a, b)是矩阵a和b矩阵相乘，torch.mul(a, b)是矩阵a和b对应位相乘，a和b的维度必须相等
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)   # torch.spmm(a,b)是稀疏矩阵相乘
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GraphConvolution_Line(Module):  # 原始GCN作者代码，只使用全连接层代替，代替后因为参数初始化不确定，导致结果不稳定
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    '''
        GraphConvolution作为一个类，定义其相关属性。主要定义了其两个输入:输入特征in_feature、
        输出特征out_feature，以及权重weight和偏移向量bias两个参数，同时调用了其参数初始化的方法。
    '''
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution_Line, self).__init__()
        self.line1 = torch.nn.Linear(in_features,out_features,bias = bias)

    #  前向传播，一般是 A ∗ X ∗ W 的计算方法，  由于A是一个sparse变量，因此其与X进行卷积的结果也是稀疏矩阵。
    def forward(self, input, adj):
        # torch.mm(a, b)是矩阵a和b矩阵相乘，torch.mul(a, b)是矩阵a和b对应位相乘，a和b的维度必须相等
        support = self.line1(input)
        output = torch.spmm(adj, support)   # torch.spmm(a,b)是稀疏矩阵相乘
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class AdaptiveGraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    '''
        GraphConvolution作为一个类，定义其相关属性。主要定义了其两个输入:输入特征维度in_feature、
        输出特征out_feature，以及权重weight和偏移向量bias两个参数，同时调用了其参数初始化的方法。
    '''
    def __init__(self, in_features_dim, out_features_dim, len , bias=True):
        super(AdaptiveGraphConvolution, self).__init__()
        self.in_features = in_features_dim
        self.out_features = out_features_dim
        self.bias = bias

        layers_line = OrderedDict()
        for i in range(len):
            layers_line[str(i)] = torch.nn.Linear(in_features_dim, out_features_dim, bias=False)
        self.line = torch.nn.Sequential(layers_line)
        self.graph_mixing_weight = Parameter(torch.FloatTensor(len, 1))   # weight要后面训练的，故用parameter定义

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features_dim))
        else: self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # torch.nn.init.kaiming_uniform(self.graph_mixing_weight)
        self.graph_mixing_weight.data.uniform_(-0.25, 0.25)
        if self.bias is not None:
            # torch.nn.init.normal_(self.bias, mean=0, std=2)
            # torch.nn.init.uniform_(self.bias, -0.25, 0.25)
            self.bias.data.uniform_(-0.25, 0.25)


    #  前向传播，一般是 A ∗ X ∗ W 的计算方法，  由于A是一个sparse变量，因此其与X进行卷积的结果也是稀疏矩阵。
    # input : torch.Size([2708, 1433]) 就是 (N样本数,D特征维度)
    # adj_list : 存放多个邻接矩阵。 每个邻接矩阵： torch.Size([2708, 2708]) -> (N样本数,N)
    def forward(self, input, adj_list):
        graphs = adj_list
        # convolve
        graph_outputs = list()
        for i,graph in enumerate(graphs):
            # implement the k-hop neighborhood aggregation  实现k-hop邻域聚合
            pre_sup = self.line[i](input)  # support = torch.mm(input, self.weight)
            graph_prod = torch.spmm(graph, pre_sup)  # output = torch.spmm(adj, support)  # torch.spmm(a,b)是稀疏矩阵相乘

            # combines different hops  结合不同的hops
            graph_outputs.append(graph_prod)

        graph_outputs = torch.stack(graph_outputs,dim=0)
        # implement mixing of the different graphs 实现不同图形的混合 || （GAM）模块
        # dims=([0], [0])，不是取第0轴，而是除去0轴，所以要取的是除第0轴外的
        # output: torch.Size([2708, 64])
        output = torch.squeeze(torch.tensordot(self.graph_mixing_weight,graph_outputs, dims=([0], [0])))
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' ('+ str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class AdaptiveGraphRecursiveConvolution(Module):
    '''
        GraphConvolution作为一个类，定义其相关属性。主要定义了其两个输入:输入特征维度in_feature、
        输出特征维度out_feature，以及权重weight和偏移向量bias两个参数，同时调用了其参数初始化的方法。
    '''
    # 相比AGCN 输入参数多了：net_input、net_input_dim、first_layer
    def __init__(self, in_features_dim, net_input_dim, out_features_dim, len , bias=True):
        super(AdaptiveGraphRecursiveConvolution, self).__init__()
        self.in_features = in_features_dim
        self.net_input = net_input_dim  # AGRCN
        self.out_features = out_features_dim
        self.bias = bias
        self.input_bias = bias

        layers_line = OrderedDict()
        for i in range(len):
            layers_line[str(i)] = torch.nn.Linear(in_features_dim, out_features_dim, bias=False)
        self.line = torch.nn.Sequential(layers_line)
        self.graph_mixing_weight = Parameter(torch.FloatTensor(len, 1))   # weight要后面训练的，故用parameter定义

        # AGRCN
        input_layers_line = OrderedDict()
        for i in range(len):
            input_layers_line[str(i)] = torch.nn.Linear(net_input_dim, out_features_dim, bias=False)
        self.input_layers_line = torch.nn.Sequential(input_layers_line)   # 用来给X原始输入做全连接的
        self.inp_graph_mixing_weight = Parameter(torch.FloatTensor(len, 1))  # weight要后面训练的，故用parameter定义

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features_dim))
            self.input_bias = Parameter(torch.FloatTensor(out_features_dim))
        else:
            self.register_parameter('bias', None)
            self.register_parameter('input_bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # torch.nn.init.kaiming_uniform(self.graph_mixing_weight)
        # torch.nn.init.kaiming_uniform(self.inp_graph_mixing_weight)
        self.graph_mixing_weight.data.uniform_(-0.25,0.25)
        self.inp_graph_mixing_weight.data.uniform_(-0.25,0.25)
        if self.bias is not None:
            # torch.nn.init.normal_(self.bias, mean=0, std=2)
            # torch.nn.init.normal_(self.input_bias, mean=0, std=2)
            self.bias.data.uniform_(-0.25,0.25)
            self.input_bias.data.uniform_(-0.25,0.25)

    #  前向传播，一般是 A ∗ X ∗ W 的计算方法，  由于A是一个sparse变量，因此其与X进行卷积的结果也是稀疏矩阵。
    def forward(self, input, X,  adj_list):
        graphs = adj_list

        # convolve
        graph_outputs = list()
        X_graph_outputs = list() # 原始X输入
        for i,graph in enumerate(graphs):
            # implement the k-hop neighborhood aggregation  实现k-hop邻域聚合
            pre_sup = self.line[i](input)  # support = torch.mm(input, self.weight)
            graph_prod = torch.spmm(graph, pre_sup)  # output = torch.spmm(adj, support)  # torch.spmm(a,b)是稀疏矩阵相乘

            # combines different hops  结合不同的hops
            graph_outputs.append(graph_prod)

            # ———————————————— 原始X输入
            X_pre_sup = self.input_layers_line[i](X)  # support = torch.mm(input, self.weight)
            X_graph_prod = torch.spmm(graph, X_pre_sup)  # output = torch.spmm(adj, support)  # torch.spmm(a,b)是稀疏矩阵相乘
            X_graph_outputs.append(X_graph_prod)

        graph_outputs = torch.stack(graph_outputs,dim=0)
        X_graph_outputs = torch.stack(X_graph_outputs,dim=0)
        # implement mixing of the different graphs 实现不同图形的混合 || （GAM）模块
        # dims=([0], [0])，不是取第0轴，而是除去0轴，所以要取的是除第0轴外的
        output = torch.squeeze(torch.tensordot(self.graph_mixing_weight,graph_outputs, dims=([0], [0])))
        X_output = torch.squeeze(torch.tensordot(self.inp_graph_mixing_weight,X_graph_outputs, dims=([0], [0])))

        # 跳跃连接拼接
        # output = torch.add(output, X_output)
        # output+= X_output
        output.add_(X_output)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' ('+ str(self.in_features) + ' -> ' + str(self.out_features) + ')'


# ----------------
# class AdaptiveGraphConvolution(Module):
#     """
#     Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
#     """
#     '''
#         GraphConvolution作为一个类，定义其相关属性。主要定义了其两个输入:输入特征in_feature、
#         输出特征out_feature，以及权重weight和偏移向量bias两个参数，同时调用了其参数初始化的方法。
#     '''
#     def __init__(self, in_features, out_features, len , bias=True):
#         super(AdaptiveGraphConvolution, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.bias = bias
#
#         layers_line = OrderedDict()
#         for i in range(len):
#             layers_line[str(i)] = torch.nn.Linear(in_features, out_features, bias=False)
#         self.line = torch.nn.Sequential(layers_line)
#         self.graph_mixing_weight = Parameter(torch.FloatTensor(len, 1))   # weight要后面训练的，故用parameter定义
#
#         if bias:
#             self.bias = Parameter(torch.FloatTensor(out_features))
#         else: self.register_parameter('bias', None)
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         torch.nn.init.kaiming_uniform(self.graph_mixing_weight)
#         if self.bias is not None:
#             torch.nn.init.normal_(self.bias, mean=0, std=2)
#
#     #  前向传播，一般是 A ∗ X ∗ W 的计算方法，  由于A是一个sparse变量，因此其与X进行卷积的结果也是稀疏矩阵。
#     def forward(self, input, adj_list):
#         graphs = adj_list
#
#         # convolve
#         graph_outputs = list()
#         for i,graph in enumerate(graphs):
#             # implement the k-hop neighborhood aggregation  实现k-hop邻域聚合
#             pre_sup = self.line[i](input)  # support = torch.mm(input, self.weight)
#             graph_prod = torch.spmm(graph, pre_sup)  # output = torch.spmm(adj, support)  # torch.spmm(a,b)是稀疏矩阵相乘
#
#             # combines different hops  结合不同的hops
#             graph_outputs.append(graph_prod)
#
#         graph_outputs = torch.stack(graph_outputs,dim=0)
#         # implement mixing of the different graphs 实现不同图形的混合 || （GAM）模块
#         # dims=([0], [0])，不是取第0轴，而是除去0轴，所以要取的是除第0轴外的
#         output = torch.squeeze(torch.tensordot(self.graph_mixing_weight,graph_outputs, dims=([0], [0])))
#         if self.bias is not None:
#             return output + self.bias
#         else:
#             return output
#
#     def __repr__(self):
#         return self.__class__.__name__ + ' ('+ str(self.in_features) + ' -> ' + str(self.out_features) + ')'
#
# class AdaptiveGraphRecursiveConvolution(Module):
#     '''
#         GraphConvolution作为一个类，定义其相关属性。主要定义了其两个输入:输入特征in_feature、
#         输出特征out_feature，以及权重weight和偏移向量bias两个参数，同时调用了其参数初始化的方法。
#     '''
#     # 相比AGCN 输入参数多了：net_input、net_input_dim、first_layer
#     def __init__(self, in_features, net_input, out_features, len , bias=True):
#         super(AdaptiveGraphRecursiveConvolution, self).__init__()
#         self.in_features = in_features
#         self.net_input = net_input  # AGRCN
#         self.out_features = out_features
#         self.bias = bias
#         self.input_bias = bias
#
#         layers_line = OrderedDict()
#         for i in range(len):
#             layers_line[str(i)] = torch.nn.Linear(in_features, out_features, bias=False)
#         self.line = torch.nn.Sequential(layers_line)
#         self.graph_mixing_weight = Parameter(torch.FloatTensor(len, 1))   # weight要后面训练的，故用parameter定义
#
#         # AGRCN
#         input_layers_line = OrderedDict()
#         for i in range(len):
#             input_layers_line[str(i)] = torch.nn.Linear(net_input, out_features, bias=False)
#         self.input_layers_line = torch.nn.Sequential(input_layers_line)   # 用来给X原始输入做全连接的
#         self.inp_graph_mixing_weight = Parameter(torch.FloatTensor(len, 1))  # weight要后面训练的，故用parameter定义
#
#         if bias:
#             self.bias = Parameter(torch.FloatTensor(out_features))
#             self.input_bias = Parameter(torch.FloatTensor(out_features))
#         else:
#             self.register_parameter('bias', None)
#             self.register_parameter('input_bias', None)
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         torch.nn.init.kaiming_uniform(self.graph_mixing_weight)
#         torch.nn.init.kaiming_uniform(self.inp_graph_mixing_weight)
#         if self.bias is not None:
#             torch.nn.init.normal_(self.bias, mean=0, std=2)
#             torch.nn.init.normal_(self.input_bias, mean=0, std=2)
#
#     #  前向传播，一般是 A ∗ X ∗ W 的计算方法，  由于A是一个sparse变量，因此其与X进行卷积的结果也是稀疏矩阵。
#     def forward(self, input, X,  adj_list):
#         graphs = adj_list
#
#         # convolve
#         graph_outputs = list()
#         X_graph_outputs = list() # 原始X输入
#         for i,graph in enumerate(graphs):
#             # implement the k-hop neighborhood aggregation  实现k-hop邻域聚合
#             pre_sup = self.line[i](input)  # support = torch.mm(input, self.weight)
#             graph_prod = torch.spmm(graph, pre_sup)  # output = torch.spmm(adj, support)  # torch.spmm(a,b)是稀疏矩阵相乘
#
#             # combines different hops  结合不同的hops
#             graph_outputs.append(graph_prod)
#
#             # ———————————————— 原始X输入
#             X_pre_sup = self.input_layers_line[i](X)  # support = torch.mm(input, self.weight)
#             X_graph_prod = torch.spmm(graph, X_pre_sup)  # output = torch.spmm(adj, support)  # torch.spmm(a,b)是稀疏矩阵相乘
#             X_graph_outputs.append(X_graph_prod)
#
#         graph_outputs = torch.stack(graph_outputs,dim=0)
#         X_graph_outputs = torch.stack(X_graph_outputs,dim=0)
#         # implement mixing of the different graphs 实现不同图形的混合 || （GAM）模块
#         # dims=([0], [0])，不是取第0轴，而是除去0轴，所以要取的是除第0轴外的
#         output = torch.squeeze(torch.tensordot(self.graph_mixing_weight,graph_outputs, dims=([0], [0])))
#         X_output = torch.squeeze(torch.tensordot(self.inp_graph_mixing_weight,X_graph_outputs, dims=([0], [0])))
#
#         # 跳跃连接拼接
#         # output = torch.add(output, X_output)
#         output+= X_output
#         if self.bias is not None:
#             return output + self.bias
#         else:
#             return output
#
#     def __repr__(self):
#         return self.__class__.__name__ + ' ('+ str(self.in_features) + ' -> ' + str(self.out_features) + ')'
#
