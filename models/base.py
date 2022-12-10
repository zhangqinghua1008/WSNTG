import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from utils_network import underline, record
from utils_network.history import HistoryTracker


class BaseConfig:
    """A base model configuration class.  基本模型配置类。 """

    # batch size for training
    batch_size = 1

    # number of epochs for training
    epochs = 10

    # numerical stability term
    epsilon = 1e-7

    def __str__(self):
        return '\n'.join(f'{attr:<32s}{getattr(self, attr)}'
                         for attr in dir(self) if not attr.startswith('_'))

    def to_dict(self):  # 将数据框数据转换为字典形式
        return {
            attr: getattr(self, attr) for attr in dir(self)
            if not attr.startswith('_') and attr != 'to_dict'
        }


class BaseTrainer(ABC):
    """A base trainer class. 基本训练器类。"""

    def __init__(self, model, **kwargs):
        """Initialize a BaseTrainer.
        Args:
            model: a model for training (should be a `torch.nn.Module`)
            kwargs (optional): additional configuration
        Returns:
            trainer: a new BaseTrainer instance
        """

        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)     # 把model放到继承他的类下
        self.kwargs = kwargs

        # Initialize logger.
        if kwargs.get('logger'):
            self.logger = kwargs.get('logger')
        else:
            self.logger = logging.getLogger('Train')
            self.logger.setLevel(logging.DEBUG)
            self.logger.addHandler(logging.StreamHandler())

        # Training components. 训练组件
        self.initial_epoch = 1
        self.record_dir = None
        self.tracker = HistoryTracker()
        self.dataloaders = None
        self.optimizer, self.scheduler = None, None
        self.metric_funcs = []

    @abstractmethod
    def get_default_dataset(self, root_dir, train=True, proportion=1.0):
        """Get default dataset for training/validation.  获取用于培训/验证的默认数据集
        Args:
            root_dir: path to dataset root
            train: whether it is a dataset for training
            proportion: proportion of data to be used  所使用的数据的比例
        Returns:
            dataset: a `torch.utils.data.Dataset` instance
        """

    def get_default_optimizer(self):
        """Get default optimizer for training.
        Returns:
            optimizer: default model optimizer
            scheduler: default learning rate scheduler (could be `None`)
        """
        return torch.optim.SGD(self.model.parameters(), lr=1e-3), None

    def preprocess(self, *data):
        """Preprocess data from dataloaders and return model inputs and targets.
            预处理来自数据加载器的数据，并返回模型输入和目标。
        Args:
            *data: data returned from dataloaders  从数据加载器返回的数据
        Returns:
            input: input to feed into the model of size (B, H, W)  输入输入尺寸模型(B, H, W)
            target: desired output (or any additional information) to compute loss and evaluate performance
        """
        return [datum.to(self.device) for datum in data]

    @abstractmethod
    def compute_loss(self, pred, target, metrics=None):
        """Compute objective function.
        Args:
            pred: model prediction from the `forward` step
            target: target computed from `preprocess` method
            metrics: dict for tracking metrics when computing loss
        Returns:  loss: model loss
        """

    def postprocess(self, pred, target=None):
        """Postprocess raw prediction and target before calling `evaluate` method.
            在调用' evaluate '方法之前，对原始预测和目标进行后处理。
        Args:
            pred: prediction computed from the `forward` step
            target: target computed from `preprocess` method (optional)
        Returns:
            pred: postprocessed prediction
            target: postprocessed target (optional)
        """
        if target is not None:
            return pred, target
        return pred

    def post_epoch_hook(self, epoch):
        """Hook for post-epoch stage.
        Args:
            epoch: current epoch
        """
        pass

    def load_checkpoint(self, ckpt_path=None,model_type = None):
        """Load checkpointed model weights, optimizer states, etc, from given path.
            从给定路径加载 检查点模型权重、优化器状态等。
        Args:
            ckpt_path: path to checkpoint
        """
        if ckpt_path is not None:
            self.record_dir = Path(ckpt_path).parent.parent
            self.logger.info(f'Loading checkpoint from {ckpt_path}.')
            checkpoint = torch.load(ckpt_path, map_location=self.device)

            self.initial_epoch = checkpoint['epoch'] + 1
            self.model.load_state_dict(checkpoint['model_state_dict'])

            if self.optimizer is not None:
                self.optimizer.load_state_dict(
                    checkpoint['optimizer_state_dict'])

            if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(
                    checkpoint['scheduler_state_dict'])
        else:
            self.record_dir = Path(record.prepare_record_dir(model_type))
            record.copy_source_files(self.record_dir)

    def save_checkpoint(self, ckpt_path, **kwargs):
        """ 保存模型检查点
        Args:
            ckpt_path: path to checkpoint to be saved
            **kwargs: 要包含在检查点对象中的附加信息
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            **kwargs,
        }
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, ckpt_path)

    def train_one_iteration(self, index, phase, *data):
        """Hook for training one iteration.  训练一个迭代 / 训练一个batch
        Args:
            index: 代表第几个数据，用来保存中间结果
            phase: either 'train' or 'val'
            *data: input data
        """
        # 预处理，在相应的模型py文件中执行
        input_, target = self.preprocess(*data)
        metrics = dict()
        # .set_grad_enabled(Bool): 将梯度计算设置成打开或者关闭的上下文管理器.
        with torch.set_grad_enabled(phase == 'train'):
            # 模型的预测
            pred = self.model(input_)
            if phase == 'train':
                # 计算loss
                loss = self.compute_loss(pred, target, metrics=metrics)

                # 如果loss为nan，报错
                if torch.isnan(loss):
                    raise ValueError('Loss is nan!')
                metrics['loss'] = loss.item()

                # 梯度累加
                # accum_steps = 1  # 梯度累加的量
                # loss = loss / accum_steps

                loss.backward()

                # if (index + 1) % accum_steps == 0: # 累加到一定的论文再进行反向传播
                self.optimizer.step()
                self.optimizer.zero_grad()

        pred, target = self.postprocess(pred, target)  # 后处理

        # 保存部分模型训练时预测图
        if phase == 'train' and index % 5 == 0:
            record.save_preAndMask(self.record_dir, pred, target, index)  # 保存模型中间输出
        # 先执行self.evaluate(pred, target),然后把返回的参数 和 现有的metrics 进行合并, 进而传给step
        self.tracker.step({**metrics, **self.evaluate(pred, target)})
        return metrics

    # 1.每个epoch 需要执行的内容
    def train_one_epoch(self, no_val=False):
        """Hook for training one epoch.  Hook为训练一个epoch
        Args:
            no_val: 是否禁用验证
        """
        phases = ['train'] if no_val else ['train', 'val']
        for phase in phases:  # 依次执行训练和val 阶段
            self.logger.info(f'{phase.capitalize()} phase:')
            start = time.time()

            if phase == 'train':
                self.model.train()
                self.tracker.train()
            else:
                self.model.eval()
                self.tracker.eval()

            pbar = tqdm(self.dataloaders[phase], total=len(self.dataloaders[phase]))
            for index, data in enumerate(pbar):  # TODO: 训练前这个地方需要加载，很慢，而且占用内存很多
                try:
                    # 训练一个迭代-iteration
                    metrics = self.train_one_iteration(index, phase, *data)
                    pbar.set_postfix(metrics)  # 在进度条上输出loss等信息
                except RuntimeError as ex:
                    self.logger.exception(ex)

            # 记录信息
            self.logger.info(f'Took {time.time() - start:.2f}s.')  # 花费的时间
            self.logger.info(self.tracker.log())  # 记录度量信息
            pbar.close()  # 关闭进度条实例

    # 开始训练过程 ★★★★★
    def train(self, data_root, model_type, **kwargs):
        """Start training process.  开始培训过程 ★★★★★
        Args:
            data_root:  应该包含一个名为“train”的子目录（ 可选'val'）
        Kwargs (optional):
            metrics: 用于计算指标的函数列表
            checkpoint: path to checkpoint for resuming training
            epochs: number of epochs for training
            batch_size: mini-batch size for training
        """

        self.kwargs = {**self.kwargs, **kwargs}  # 合并配置

        self.optimizer, self.scheduler = self.get_default_optimizer()
        self.load_checkpoint(self.kwargs.get('checkpoint'), model_type=model_type)  # checkpoint 一般为Nne
        self.logger.addHandler(logging.FileHandler(self.record_dir / 'train.log', encoding='utf-8'))  # 将文件handler添加到logger
        serializable_kwargs = {  # serializable 可串行化的
            k: v for k, v in self.kwargs.items()
            if isinstance(v, (int, float, str, tuple))
        }
        # 记录实验参数
        record.save_params(self.record_dir, serializable_kwargs)  # 将实验参数保存到json
        self.logger.info(str(serializable_kwargs) + '\n')  # 将实验参数记录到.log
        self.tracker.save_path = self.record_dir / 'history.csv'

        val_path = self.getDataset(data_root)  # 加载数据集

        self.logger.info(underline('\nTraining Stage', '='))
        self.metric_funcs = self.kwargs.get('metrics')  # 度量函数

        epochs = self.kwargs.get('epochs')
        total_epochs = epochs + self.initial_epoch - 1

        # 进行n轮训练
        for epoch in range(self.initial_epoch, total_epochs + 1):
            startTime = time.time()

            self.logger.info(underline('\nEpoch {}/{}'.format(epoch, total_epochs), '-'))

            self.tracker.start_new_epoch(self.optimizer.param_groups[0]['lr'])
            self.train_one_epoch(no_val=(not val_path.exists()))
            self.post_epoch_hook(epoch)

            print("训练花费时间：", time.time() - startTime)

            self.tracker.save()  # 保存度量到CSV文件
            record.plot_learning_curves(self.tracker.save_path)  # 保存学习曲线

            # 保存检查点以便恢复训练
            if epoch % 2 == 0:
                ckpt_path = self.record_dir / 'checkpoints' / f'ckpt.{epoch:04d}.pth'
                self.save_checkpoint(ckpt_path, epoch=epoch, optimizer_state_dict=self.optimizer.state_dict())

            torch.cuda.empty_cache()
            print("花费时间：", time.time() - startTime)

        self.logger.info(self.tracker.report())

    def getDataset(self, data_root):
        '''
            加载数据: 加载train数据 和 val 数据
        '''
        data_root = Path(data_root)  # 使data_path 变成 pathlib.WindowsPath类型
        train_path = data_root / 'train'  # 拼接
        val_path = data_root / 'val'

        # 加载数据: 加载train数据
        train_dataset = self.get_default_dataset(train_path,
                                                 proportion=self.kwargs.get('proportion', 1))
        train_dataset.summary(logger=self.logger)  # 使用logger记录数据集信息
        self.dataloaders = {
            'train': torch.utils.data.DataLoader(
                train_dataset, batch_size=self.kwargs.get('batch_size'), shuffle=True,
                num_workers=4, drop_last=True)
        }
        # 加载val数据
        if val_path.exists():
            val_dataset = self.get_default_dataset(val_path, train=False)
            val_dataset.summary(logger=self.logger)
            self.dataloaders['val'] = torch.utils.data.DataLoader(
                val_dataset, batch_size=self.kwargs.get('batch_size'), num_workers=4)
        return val_path

    # 开启标配传播
    def openPropagated(self, epoch):
        # 等backbone恰当了再开启GCN
        # if epoch >= 1 and 'is_gcn' in self.kwargs:
        #     self.kwargs['is_gcn'] = True
        # 等backbone恰当了再去标签传播
        if epoch >= 6 and 'is_propagated' in self.kwargs:
            self.kwargs['is_propagated'] = True

    def evaluate(self, pred, target=None, verbose=False):
        """Running several metrics to evaluate model performance.  运行多个度量来评估模型性能。
        Args:
            pred: prediction of size (B, H, W), either torch.Tensor or numpy array  预测 size: (B, H, W)，
            target: ground truth of size (B, H, W), either torch.Tensor or numpy array 地面真实size:(B, H, W)
            verbose: whether to show progress bar  Verbose:是否显示进度条  . 猜想B为batchsize的大小？
        Returns:
            metrics: a dictionary containing all metrics metrics:包含所有度量标准的字典
        """

        if target is None:
            return dict()

        metrics = defaultdict(list)  # 初始化,指定value的类型为list

        iterable = zip(pred, target)
        if verbose:
            iterable = tqdm(iterable, total=len(pred))

        for P, G in iterable:
            for func in self.metric_funcs:
                metrics[func.__name__].append(func(P, G))

        return {k: np.mean(v) for k, v in metrics.items()}
