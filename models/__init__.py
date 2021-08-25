import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.absolute()))

from .wesup import WESUP, WESUPConfig, WESUPTrainer
from .tgcn import TGCN, TGCNConfig, TGCNTrainer
from .fcn import FCN32s, FCNConfig, FCNTrainer


def initialize_trainer(model_type, **kwargs):
    """Initialize a trainer for model.  初始化模型的训练器。
    Args:
        model_type: either 'wesup', 'wesupv2, 'cdws' 'sizeloss' or 'mild'
        kwargs: additional training config 训练配置
    Returns:
        model: a model instance with given type  具有给定类型的模型实例
    """

    if model_type == 'wesup':
        kwargs = {**WESUPConfig().to_dict(), **kwargs}
        model = WESUP(**kwargs)
        trainer = WESUPTrainer(model, **kwargs)
    elif model_type == 'tgcn':
        kwargs = {**TGCNConfig().to_dict(), **kwargs}
        model = TGCN(**kwargs)
        trainer = TGCNTrainer(model, **kwargs)
    elif model_type == 'fcn':
        kwargs = {**FCNConfig().to_dict(), **kwargs}
        model = FCN32s(2)
        trainer = FCNTrainer(model, **kwargs)
    else:
        raise ValueError(f'Unsupported model: {model_type}')

    return trainer
