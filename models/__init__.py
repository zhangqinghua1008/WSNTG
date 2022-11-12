# sys.path.append(str(Path(__file__).parent.parent.absolute()))
from .unext import UNeXt, UNeXtConfig, UNeXtTrainer
from .fcn import FCN32s, FCNConfig, FCNTrainer
from .TGCN.tgcn import TGCN, TGCNConfig, TGCNTrainer
from .wesup import WESUP, WESUPConfig, WESUPTrainer
from .cdws_mil import CDWS, CDWSConfig, CDWSTrainer
from .sizeloss import SizeLoss, SizeLossConfig, SizeLossTrainer
from .unet import UNET, UNETTrainer, UNETConfig
from .Yamu import YAMU, YAMUConfig, YAMUTrainer
from .WSGNet import WSGNet, WSGNetConfig, WSGNetTrainer


def initialize_trainer(model_type, **kwargs):
    """Initialize a trainer for model.  初始化模型的训练器。
    Args:
        model_type: either 'wesup', 'wesupv2, 'cdws' 'sizeloss' or 'mild'
        kwargs: additional training config 训练配置
    Returns:
        model: a model instance with given type  具有给定类型的模型实例
    """

    if model_type == 'tgcn':
        kwargs = {**TGCNConfig().to_dict(), **kwargs}
        model = TGCN(**kwargs)
        trainer = TGCNTrainer(model, **kwargs)
    elif model_type == 'wsgsn':
        kwargs = {**WSGNetConfig().to_dict(), **kwargs}
        model = WSGNet(**kwargs)
        trainer = WSGNetTrainer(model, **kwargs)
    elif model_type == 'wesup':
        kwargs = {**WESUPConfig().to_dict(), **kwargs}
        model = WESUP(**kwargs)
        trainer = WESUPTrainer(model, **kwargs)
    elif model_type == 'fcn':
        kwargs = {**FCNConfig().to_dict(), **kwargs}
        model = FCN32s(2)
        trainer = FCNTrainer(model, **kwargs)
    elif model_type == 'unet':
        kwargs = {**UNETConfig().to_dict(), **kwargs}
        model = UNET()
        trainer = UNETTrainer(model, **kwargs)
    elif model_type == 'yamu':
        kwargs = {**YAMUConfig().to_dict(), **kwargs}
        model = YAMU(2)
        trainer = YAMUTrainer(model, **kwargs)
    elif model_type == 'cdws':
        kwargs = {**CDWSConfig().to_dict(), **kwargs}
        model = CDWS(**kwargs)
        trainer = CDWSTrainer(model, **kwargs)
    elif model_type == 'sizeloss':
        kwargs = {**SizeLossConfig().to_dict(), **kwargs}
        model = SizeLoss(**kwargs)
        trainer = SizeLossTrainer(model, **kwargs)
    elif model_type == 'unext':
        kwargs = {**UNeXtConfig().to_dict(), **kwargs}
        model = UNeXt()
        trainer = UNeXtTrainer(model, **kwargs)
    else:
        raise ValueError(f'Unsupported model: {model_type}')

    return trainer
