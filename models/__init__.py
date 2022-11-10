# sys.path.append(str(Path(__file__).parent.parent.absolute()))
from UNeXt import UNeXt, UNeXtConfig, UNeXtTrainer
from .fcn import FCN32s, FCNConfig, FCNTrainer
from WSNTG.wsntg import WSNTG, WSNTGConfig, WSNTGTrainer,WSNTGPixelInference
from .wesup import WESUP,WESUPConfig,WESUPTrainer
from .cdws_mil import CDWS, CDWSConfig, CDWSTrainer
from .test_unet import TEST_UNET,TEST_UNETConfig,TEST_UNETTrainer
from .sizeloss import SizeLoss, SizeLossConfig, SizeLossTrainer
from .unet import UNET,UNETTrainer,UNETConfig
from .Yamu import YAMU,YAMUConfig,YAMUTrainer
# from .hovernet.test_hover import Test_HoverNet,Test_HoverNetTrainer,Test_HoverNetConfig
# from .test_resnet18 import TEST_RESNET18Config,TEST_RESNET18Trainer,TEST_RESNET18


def initialize_trainer(model_type, **kwargs):
    """Initialize a trainer for model.  初始化模型的训练器。
    Args:
        model_type: either 'wesup', 'wesupv2, 'cdws' 'sizeloss' or 'mild'
        kwargs: additional training config 训练配置
    Returns:
        model: a model instance with given type  具有给定类型的模型实例
    """

    if model_type == 'WSNTG':
        kwargs = {**WSNTGConfig().to_dict(), **kwargs}
        model = WSNTG(**kwargs)
        trainer = WSNTGTrainer(model, **kwargs)
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
    elif model_type == 'testunet':
        kwargs = {**TEST_UNETConfig().to_dict(), **kwargs}
        model = TEST_UNET(**kwargs)
        trainer = TEST_UNETTrainer(model, **kwargs)
    elif model_type == 'cdws':
        kwargs = {**CDWSConfig().to_dict(), **kwargs}
        model = CDWS(**kwargs)
        trainer = CDWSTrainer(model, **kwargs)
    elif model_type == 'sizeloss':
        kwargs = {**SizeLossConfig().to_dict(), **kwargs}
        model = SizeLoss(**kwargs)
        trainer = SizeLossTrainer(model, **kwargs)
    elif model_type == 'UNeXt':
        kwargs = {**UNeXtConfig().to_dict(), **kwargs}
        model = UNeXt()
        trainer = UNeXtTrainer(model, **kwargs)
    else:
        raise ValueError(f'Unsupported model: {model_type}')

    return trainer
