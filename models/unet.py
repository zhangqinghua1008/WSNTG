import numpy as np
import torch
import torch.nn as nn

from utils_network.data import SegmentationDataset
from .base import BaseConfig, BaseTrainer

import segmentation_models_pytorch as smp
from torchsummary import summary


class UNETConfig(BaseConfig):
    batch_size = 8

    lr = 1e-3   # 6e-4

    epochs = 250

    n_classes = 2 # Number of target classes.

    target_size = (512, 512)

    # Optimization parameters. 优化参数
    momentum = 0.9
    weight_decay = 0.0005


class UNET(nn.Module):

    def __init__(self, n_classes=2):
        super().__init__()
        # encoder_weights="imagenet"
        self.backbone = smp.Unet(in_channels=3,classes=n_classes, encoder_name='vgg16',encoder_weights="imagenet")
        # self.backbone = smp.Unet(in_channels=3,classes=n_classes, encoder_name='vgg16',encoder_weights=None)

    def forward(self,x):
        backbone_out = self.backbone(x)
        return nn.functional.softmax(backbone_out,dim=1)


class UNETTrainer(BaseTrainer):
    """Trainer for FCN."""

    def __init__(self, model, **kwargs):
        """Initialize a MILDNetTrainer.
        Kwargs:
            input_size: input spatial size
            contour_threshold: threshold for predicting contours
            aux_decay_period: number of epochs for decaying auxillary loss
            initial_lr: initial learning rate
            weight_decay: weight decay for optimizer
            epsilon: numerical stability term
        Returns:
            trainer: a new MILDNetTrainer instance
        """
        config = UNETConfig()
        kwargs = {**config.to_dict(), **kwargs}  # 把设置和命令行参数全都放到 kwargs下
        super().__init__(model, **kwargs)
        # self.kwargs = kwargs

    def get_default_dataset(self, root_dir, train=True, proportion=1.0):
        if train:
            return SegmentationDataset(root_dir, target_size=self.kwargs.get('target_size'))

        return SegmentationDataset(root_dir, train=False,
                                   target_size=self.kwargs.get('target_size'))

    def get_default_optimizer(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.kwargs.get('lr'),
            betas=(0.9, 0.999),
            weight_decay=self.kwargs.get('weight_decay'), )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=10, factor=0.1, min_lr=1e-5, verbose=True)

        return optimizer, scheduler

    def preprocess(self, *data):
        return [datum.to(self.device) for datum in data]

    def compute_loss(self, pred, target, metrics=None):
        pred = torch.clamp(pred, min=1e-7, max=1-1e-7)
        return torch.mean(target.float() * -torch.log(pred))

    def postprocess(self, pred, target=None):
        pred = pred.round()[:, 1, ...].long()

        if target is not None:
            return pred, target.argmax(dim=1)

        return pred
    
    def post_epoch_hook(self, epoch):
        if self.scheduler is not None:
            loss = np.mean(self.tracker.history['loss'])

            self.scheduler.step(loss)

# if __name__ == '__main__':
#     model = UNET().cuda()
#     summary(model,(3,256,256))  # 找到8*偶数即可，比如288=8*36, 320
