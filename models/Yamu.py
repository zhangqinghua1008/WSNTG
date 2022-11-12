import numpy as np
from utils_network.data import SegmentationDataset
from .base import BaseConfig, BaseTrainer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

# https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py

class YAMUConfig(BaseConfig):
    batch_size = 16

    lr = 4e-4  # 6e-4

    epochs = 400

    n_classes = 2  # Number of target classes.

    target_size = (256, 256)

    # Optimization parameters. 优化参数
    momentum = 0.9
    weight_decay = 0.0005


# https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class ConvRelu2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), bias=False):  # todo: bias?
        super(ConvRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-4)  # todo: eps setting?
        self.relu = nn.ReLU(inplace=True)


class Stem(nn.Module):
    """ a stem block in the initial encoder layer."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.setm_conv = nn.Sequential(
            nn.ZeroPad2d(3),
            nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2),
            nn.ReLU(inplace=True)
            # nn.MaxPool2d(2, stride=2)
        )

    def forward(self, x):
        x = self.setm_conv(x)
        return x


class DilatedConvBnRelu2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), bias=False,
                 dilation=2):  # todo: bias?
        super(DilatedConvBnRelu2d, self).__init__()
        # conv1 = nn.Conv2d(1, 1, 3, stride=1, bias=False, dilation=1)  # 普通卷积
        # dilation 膨胀率
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias,
                              dilation=dilation)
        self.relu = nn.ReLU(inplace=True)


class InceptionPlus(nn.Module):
    """Inception+ Model"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1x1 = ConvRelu2d(in_channels, in_channels, kernel_size=(1, 1), padding=(0, 0))
        self.conv3x3_1 = ConvRelu2d(in_channels, in_channels // 2, kernel_size=(3, 3), padding=(1, 1))
        self.conv3x3_2 = ConvRelu2d(in_channels // 2, in_channels // 2, kernel_size=(3, 3), padding=(1, 1))
        self.dilatedConv3x3_1 = DilatedConvBnRelu2d(in_channels, in_channels // 2, kernel_size=(3, 3), padding=(2, 2),
                                                    dilation=2)
        self.dilatedConv3x3_2 = DilatedConvBnRelu2d(in_channels // 2, in_channels // 2, kernel_size=(3, 3),
                                                    padding=(2, 2), dilation=2)

        self.filter = nn.Conv2d(in_channels * 3, out_channels, kernel_size=(1, 1), padding=(0, 0))
        self.relu = nn.ReLU(inplace=True)

    def _inception_forward(self, x):
        x1 = self.conv1x1(x)  # 最左侧1*1

        x3_1 = self.conv3x3_1(x)  # 第三列
        x4_1 = self.dilatedConv3x3_1(x)  # 第四列
        x_add_1 = torch.add(x3_1, x4_1)

        x3_2 = self.conv3x3_2(x_add_1)  # 第一个3*3
        x4_2 = self.dilatedConv3x3_2(x_add_1)
        x_add_2 = torch.add(x3_2, x4_2)

        return torch.cat((x1, x, x_add_1, x_add_2), dim=1)

    def forward(self, x):
        x_cat = self._inception_forward(x)
        return self.relu(self.filter(x_cat))


class InceptionDown(nn.Module):
    """InceptionDown """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            InceptionPlus(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class downInput(nn.Module):
    """InceptionDown """

    def __init__(self, in_channels, out_channels, kernel_size=7, stride=4, padding=3):
        super().__init__()
        self.downInput_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.downInput_conv(x)


class bottleneckLayer(nn.Module):
    """InceptionDown """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bottleneckLayer_conv = nn.Sequential(
            nn.MaxPool2d(2),
            # nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.bottleneckLayer_conv(x)


class Decoder(nn.Module):
    """Decoder Block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv3x3_1 = ConvRelu2d(in_channels, in_channels // 2, kernel_size=(3, 3), padding=(1, 1))
        self.conv3x3_2 = ConvRelu2d(in_channels // 2, in_channels // 2, kernel_size=(3, 3), padding=(1, 1))
        self.dilatedConv3x3_1 = DilatedConvBnRelu2d(in_channels, in_channels // 2, kernel_size=(3, 3), padding=(2, 2),
                                                    dilation=2)
        self.dilatedConv3x3_2 = DilatedConvBnRelu2d(in_channels // 2, in_channels // 2, kernel_size=(3, 3),
                                                    padding=(2, 2), dilation=2)

        self.filter = nn.Conv2d(in_channels * 2, out_channels, kernel_size=(1, 1))

    def _decoder_forward(self, x):
        x1_1 = self.conv3x3_1(x)  # 第1列
        x2_1 = self.dilatedConv3x3_1(x)  # 第2列
        x_add_1 = torch.add(x1_1, x2_1)

        x3_2 = self.conv3x3_2(x_add_1)  # 第1列 第2行
        x4_2 = self.dilatedConv3x3_2(x_add_1)
        x_add_2 = torch.add(x3_2, x4_2)

        return torch.cat((x, x_add_1, x_add_2), dim=1)

    def forward(self, x):
        x_cat = self._decoder_forward(x)
        return self.filter(x_cat)


class Up_Decoder(nn.Module):
    """ Upscaling + Decoder """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = Decoder(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class SSBlock(nn.Module):
    """MRcSE Block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.globalPool = nn.AdaptiveAvgPool2d((1, 1))
        # self.globalPool = torch.nn.functional.adaptive_avg_pool2d(a, (1, 1))

    def forward(self, x):
        x1 = self.globalPool(x)
        x1 = F.relu(x1)
        x1 = torch.sigmoid(x1)
        return x1


class MRcSE_I0(nn.Module):
    """MRcSE Block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.ss1 = SSBlock(in_channels, out_channels)

    def forward(self, x):
        xSS1 = self.ss1(x)
        x_mul = torch.mul(xSS1, x)
        return x_mul


class MRcSE_I1(nn.Module):
    """MRcSE_I1 Block"""
    def __init__(self, cI0, out_channels):
        super().__init__()
        self.conv1x1_0 = nn.Conv2d(cI0, out_channels, kernel_size=(1, 1))  # 对I0使用的1*1卷积
        self.ss0 = SSBlock(out_channels, out_channels)  # 第I0的SS层
        self.ss1 = SSBlock(out_channels, out_channels)  # 第I1的SS层

    def forward(self, x, xI0):
        xSS1 = self.ss1(x)

        xI0 = self.conv1x1_0(xI0)
        xSS0 = self.ss0(xI0)

        x_add = torch.add(xSS1,xSS0)
        x_mul = torch.mul(x_add, x)
        return x_mul


class MRcSE_I2(nn.Module):
    """MRcSE_I2 Block"""
    def __init__(self, cI0, cI1, out_channels):
        super().__init__()
        self.conv1x1_0 = nn.Conv2d(cI0, out_channels, kernel_size=(1, 1))  # 对I0使用的1*1卷积
        self.conv1x1_1 = nn.Conv2d(cI1, out_channels, kernel_size=(1, 1))  # 对I0使用的1*1卷积
        self.ss0 = SSBlock(out_channels, out_channels)  # 第I0的SS层
        self.ss1 = SSBlock(out_channels, out_channels)  # 第I1的SS层
        self.ss2 = SSBlock(out_channels, out_channels)  # 第I2的SS层

    def forward(self, x, xI0, xI1):
        xSS2 = self.ss2(x)

        xI0 = self.conv1x1_0(xI0)
        xSS0 = self.ss0(xI0)

        xI1 = self.conv1x1_1(xI1)
        xSS1 = self.ss1(xI1)

        x_add = torch.add(xSS2, torch.add(xSS1, xSS0))
        x_mul = torch.mul(x_add, x)
        return x_mul


class MRcSE_I3(nn.Module):
    """MRcSE_I3 Block"""
    def __init__(self, cI0, cI1, cI2, out_channels):
        super().__init__()
        self.conv1x1_0 = nn.Conv2d(cI0, out_channels, kernel_size=(1, 1))  # 对I0使用的1*1卷积
        self.conv1x1_1 = nn.Conv2d(cI1, out_channels, kernel_size=(1, 1))  # 对I1使用的1*1卷积
        self.conv1x1_2 = nn.Conv2d(cI2, out_channels, kernel_size=(1, 1))  # 对I2使用的1*1卷积
        self.ss0 = SSBlock(out_channels, out_channels)  # 第I0的SS层
        self.ss1 = SSBlock(out_channels, out_channels)  # 第I1的SS层
        self.ss2 = SSBlock(out_channels, out_channels)  # 第I2的SS层
        self.ss3 = SSBlock(out_channels, out_channels)  # 第I3的SS层

    def forward(self, x, xI0, xI1, xI2):
        xSS3 = self.ss1(x)

        xI0 = self.conv1x1_0(xI0)
        xSS0 = self.ss0(xI0)

        xI1 = self.conv1x1_1(xI1)
        xSS1 = self.ss1(xI1)

        xI2 = self.conv1x1_2(xI2)
        xSS2 = self.ss2(xI2)

        x_add = torch.add(xSS3, torch.add(xSS2, torch.add(xSS1, xSS0)))
        x_mul = torch.mul(x_add, x)
        return x_mul

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class YAMU(nn.Module):

    def __init__(self, n_classes=2, bilinear=False):
        super().__init__()
        self.input_channels = 3

        # MSc input
        self.MSc1 = downInput(self.input_channels, 16, kernel_size=7, stride=4, padding=3)
        self.MSc2 = downInput(self.input_channels, 16, kernel_size=15, stride=8, padding=6)
        self.MSc3 = downInput(self.input_channels, 16, kernel_size=15, stride=16, padding=6)

        # Encoder
        self.stemI3 = Stem(self.input_channels, 16)
        self.downI2 = InceptionDown(16+16, 32)
        self.downI1 = InceptionDown(32+16, 64)
        self.downI0 = InceptionDown(64+16, 128)

        self.footBottle = bottleneckLayer(128, 256)

        self.upD1 = Up_Decoder(256, 128)
        self.upD2 = Up_Decoder(128, 64)
        self.upD3 = Up_Decoder(64, 32)
        self.upD4 = Up_Decoder(32, 16)
        self.up5 = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)

        # MRcSE block
        self.mrcseI0 = MRcSE_I0(128, 128)
        self.mrcseI1 = MRcSE_I1(cI0=128, out_channels=64)
        self.mrcseI2 = MRcSE_I2(cI0=128, cI1=64, out_channels=32)
        self.mrcseI3 = MRcSE_I3(cI0=128, cI1=64, cI2=32, out_channels=16)

        self.outc = OutConv(16, n_classes)

    def forward(self, x):
        # x1 = self.inc(x)

        # MSc input
        xDown4 = self.MSc1(x)
        xDown8 = self.MSc2(x)
        xDown16 = self.MSc3(x)

        # ============ Encoder
        xI3 = self.stemI3(x)
        #
        xI2 = F.max_pool2d(xI3, 2)
        xI2 = torch.cat([xDown4, xI2], dim=1)
        xI2 = self.downI2(xI2)
        #
        xI1 = F.max_pool2d(xI2, 2)
        xI1 = torch.cat([xDown8, xI1], dim=1)
        xI1 = self.downI1(xI1)

        xI0 = F.max_pool2d(xI1, 2)
        xI0 = torch.cat([xDown16, xI0], dim=1)
        xI0 = self.downI0(xI0)

        xB = self.footBottle(xI0)  # footBottle 中含有maxpool了已经

        # =========== MRcSE
        xMRCI0 = self.mrcseI0(xI0)
        xMRCI1 = self.mrcseI1(xI1, xI0)
        xMRCI2 = self.mrcseI2(xI2, xI0, xI1)
        xMRCI3 = self.mrcseI3(xI3, xI0, xI1, xI2)

        # =========== Encoder
        xD1 = self.upD1(xB, xMRCI0)
        xD2 = self.upD2(xD1, xMRCI1)
        xD3 = self.upD3(xD2, xMRCI2)
        xD4 = self.upD4(xD3, xMRCI3)
        xD5 = self.up5(xD4)

        out = F.softmax(self.outc(xD5), dim=1)
        return out


class YAMUTrainer(BaseTrainer):
    """Trainer for YAMU."""

    def __init__(self, model, **kwargs):
        """Initialize a YAMU.
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
        config = YAMUConfig()
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
            optimizer, 'min', patience=5, factor=0.5, min_lr=1e-5, verbose=True)

        return optimizer, scheduler

    def preprocess(self, *data):
        return [datum.to(self.device) for datum in data]

    def compute_loss(self, pred, target, metrics=None):
        pred = torch.clamp(pred, min=1e-7, max=1 - 1e-7)
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


if __name__ == '__main__':
    model = YAMU().cpu()
    summary(model=model, input_size=(3, 256, 256), device='cpu')  # 找到8*偶数即可，比如288=8*36, 320
