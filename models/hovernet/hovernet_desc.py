import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from net_utils import (DenseBlock, Net, ResidualBlock, TFSamepaddingLayer,
                        UpSample2x)
from utils import crop_op, crop_to_shape
from model_utils.summary import summary

####
class HoVerNet(Net):
    """Initialise 初始化 HoVer-Net."""

    def __init__(self, input_ch=3, nr_types=None, freeze=False, mode='original'):
        super().__init__()
        self.mode = mode
        self.freeze = freeze
        self.nr_types = nr_types
        self.output_ch = 3 if nr_types is None else 4

        assert mode == 'original' or mode == 'fast', \
                'Unknown mode `%s` for HoVerNet %s. Only support `original` or `fast`.' % mode

        module_list = [
            ("/", nn.Conv2d(input_ch, 64, 7, stride=1, padding=0, bias=False)),
            ("bn", nn.BatchNorm2d(64, eps=1e-5)),
            ("relu", nn.ReLU(inplace=True)),
        ]
        if mode == 'fast': # prepend the padding for `fast` mode
            module_list = [("pad", TFSamepaddingLayer(ksize=7, stride=1))] + module_list

        self.conv0 = nn.Sequential(OrderedDict(module_list))
        self.d0 = ResidualBlock(64, [1, 3, 1], [64, 64, 256], 3, stride=1)
        self.d1 = ResidualBlock(256, [1, 3, 1], [128, 128, 512], 4, stride=2)
        self.d2 = ResidualBlock(512, [1, 3, 1], [256, 256, 1024], 6, stride=2)
        self.d3 = ResidualBlock(1024, [1, 3, 1], [512, 512, 2048], 3, stride=2)

        self.conv_bot = nn.Conv2d(2048, 1024, 1, stride=1, padding=0, bias=False)

        def create_decoder_branch(out_ch=2, ksize=5):
            module_list = [ 
                ("conva", nn.Conv2d(1024, 256, ksize, stride=1, padding=0, bias=False)),
                ("dense", DenseBlock(256, [1, ksize], [128, 32], 8, split=4)),
                ("convf", nn.Conv2d(512, 512, 1, stride=1, padding=0, bias=False),),
            ]
            u3 = nn.Sequential(OrderedDict(module_list))

            module_list = [ 
                ("conva", nn.Conv2d(512, 128, ksize, stride=1, padding=0, bias=False)),
                ("dense", DenseBlock(128, [1, ksize], [128, 32], 4, split=4)),
                ("convf", nn.Conv2d(256, 256, 1, stride=1, padding=0, bias=False),),
            ]
            u2 = nn.Sequential(OrderedDict(module_list))

            module_list = [ 
                ("conva/pad", TFSamepaddingLayer(ksize=ksize, stride=1)),
                ("conva", nn.Conv2d(256, 64, ksize, stride=1, padding=0, bias=False),),
            ]
            u1 = nn.Sequential(OrderedDict(module_list))

            module_list = [ 
                ("bn", nn.BatchNorm2d(64, eps=1e-5)),
                ("relu", nn.ReLU(inplace=True)),
                ("conv", nn.Conv2d(64, out_ch, 1, stride=1, padding=0, bias=True),),
            ]
            u0 = nn.Sequential(OrderedDict(module_list))

            decoder = nn.Sequential(
                OrderedDict([("u3", u3), ("u2", u2), ("u1", u1), ("u0", u0),])
            )
            return decoder

        ksize = 5 if mode == 'original' else 3
        if nr_types is None:
            self.decoder = nn.ModuleDict(
                OrderedDict(
                    [
                        ("np", create_decoder_branch(ksize=ksize,out_ch=2)),
                        ("hv", create_decoder_branch(ksize=ksize,out_ch=2)),
                    ]
                )
            )
        else:
            self.decoder = nn.ModuleDict(
                OrderedDict(
                    [
                        ("tp", create_decoder_branch(ksize=ksize, out_ch=nr_types)),
                        ("np", create_decoder_branch(ksize=ksize, out_ch=2)),
                        ("hv", create_decoder_branch(ksize=ksize, out_ch=2)),
                    ]
                )
            )

        self.upsample2x = UpSample2x()
        # TODO: pytorch still require the channel eventhough its ignored
        self.weights_init()

    def forward(self, imgs):

        imgs = imgs / 255.0  # to 0-1 range to match XY

        if self.training:
            d0 = self.conv0(imgs)
            d0 = self.d0(d0, self.freeze)
            with torch.set_grad_enabled(not self.freeze):
                d1 = self.d1(d0)
                d2 = self.d2(d1)
                d3 = self.d3(d2)
            d3 = self.conv_bot(d3)
            d = [d0, d1, d2, d3]
        else:
            d0 = self.conv0(imgs)
            d0 = self.d0(d0)
            d1 = self.d1(d0)
            d2 = self.d2(d1)
            d3 = self.d3(d2)
            d3 = self.conv_bot(d3)
            d = [d0, d1, d2, d3]

        # TODO: switch to `crop_to_shape` ?
        if self.mode == 'original':
            d[0] = crop_op(d[0], [184, 184])
            d[1] = crop_op(d[1], [72, 72])
        else:
            d[0] = crop_op(d[0], [92, 92])
            d[1] = crop_op(d[1], [36, 36])

        out_dict = OrderedDict()
        # 循环遍历定义的分支 out_dict 存的是每个分支的输出结果
        for branch_name, branch_desc in self.decoder.items():
            u3 = self.upsample2x(d[-1]) + d[-2]    # d[-1]: torch.Size([1, 1024, 33, 33])  d[-2] :[1, 1024, 66, 66]
            u3 = branch_desc[0](u3)                # [1, 1024, 66, 66]   ->  torch.Size([1, 512, 30, 30])

            u2 = self.upsample2x(u3) + d[-3]       # d[-3] :[1, 1024, 60, 60]
            u2 = branch_desc[1](u2)                # [1, 512, 60, 60] ->  [1, 256, 40, 40]

            u1 = self.upsample2x(u2) + d[-4]       # d[-4]: [1, 256, 80, 80]
            u1 = branch_desc[2](u1)                # [1, 256, 80, 80] -> [1, 64, 80, 80]

            u0 = branch_desc[3](u1)                # u0: [1, 2, 80, 80]
            out_dict[branch_name] = u0

        return out_dict


####
def create_model(mode=None, **kwargs):
    if mode not in ['original', 'fast']:
        assert "Unknown Model Mode %s" % mode
    return HoVerNet(mode=mode, **kwargs)

if __name__ == '__main__':
    # create_model("original")

    model = create_model("original").cuda()
    # checkpoint = "G:/py_code/pycharm_Code/WESUP-TGCN/pretrained/checkpoint/hovernet_original_consep_type_tf2pytorch.tar"
    # model.load_state_dict(torch.load(checkpoint)['desc'],strict=False)

    summary( model, (3,270,270) )

    fm_channels_sum = 0
    np = model.decoder.np
    for name, layer in model.d0.named_modules():
        if isinstance(layer, torch.nn.Conv2d) and layer.out_channels>2:
            # layer.register_forward_hook(self._hook_fn)  # 对需要的层注册hook，后续该层执行完forword后去执行_hook_fn函数
            # setattr(self, f'side_conv{self.fm_channels_sum}',  # setattr(object, name, value) 函数指定对象的指定属性的值
            #         nn.Conv2d(layer.out_channels, layer.out_channels // 2,
            #                   1))  # 指定side_conv{int} = Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1))
            print("fm_channels_sum: ",fm_channels_sum,end=" ")
            fm_channels_sum += layer.out_channels // 2
            print("layer.out_channels:",layer.out_channels,"  ->  fm_channels:",fm_channels_sum)
    print(fm_channels_sum)

    for name, layer in model.d1.named_modules():
        if isinstance(layer, torch.nn.Conv2d) and layer.out_channels>2:
            # layer.register_forward_hook(self._hook_fn)  # 对需要的层注册hook，后续该层执行完forword后去执行_hook_fn函数
            # setattr(self, f'side_conv{self.fm_channels_sum}',  # setattr(object, name, value) 函数指定对象的指定属性的值
            #         nn.Conv2d(layer.out_channels, layer.out_channels // 2,
            #                   1))  # 指定side_conv{int} = Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1))
            print("fm_channels_sum: ",fm_channels_sum,end=" ")
            fm_channels_sum += layer.out_channels // 2
            print("layer.out_channels:",layer.out_channels,"  ->  fm_channels:",fm_channels_sum)
    print(fm_channels_sum)

    for name, layer in model.d2.named_modules():
        if isinstance(layer, torch.nn.Conv2d) and layer.out_channels > 2:
            # layer.register_forward_hook(self._hook_fn)  # 对需要的层注册hook，后续该层执行完forword后去执行_hook_fn函数
            # setattr(self, f'side_conv{self.fm_channels_sum}',  # setattr(object, name, value) 函数指定对象的指定属性的值
            #         nn.Conv2d(layer.out_channels, layer.out_channels // 2,
            #                   1))  # 指定side_conv{int} = Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1))
            print("fm_channels_sum: ", fm_channels_sum, end=" ")
            fm_channels_sum += layer.out_channels // 2
            print("layer.out_channels:", layer.out_channels, "  ->  fm_channels:", fm_channels_sum)
    print(fm_channels_sum)

    # for name, layer in model.d3.named_modules():
    #     if isinstance(layer, torch.nn.Conv2d) and layer.out_channels > 2:
    #         # layer.register_forward_hook(self._hook_fn)  # 对需要的层注册hook，后续该层执行完forword后去执行_hook_fn函数
    #         # setattr(self, f'side_conv{self.fm_channels_sum}',  # setattr(object, name, value) 函数指定对象的指定属性的值
    #         #         nn.Conv2d(layer.out_channels, layer.out_channels // 2,
    #         #                   1))  # 指定side_conv{int} = Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1))
    #         print("fm_channels_sum: ", fm_channels_sum, end=" ")
    #         fm_channels_sum += layer.out_channels // 2
    #         print("layer.out_channels:", layer.out_channels, "  ->  fm_channels:", fm_channels_sum)
    # print(fm_channels_sum)


    print("---------")

