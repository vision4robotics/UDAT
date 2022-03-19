from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from libs.modules.FuseBlock import MakeFB
from .resnet_dilation import resnet50, resnet101, Bottleneck, conv1x1

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class _ConvBatchNormReLU(nn.Sequential):
    def __init__(self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        relu=True,
    ):
        super(_ConvBatchNormReLU, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
        )
        self.add_module(
            "bn",
            nn.BatchNorm2d(out_channels),
        )

        if relu:
            self.add_module("relu", nn.ReLU(inplace=True))

    def forward(self, x):
        return super(_ConvBatchNormReLU, self).forward(x)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=(0, 0), dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class _DenseDecoder(nn.Module):
    def __init__(self, reduce_channel, n_classes):
        super(_DenseDecoder, self).__init__()

        # Decoder
        self.decoder = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", _ConvBatchNormReLU(128, 256, 3, 1, 1, 1)),    # 换成短连接残差块
                    ("conv2", nn.Conv2d(256, n_classes, kernel_size=1)),
                ]
            )
        )
        self.refine4_3 = _ConvBatchNormReLU(reduce_channel, reduce_channel, 3, 1, 1, 1)
        self.refine4_2 = _ConvBatchNormReLU(reduce_channel, reduce_channel, 3, 1, 1, 1)
        self.refine4_1 = _ConvBatchNormReLU(reduce_channel, reduce_channel, 3, 1, 1, 1)
        self.refine3_2 = _ConvBatchNormReLU(reduce_channel, reduce_channel, 3, 1, 1, 1)
        self.refine3_1 = _ConvBatchNormReLU(reduce_channel, reduce_channel, 3, 1, 1, 1)
        self.refine2_1 = _ConvBatchNormReLU(reduce_channel, reduce_channel, 3, 1, 1, 1)
        self.conv_cat_block4 = _ConvBatchNormReLU(reduce_channel, reduce_channel, 3, 1, 1, 1)
        self.conv_cat_block3 = _ConvBatchNormReLU(reduce_channel, reduce_channel, 3, 1, 1, 1)
        self.conv_cat_block2 = _ConvBatchNormReLU(reduce_channel, reduce_channel, 3, 1, 1, 1)
        self.conv_cat_block1 = _ConvBatchNormReLU(reduce_channel, reduce_channel, 3, 1, 1, 1)
        self.fuse_sal = _ConvBatchNormReLU(reduce_channel * 4, 128, 3, 1, 1, 1)

    def seg_conv(self, block3, block4):

        bu1 = block3 + self.refine4_3(block4)
        # bu1 = F.interpolate(x, size=block2.shape[2:], mode="bilinear", align_corners=False)
        return bu1

    def seg_conv2(self, block2, block4, bu1):

        block4 = F.interpolate(block4, size=block2.shape[2:], mode="bilinear", align_corners=True)
        bu1 = F.interpolate(bu1, size=block2.shape[2:], mode="bilinear", align_corners=True)
        bu2 = block2 + self.refine3_2(bu1) + self.refine4_2(block4)
        # bu2 = F.interpolate(x, size=block1.shape[2:], mode="bilinear", align_corners=False)
        return bu2

    def seg_conv3(self, block1, block4, bu1, bu2):

        # bu1_2 = F.interpolate(bu1, size=block1.shape[2:], mode="bilinear", align_corners=False)
        block4_1 = F.interpolate(block4, size=block1.shape[2:], mode="bilinear", align_corners=True)
        bu2_1 = F.interpolate(bu2, size=block1.shape[2:], mode="bilinear", align_corners=True)
        bu1_1 = F.interpolate(bu1, size=block1.shape[2:], mode="bilinear", align_corners=True)
        # x = torch.cat((block1, bu2), dim=1)

        bu3 = block1 + self.refine2_1(bu2_1) + self.refine3_1(bu1_1) + self.refine4_1(block4_1)
        return bu3, block4_1, bu2_1, bu1_1

    def segment(self, bu3, block4_1, bu2_1, bu1_1, shape):
        agg = torch.cat((self.conv_cat_block1(bu3), self.conv_cat_block2(bu2_1), self.conv_cat_block3(bu1_1),
                         self.conv_cat_block4(block4_1)), dim=1)
        sal = self.fuse_sal(agg)
        sal = self.decoder(sal)
        sal = F.interpolate(sal, size=shape, mode="bilinear", align_corners=True)
        # sal= self.decoder(sal)
        return sal

    def forward(self, block1, block2, block3, block4, x):
        bu1 = self.seg_conv(block3, block4)
        bu2 = self.seg_conv2(block2, block4, bu1)
        bu3, block4_1, bu2_1, bu1_1 = self.seg_conv3(block1, block4, bu1, bu2)
        seg = self.segment(bu3, block4_1, bu2_1, bu1_1, x.shape[2:])
        # return seg, E_sup, E_att, bu1_res
        return seg


class _ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling with image pool"""

    def __init__(self, in_channels, out_channels, output_stride):
        super(_ASPPModule, self).__init__()
        if output_stride == 8:
            pyramids = [12, 24, 36]
        elif output_stride == 16:
            pyramids = [6, 12, 18]
        self.stages = nn.Module()
        self.stages.add_module(
            "c0", _ConvBatchNormReLU(in_channels, out_channels, 1, 1, 0, 1)
        )
        for i, (dilation, padding) in enumerate(zip(pyramids, pyramids)):
            self.stages.add_module(
                "c{}".format(i + 1),
                _ConvBatchNormReLU(in_channels, out_channels, 3, 1, padding, dilation),
            )
        self.imagepool = nn.Sequential(
            OrderedDict(
                [
                    ("pool", nn.AdaptiveAvgPool2d((1,1))),
                    ("conv", _ConvBatchNormReLU(in_channels, out_channels, 1, 1, 0, 1)),
                ]
            )
        )
        self.fire = nn.Sequential(
            OrderedDict(
                [
                    ("conv", _ConvBatchNormReLU(out_channels * 5, out_channels, 3, 1, 1, 1)),
                    ("dropout", nn.Dropout2d(0.1))
                ]
            )
        )

    def forward(self, x):
        h = self.imagepool(x)
        h = [F.interpolate(h, size=x.shape[2:], mode="bilinear", align_corners=False)]
        for stage in self.stages.children():
            h += [stage(x)]
        h = torch.cat(h, dim=1)
        h = self.fire(h)
        return h


class DCFNet_backbone(nn.Module):

    def __init__(self, cfg, output_stride, input_channels=3, pretrained=False):
        super(DCFNet_backbone, self).__init__()

        self.os = output_stride
        self.resnet = resnet101(pretrained=pretrained, output_stride=output_stride, input_channels=input_channels)
        self.aspp = _ASPPModule(2048, 256, output_stride)
        self.DenseDecoder = _DenseDecoder(reduce_channel=128, n_classes=1)

        # stage3----> stage4
        self.stage4_cfg = cfg['stage4_cfg']
        self.stage4 = self._make_stage(self.stage4_cfg)

        if pretrained:
            for key in self.state_dict():
                if 'resnet' not in key:
                    self.init_layer(key)

    def init_layer(self, key):
        if key.split('.')[-1] == 'weight':
            if 'conv' in key:
                if self.state_dict()[key].ndimension() >= 2:
                    nn.init.kaiming_normal_(self.state_dict()[key], mode='fan_out', nonlinearity='relu')
            elif 'bn' in key:
                self.state_dict()[key][...] = 1
        elif key.split('.')[-1] == 'bias':
            self.state_dict()[key][...] = 0.001

    def feat_conv(self, x):

        x_list = []
        block0 = self.resnet.conv1(x)
        block0 = self.resnet.bn1(block0)
        block0 = self.resnet.relu(block0)
        block0 = self.resnet.maxpool(block0)

        block1 = self.resnet.layer1(block0)
        x_list.append(block1)
        block2 = self.resnet.layer2(block1)
        x_list.append(block2)
        block3 = self.resnet.layer3(block2)
        x_list.append(block3)
        block4 = self.resnet.layer4(block3)
        # if self.os == 16:
        #     block4 = F.upsample(block4, scale_factor=2, mode='bilinear', align_corners=False)
        block4 = self.aspp(block4)
        x_list.append(block4)
        return block1, block2, block3, block4, x_list

    def _make_stage(self, layer_config, multi_scale_output=True):
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        modules = []
        modules.append(
            MakeFB(
                num_branches,
                num_blocks,
                num_channels,
                multi_scale_output
            )
        )
        return nn.Sequential(*modules)

    def forward(self, x):
        block1, block2, block3, block4, x_list = self.feat_conv(x)

        y_list = self.stage4(x_list)

        seg = self.DenseDecoder(y_list[0], y_list[1], y_list[2], y_list[3], x)

        return F.sigmoid(seg)
