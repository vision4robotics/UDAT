from libs.networks.DCFNet_bk import DCFNet_backbone
from libs.modules.dynamic_context_filtering import DCFM

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, groups=1, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, groups=groups)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ImageModel(nn.Module):
    def __init__(self, cfg, pretrained=False):
        super(ImageModel, self).__init__()
        self.backbone = DCFNet_backbone(
            cfg=cfg,
            output_stride=32,
            pretrained=pretrained
        )

    def forward(self, frame):
        seg = self.backbone(frame)
        return seg

    def freeze_bn(self):
        for m in self.backbone.named_modules():
            if isinstance(m[1], nn.BatchNorm2d):
                m[1].eval()


class GateWeightGenerator(nn.Module):

    def __init__(self, in_channels, num_experts):
        super(GateWeightGenerator, self).__init__()

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(in_channels, num_experts)

    def forward(self, x):
        x = self.pool(x)
        x = torch.flatten(x)
        # x = self.dropout(x)
        x = self.fc(x)
        return x


class DCFMRear(nn.Module):
    def __init__(self, channel=128, add=True, k1=3, k2=3, k3=3, d1=1, d2=3, d3=5):
        super(DCFMRear, self).__init__()
        self.MDK_front = DCFM(channel=channel//2, k1=k1, k2=k2, k3=k3, d1=d1, d2=d2, d3=d3)
        self.MDK_rear = DCFM(channel=channel//2, k1=k1, k2=k2, k3=k3, d1=d1, d2=d2, d3=d3)
        # self.bn = nn.BatchNorm2d(channel)
        self.add = add
        self.conva = nn.Conv2d(channel, channel//2, 1, padding=0, bias=False)
        self.convc = nn.Conv2d(channel, channel//2, 1, padding=0, bias=False)
        self.conv1 = nn.Conv2d(channel//2, channel, 1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(channel//2, channel, 1, padding=0, bias=False)

        self.Alpha = GateWeightGenerator(channel, 1)

        self.MDK_fire = nn.Sequential(
            nn.Conv2d(in_channels=channel * 2, out_channels=channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channel)
        )

    def forward(self, feats_encoder_front, feats_encode, feats_encoder_rear):
        feats_encode1 = self.conva(feats_encode)
        y_front = self.MDK_front(feats_encoder_front, feats_encode1)
        y_front = self.conv1(y_front)

        feats_encode2 = self.convc(feats_encode)
        y_rear = self.MDK_rear(feats_encoder_rear, feats_encode2)
        y_rear = self.conv2(y_rear)

        dynamic_output = self.MDK_fire(torch.cat((y_front, y_rear), dim=1))
        if self.add:
            alpha = self.Alpha(dynamic_output)
            dynamic_output = alpha * dynamic_output + (1-alpha) * feats_encode
        return dynamic_output


class VideoModel(nn.Module):

    def __init__(self, output_stride=16, pretrained=True, cfg=None):
        super(VideoModel, self).__init__()
        self.backbone = DCFNet_backbone(
            cfg=cfg,
            output_stride=output_stride,
            pretrained=pretrained
        )

        self.MDK_module_R3 = DCFMRear(channel=128, k1=3, k2=3, k3=3, d1=1, d2=3, d3=5)
        self.MDK_module_R2 = DCFMRear(channel=128, k1=3, k2=3, k3=3, d1=1, d2=3, d3=5)
        self.MDK_module_R1 = DCFMRear(channel=128, k1=3, k2=3, k3=3, d1=1, d2=3, d3=5)
        self.MDK_module_R0 = DCFMRear(channel=128, k1=3, k2=3, k3=3, d1=1, d2=3, d3=5)

        # self.freeze_bn()
        if pretrained:
            for key in self.state_dict():
                if 'backbone' not in key:
                    self.video_init_layer(key)
        else:
            for key in self.state_dict():
                self.video_init_layer(key)

    def video_init_layer(self, key):
        if key.split('.')[-1] == 'weight':
            if 'conv' in key:
                if self.state_dict()[key].ndimension() >= 2:
                    nn.init.kaiming_normal_(self.state_dict()[key], mode='fan_out', nonlinearity='relu')
            elif 'bn' in key:
                self.state_dict()[key][...] = 1
        elif key.split('.')[-1] == 'bias':
            self.state_dict()[key][...] = 0.001

    def freeze_bn(self):
        for m in self.backbone.named_modules():
            if isinstance(m[1], nn.BatchNorm2d):
                m[1].eval()

    def forward(self, clip):

        clip_feats = [self.backbone.feat_conv(frame) for frame in clip]

        y_list = [self.backbone.stage4(clip_feats[p]) for p in range(4)]

        premask_block1 = []
        premask_block2 = []
        premask_block3 = []
        premask_block4 = []

        i = 0
        while i < 4:
            if i == 0:
                feats_encoder_front = y_list[0][3]
                feats_input = y_list[0][3]
                feats_encoder_rear = y_list[1][3]

            elif i == 1:
                feats_encoder_front = y_list[0][3]
                feats_input = y_list[1][3]
                feats_encoder_rear = y_list[2][3]
                premask_block4.append(saliency_feat_res)

            elif i == 2:
                feats_encoder_front = y_list[1][3]
                feats_input = y_list[2][3]
                feats_encoder_rear = y_list[3][3]
                premask_block4.append(saliency_feat_res)

            elif i == 3:
                feats_encoder_front = y_list[2][3]
                feats_input = y_list[3][3]
                feats_encoder_rear = y_list[3][3]
                premask_block4.append(saliency_feat_res)

            saliency_feat_res = self.MDK_module_R0(feats_encoder_front, feats_input, feats_encoder_rear)
            i = i + 1
        premask_block4.append(saliency_feat_res)

        feats_encode_block3 = [self.backbone.DenseDecoder.seg_conv(y_list[k][2], premask_block4[k])
                               for k in range(4)]

        i = 0
        while i < 4:
            if i == 0:
                feats_encoder_front = feats_encode_block3[0]
                feats_input = feats_encode_block3[0]
                feats_encoder_rear = feats_encode_block3[1]

            elif i == 1:
                feats_encoder_front = feats_encode_block3[0]
                feats_input = feats_encode_block3[1]
                feats_encoder_rear = feats_encode_block3[2]
                premask_block3.append(saliency_feat_res)

            elif i == 2:
                feats_encoder_front = feats_encode_block3[1]
                feats_input = feats_encode_block3[2]
                feats_encoder_rear = feats_encode_block3[3]
                premask_block3.append(saliency_feat_res)

            elif i == 3:
                feats_encoder_front = feats_encode_block3[2]
                feats_input = feats_encode_block3[3]
                feats_encoder_rear = feats_encode_block3[3]
                premask_block3.append(saliency_feat_res)

            saliency_feat_res = self.MDK_module_R1(feats_encoder_front, feats_input, feats_encoder_rear)
            i = i + 1
        premask_block3.append(saliency_feat_res)  

        feats_encode_block2 = [self.backbone.DenseDecoder.seg_conv2(y_list[k][1], premask_block4[k], premask_block3[k])
                               for k in range(4)]

        i = 0
        while i < 4:
            if i == 0:
                feats_encoder_front = feats_encode_block2[0]
                feats_input = feats_encode_block2[0]
                feats_encoder_rear = feats_encode_block2[1]

            elif i == 1:
                feats_encoder_front = feats_encode_block2[0]
                feats_input = feats_encode_block2[1]
                feats_encoder_rear = feats_encode_block2[2]
                premask_block2.append(saliency_feat_res)

            elif i == 2:
                feats_encoder_front = feats_encode_block2[1]
                feats_input = feats_encode_block2[2]
                feats_encoder_rear = feats_encode_block2[3]
                premask_block2.append(saliency_feat_res)

            elif i == 3:
                feats_encoder_front = feats_encode_block2[2]
                feats_input = feats_encode_block2[3]
                feats_encoder_rear = feats_encode_block2[3]
                premask_block2.append(saliency_feat_res)

            saliency_feat_res = self.MDK_module_R2(feats_encoder_front, feats_input, feats_encoder_rear)
            i = i + 1
        premask_block2.append(saliency_feat_res)

        feats_encode_block1s = [
            self.backbone.DenseDecoder.seg_conv3(y_list[k][0], premask_block4[k], premask_block3[k], premask_block2[k])
            for k in range(4)]

        i = 0
        while i < 4:
            if i == 0:
                feats_encoder_front = feats_encode_block1s[0][0]
                feats_input = feats_encode_block1s[0][0]
                feats_encoder_rear = feats_encode_block1s[1][0]

            elif i == 1:
                feats_encoder_front = feats_encode_block1s[0][0]
                feats_input = feats_encode_block1s[1][0]
                feats_encoder_rear = feats_encode_block1s[2][0]
                premask_block1.append(saliency_feat_res)

            elif i == 2:
                feats_encoder_front = feats_encode_block1s[1][0]
                feats_input = feats_encode_block1s[2][0]
                feats_encoder_rear = feats_encode_block1s[3][0]
                premask_block1.append(saliency_feat_res)

            elif i == 3:
                feats_encoder_front = feats_encode_block1s[2][0]
                feats_input = feats_encode_block1s[3][0]
                feats_encoder_rear = feats_encode_block1s[3][0]
                premask_block1.append(saliency_feat_res)

            saliency_feat_res = self.MDK_module_R3(feats_encoder_front, feats_input, feats_encoder_rear)
            i = i + 1
        premask_block1.append(saliency_feat_res)

        preds = []
        for i, frame in enumerate(clip):
            seg = self.backbone.DenseDecoder.segment(premask_block1[i], feats_encode_block1s[i][1],
                                                     feats_encode_block1s[i][2], feats_encode_block1s[i][3],
                                                     frame.shape[2:])
            preds.append(torch.sigmoid(seg))
        return preds



