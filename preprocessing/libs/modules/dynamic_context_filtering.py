import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BasicConv2d(nn.Module):
    def __init__(
        self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
    ):
        super(BasicConv2d, self).__init__()

        self.basicconv = nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.basicconv(x)


class AdaptiveConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(AdaptiveConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.dilation = (dilation, dilation)
        self.groups = groups
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, feature_in, dynamic_weight):
        # Get batch num
        batch_num = feature_in.size(0)

        # Reshape input tensor from size (N, C, H, W) to (1, N*C, H, W)
        feature_in = feature_in.view(1, -1, feature_in.size(2), feature_in.size(3))

        # Reshape dynamic_weight tensor from size (N, C*C, H, W) to (N*C, C, H, W)
        # dynamic_weight = dynamic_weight.view(-1, math.sqrt(dynamic_weight.size(1)),
        # dynamic_weight.size(2), dynamic_weight.size(3))

        # Do convolution
        dynamic_out = F.conv2d(feature_in, dynamic_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        # Reshape dynamic_out tensor from (1, N*C, H, W) to (N, C, H, W)
        dynamic_out = dynamic_out.view(batch_num, -1, dynamic_out.size(2), dynamic_out.size(3))

        return dynamic_out


class DCFMlayer(nn.Module):
    def __init__(self, k1, d1, channel):
        super(DCFMlayer, self).__init__()
        self.k1 = k1

        self.channel = channel
        self.query_conv = nn.Conv2d(in_channels=channel*2, out_channels=channel//2, kernel_size=1, padding=0, bias=True)
        self.key_conv = nn.Conv2d(in_channels=channel, out_channels=channel // 2, kernel_size=1, padding=0, bias=True)
        # self.key_future = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=channel, out_channels=channel//2, kernel_size=1, padding=0, bias=True)
        self.filter_conv = nn.Conv2d(in_channels=channel//2, out_channels=(channel // 2) * (channel // 2), kernel_size=1,
                                    padding=0, bias=True)

        self.encode_conv_k1 = nn.Conv2d(channel, channel//2, 1, padding=0, bias=True)
        # self.encoder_conv_k1 = nn.Conv2d(channel, (channel//2)*(channel//2), 1, padding=0, bias=True)
        # # self.encoder_conv_ln = nn.LayerNorm([channel//2, k1, k1])
        self.acf_conv = AdaptiveConv(self.channel//2, self.channel//2, kernel_size=self.k1, padding=d1, dilation=d1)
        # # self.bn = nn.BatchNorm2d(channel//2)
        # self.encoder_conv_k0 = nn.Conv2d(channel*2, channel, 1, padding=0)
        # self.conv_fusion = nn.Conv2d(channel, channel, 1, padding=0)
        self.pool_k1 = nn.AdaptiveAvgPool2d(k1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, feats_encoder, feats_encode):
        # N, C, H, W = N, C, 56, 56
        N, C, H, W = feats_encode.shape
        # [N * C*2 * H * W]
        # query
        feats_query = self.query_conv(feats_encoder).view(N, -1, H * W)
        # key
        feats_key = self.key_conv(feats_encode).view(N, -1, H * W).permute(0, 2, 1)
        # value
        feats_value = self.value_conv(feats_encode).view(N, -1, H * W)
        # affinity
        affinity = torch.matmul(feats_query, feats_key)
        affinity = torch.softmax(affinity, -1)
        # out
        out = torch.matmul(affinity, feats_value)+feats_query
        out = out.view(N, -1, H, W)
        out = self.filter_conv(out)
        # [N * C*2 * K * K]
        g_k1 = self.pool_k1(out)
        g_k1 = g_k1.view(self.channel//2, self.channel//2, self.k1, self.k1)
        # f_k1 = self.encode_conv_k1(feats_encode)
        f_k1 = feats_value.view(N, -1, H, W)

        dynamic_out = self.acf_conv(f_k1, g_k1)

        return dynamic_out


class Alpha(nn.Module):

    def __init__(self, in_channels, num_experts, dropout_rate):
        super(Alpha, self).__init__()

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(in_channels, num_experts)

    def forward(self, x):
        x = self.pool(x)
        x = torch.flatten(x)
        # x = self.dropout(x)
        x = self.fc(x)
        return x


# dynamic context-aware filtering module DCFM
class DCFM(nn.Module):
    def __init__(self, channel, k1, k2, k3, d1=1, d2=3, d3=5):
        super(DCFM, self).__init__()
        self.ACFM1 = DCFMlayer(k1, d1, channel)
        self.ACFM2 = DCFMlayer(k2, d2,  channel)
        self.ACFM3 = DCFMlayer(k3, d3,  channel)
        self.Alpha1 = Alpha(channel//2, 1, 0.1)
        self.Alpha2 = Alpha(channel//2, 1, 0.1)
        self.Alpha3 = Alpha(channel//2, 1, 0.1)
        self.relu = nn.ReLU(inplace=True)
        self.fire = nn.Sequential(
            nn.Conv2d(channel//2, channel, 3, padding=1, bias=True),
            nn.BatchNorm2d(channel)
            # nn.ReLU(inplace=True)
        )

    def forward(self, feats_encoder, feats_encode):
        N, C, H, W = feats_encode.shape
        acf1 = self.ACFM1(feats_encoder, feats_encode)
        acf2 = self.ACFM2(feats_encoder, feats_encode)
        acf3 = self.ACFM3(feats_encoder, feats_encode)

        out = []
        acf1_list = torch.split(acf1, 1, 0)
        acf2_list = torch.split(acf2, 1, 0)
        acf3_list = torch.split(acf3, 1, 0)
        for i in range(N):
            alpha1 = self.Alpha1(acf1_list[i])
            alpha2 = self.Alpha2(acf2_list[i])
            alpha3 = self.Alpha3(acf3_list[i])
            alpha = torch.cat([alpha1, alpha2, alpha3], dim=0)
            # alpha = torch.sigmoid(alpha)
            alpha = torch.softmax(alpha, dim=0)
            f_mdk = alpha[0]*acf1_list[i] + alpha[1]*acf2_list[i] + alpha[2]*acf3_list[i]
            out.append(self.fire(f_mdk))
        out = torch.cat(out, dim=0)
        # out = F.relu(feats_encode + out)
        out = F.relu(out)

        return out

