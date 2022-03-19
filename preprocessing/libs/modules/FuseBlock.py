import torch
import torch.nn as nn
# import torch.nn.functional as F


class MakeFB(nn.Module):
    def __init__(self, num_branches, num_blocks, num_channels, multi_scale_output=True):
        super(MakeFB, self).__init__()
        # self.BasicBlock = BasicBlock
        self.num_branches = num_branches
        self.num_blocks = num_blocks
        self.num_channels = num_channels
        self.multi_scale_output = multi_scale_output
        # self.branches = self._make_branches(BasicBlock, num_branches, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _make_one_branch(self, BasicBlock, branch_index, num_blocks, num_channels, stride=1):
        layers = []
        layers.append(
            BasicBlock(
                self.num_channels[branch_index],
                num_channels[branch_index],
                stride
            )
        )
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                BasicBlock(
                    self.num_channels[branch_index],
                    num_channels[branch_index]
                )
            )

        return nn.Sequential(*layers)

    def _make_branches(self, BasicBlock, num_branches, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(BasicBlock, i, num_blocks, num_channels)
            )

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        # if self.num_branches == 1:
        #     return None

        num_branches = self.num_branches
        num_inchannels = self.num_channels
        fuse_layers = []
        for i in range(num_branches):
            fuse_layer = []
            for j in range(num_branches):
                if j > i and j != 3:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                128,
                                1, 1, 0, bias=False
                            ),
                            nn.BatchNorm2d(128),
                            nn.Upsample(scale_factor=2 ** (j - i), mode='nearest')
                        )
                    )

                elif j > i and j == 3:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                128,
                                1, 1, 0, bias=False
                            ),
                            nn.BatchNorm2d(128),
                            nn.Upsample(scale_factor=2 ** (j - i - 1), mode='nearest')
                        )
                    )
                elif j == i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                128,
                                1, 1, 0, bias=False
                            ),
                            nn.BatchNorm2d(128))
                    )
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1 and i != 3:
                            # num_outchannels_conv3x3 = num_inchannels[i]
                            num_outchannels_conv3x3 = 128
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3)
                                )
                            )
                        elif k == i - j - 2 and i == 3:
                            # num_outchannels_conv3x3 = num_inchannels[i]
                            num_outchannels_conv3x3 = 128
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3)
                                )
                            )
                        elif k == i - j - 1 and i == 3 and j != 2:
                            pass
                        elif k == i - j - 1 and i == 3 and j == 2:
                            num_outchannels_conv3x3 = 128
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 1, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3)
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            # num_outchannels_conv3x3 = 96
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3),
                                    nn.ReLU(True)
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def forward(self, x):
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            # y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            y = self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse
