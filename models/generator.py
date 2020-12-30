import torch
import torch.nn as nn


def conv_block(in_channels,
               out_channels,
               dropout=False,
               apply_batchnorm=True,
               bias=False):
    module_list = [
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(in_channels,
                  out_channels,
                  kernel_size=4,
                  stride=2,
                  padding=1,
                  bias=bias)
    ]
    if apply_batchnorm:
        module_list.append(nn.BatchNorm2d(out_channels))
    if dropout:
        module_list.append(nn.Dropout(p=0.1))
    return nn.Sequential(*module_list)


def conv_d_block(in_channels,
                 out_channels,
                 dropout=False,
                 apply_batchnorm=True,
                 bias=False):
    module_list = [
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(in_channels,
                           out_channels,
                           kernel_size=4,
                           stride=2,
                           padding=1,
                           bias=bias)
    ]
    if apply_batchnorm:
        module_list.append(nn.BatchNorm2d(out_channels))
    if dropout:
        module_list.append(nn.Dropout(p=0.1))
    return nn.Sequential(*module_list)


class Generator(nn.Module):
    def __init__(self, dataset):
        super(Generator, self).__init__()

        is_facades = (dataset == "facades")

        self.down_convolutions = nn.ModuleList([
            conv_block(3, 64, bias=is_facades),
            conv_block(64, 128, bias=is_facades),
            conv_block(128, 256, bias=is_facades),
            conv_block(256, 512, dropout=True, bias=is_facades),
            conv_block(512, 512, dropout=True, bias=is_facades),
            conv_block(512, 512, bias=is_facades),
        ])

        self.bottleneck = conv_block(512,
                                     512,
                                     apply_batchnorm=False,
                                     bias=is_facades)
        self.unbottleneck = conv_d_block(512, 512, bias=is_facades)

        self.up_convolutions = nn.ModuleList([
            conv_d_block(1024, 512, bias=is_facades),
            conv_d_block(1024, 512, bias=is_facades),
            conv_d_block(1024, 512, bias=is_facades),
            conv_d_block(768, 512, bias=is_facades),
            conv_d_block(640, 256, bias=is_facades),
            conv_d_block(320, 128, bias=is_facades),
        ])

        self.output_conv = nn.Sequential(
            nn.ReLU(), nn.Conv2d(128, 3, kernel_size=1, padding=0), nn.Tanh())

    def forward(self, x):
        down_outputs = []
        for i in range(len(self.down_convolutions)):
            down_block = self.down_convolutions[i]
            x = down_block(x)
            down_outputs.append(x)
        cell = self.bottleneck(x)
        cell = self.unbottleneck(cell)

        cell = x
        for i in range(len(down_outputs)):
            current = len(down_outputs) - i - 1
            current_output = down_outputs[current]
            cell = torch.cat([cell, current_output], dim=1)

            up_block = self.up_convolutions[i]
            cell = up_block(cell)

        output = self.output_conv(cell)
        return output
