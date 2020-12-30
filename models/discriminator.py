import torch.nn as nn


def basic_list(in_channels,
               out_channels,
               stride=2,
               apply_batchnorm=True,
               leaky=True):
    module_list = [
        nn.Conv2d(in_channels,
                  out_channels,
                  kernel_size=4,
                  stride=stride,
                  padding=1)
    ]
    if apply_batchnorm:
        module_list.append(nn.BatchNorm2d(out_channels))
    if leaky:
        module_list.append(nn.LeakyReLU(0.2, inplace=True))
    else:
        module_list.append(nn.ReLU(inplace=True))
    return module_list


class Discriminator(nn.Module):
    def __init__(self, n_layers=3):
        super(Discriminator, self).__init__()

        self.increase_channels(first=True)
        blocks = basic_list(self.in_channels,
                            self.out_channels,
                            apply_batchnorm=False)

        for layer in range(n_layers - 2):
            self.increase_channels()
            blocks.extend(basic_list(self.in_channels, self.out_channels))

        # Предпоследний слой включает страйд = 1
        self.increase_channels()
        blocks.extend(basic_list(self.in_channels, self.out_channels,
                                 stride=1))

        # Добавим последний (классификационный) слой
        blocks.append(
            nn.Conv2d(self.out_channels, 1, kernel_size=4, stride=1,
                      padding=1))

        self.conv_net = nn.Sequential(*blocks)

    def increase_channels(self, first=False):
        if first:
            self.in_channels = 6
            self.out_channels = 64
            return
        self.in_channels = self.out_channels
        if self.out_channels < 512:
            self.out_channels *= 2
        return

    def forward(self, x):
        return self.conv_net(x)
