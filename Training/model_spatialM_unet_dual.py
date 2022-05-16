import torch
from torch import nn
import torch.nn.functional as F


class UNet_dual(nn.Module):
    def __init__(
        self,
        in_channels=1,
        n_classes=2,
        depth=5,
        wf=6,
        padding=False,
        batch_norm=False,
        up_mode='upconv',
    ):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597

        Using the default arguments will yield the exact version used
        in the original paper

        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(UNet_dual, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        x1_channels = 166
        self.down_path1 = nn.ModuleList()
        for i in range(depth):
            self.down_path1.append(
                UNetConvBlock(x1_channels, 2 ** (wf + i), padding, batch_norm)
            )
            x1_channels = 2 ** (wf + i)

        x2_channels = 3
        self.down_path2 = nn.ModuleList()
        for i in range(depth):
            self.down_path2.append(
                UNetConvBlock(x2_channels, 2 ** (wf + i), padding, batch_norm)
            )
            x2_channels = 2 ** (wf + i)

        up_channels = x1_channels + x2_channels
        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            cat_channels = 2 ** (wf + i) * 3
            # cat_channels = 2 ** (wf + i)
            self.up_path.append(
                UNetUpBlock(up_channels, cat_channels, 2 ** (wf + i), up_mode, padding, batch_norm)
            )
            up_channels = 2 ** (wf + i)

        self.last = nn.Sequential(
            nn.Conv2d(up_channels, n_classes, kernel_size=1),
            nn.Tanh())

    def forward(self, x):
        x1 = x[:, :-3, :, :]
        x2 = x[:, -3:, :, :]

        blocks1 = []
        for i, down in enumerate(self.down_path1):
            x1 = down(x1)
            if i != len(self.down_path1) - 1:
                blocks1.append(x1)
                x1 = F.max_pool2d(x1, 2)

        blocks2 = []
        for i, down in enumerate(self.down_path2):
            x2 = down(x2)
            if i != len(self.down_path2) - 1:
                blocks2.append(x2)
                x2 = F.max_pool2d(x2, 2)

        x12 = torch.cat([x1, x2], 1)
        for i, up in enumerate(self.up_path):
            x12 = up(x12, blocks1[-i - 1], blocks2[-i - 1])

        return 5*self.last(x12)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size_up, in_size_cat, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size_up, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_size_up, out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(in_size_cat, out_size, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])
        ]

    def forward(self, x, bridge1, bridge2):
        up = self.up(x)
        crop1 = self.center_crop(bridge1, up.shape[2:])
        crop2 = self.center_crop(bridge2, up.shape[2:])
        out = torch.cat([up, crop1, crop2], 1)
        # out = torch.cat([up], 1)
        out = self.conv_block(out)
        return out

