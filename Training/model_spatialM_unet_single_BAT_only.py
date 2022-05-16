import torch
from torch import nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        n_classes=1,
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
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth

        x2_channels = 1
        self.down_path2 = nn.ModuleList()
        for i in range(depth):
            self.down_path2.append(
                UNetConvBlock(x2_channels, 2 ** (wf + i), padding, batch_norm)
            )
            x2_channels = 2 ** (wf + i)

        up_channels_c = x2_channels
        self.up_path_CVR = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            cat_channels = 2 ** (wf + i) * 2
            # cat_channels = 2 ** (wf + i)
            self.up_path_CVR.append(
                UNetUpBlock(up_channels_c, cat_channels, 2 ** (wf + i), up_mode, padding, batch_norm)
            )
            up_channels_c = 2 ** (wf + i)

        self.last_CVR = nn.Sequential(
            nn.Conv2d(up_channels_c, n_classes, kernel_size=1),
            nn.Tanh())

    def forward(self, x):
        x2 = x

        blocks2 = []
        for i, down in enumerate(self.down_path2):
            x2 = down(x2)
            if i != len(self.down_path2) - 1:
                blocks2.append(x2)
                x2 = F.max_pool2d(x2, 2)

        x_CVR = x2
        for i, up_c in enumerate(self.up_path_CVR):
            x_CVR = up_c(x_CVR, blocks2[-i - 1])

        CVR_out = 5 * self.last_CVR(x_CVR)

        return torch.cat([CVR_out], 1)


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

    def forward(self, x, bridge2):
        up = self.up(x)
        crop2 = self.center_crop(bridge2, up.shape[2:])
        out = torch.cat([up, crop2], 1)
        # out = torch.cat([up], 1)
        out = self.conv_block(out)
        return out

