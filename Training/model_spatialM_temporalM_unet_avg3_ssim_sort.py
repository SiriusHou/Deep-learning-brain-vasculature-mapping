import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class ChannelAttnUNet(nn.Module):
    def __init__(self, spatial_channels=20, temporal_channels=25, n_classes=1, depth=5, wf=6, padding=False, batch_norm=False, up_mode='upconv'):
        """
        Args:
            spatial_channels (int): number of spatial inputs
            temporal_channels (int): number of temporal inputs
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
        super(ChannelAttnUNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = spatial_channels*temporal_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (wf + i), padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        self.last = nn.Sequential(
            nn.Conv2d(prev_channels, n_classes, kernel_size=1),
            nn.Tanh(),
        )

        self.Attn_block = ChannelAttnBlock(make_layers([50, 'M', 100, 'M'], in_channels=temporal_channels, batch_norm=True))

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x_unet_list = []
        h_list = []
        for i in range(x.size(0)): #x=batch*spatial_channels(20)*temporal_channels(25)*H(96)*W(112)
            h, x_output = self.Attn_block(x[i])
            x_output = torch.reshape(x_output, (-1, 1, x_output.size(2), x_output.size(3)))
            x_unet_list.append(x_output.squeeze(1).unsqueeze(0))
            h_list.append(h.unsqueeze(0))
        x_unet = torch.cat(x_unet_list, dim=0)
        h_final = torch.cat(h_list, dim=0)

        blocks = []
        for i, down in enumerate(self.down_path):
            x_unet = down(x_unet)
            if i != len(self.down_path) - 1:
                blocks.append(x_unet)
                x_unet = F.max_pool2d(x_unet, 2)

        for i, up in enumerate(self.up_path):
            x_unet = up(x_unet, blocks[-i - 1])

        return h_final, 5*self.last(x_unet)

class ChannelAttnBlock(nn.Module):
    '''
    Similar to VGG Architecture
    '''

    def __init__(self, features):
        super(ChannelAttnBlock, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))
        self.classifier = nn.Sequential(
            nn.Linear(100*3*3, 200),
            nn.ReLU(True),
            # nn.Dropout(),
            # nn.Linear(128, 128),
            # nn.ReLU(True),
            # nn.Dropout(),
            nn.Linear(200, 25),
            # nn.Sigmoid(),
            nn.Softmax(dim=1)
        )

        # Initialize weights
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = x.clone()
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        h = self.classifier(x)

        # without channel compression
        h_output = h.clone()
        h_repeat = h.unsqueeze(2).expand(*h.size(), y.size(2))
        h_repeat = h_repeat.unsqueeze(3).expand(*h_repeat.size(), y.size(3))
        y = torch.mul(h_repeat, y)

        # #with channel compression
        # h_output = h.clone()
        # h_max = torch.argmax(h, 1, keepdim=True)
        # h = torch.zeros_like(h)
        # h.scatter_(1, h_max, 1)
        #
        # h_repeat = h.unsqueeze(2).expand(*h.size(), y.size(2))
        # h_repeat = h_repeat.unsqueeze(3).expand(*h_repeat.size(), y.size(3))
        # y = torch.sum(torch.mul(h_repeat, y), 1, keepdim=True)

        return h_output, y

def make_layers(cfg, in_channels, batch_norm=False):
    layers = []
    in_channels_initial = in_channels
    layers += [nn.Conv2d(in_channels, 1000, kernel_size=7, padding=3, stride=2, groups=in_channels_initial), nn.ReLU(inplace=True)]
    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
    layers += [nn.Conv2d(1000, in_channels_initial, kernel_size=1, padding=0, stride=1, groups=in_channels_initial), nn.ReLU(inplace=True)]
    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

    for v in cfg:
        if v == 'M':
            # layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, stride=1, groups=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

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

        # initialise weights
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

        # initialise weights
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
               :, :, diff_y: (diff_y + target_size[0]), diff_x: (diff_x + target_size[1])
               ]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out