import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import msssim, ssim

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        # self.spatial = BasicConv(3, 1, (1, 1), stride=1, padding=0, relu=False)
        self.spatial = nn.Conv2d(3, 1, kernel_size=(3, 1), padding=(1, 0), stride=1, groups=1, bias=False)
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels),
            nn.ReLU(),
            nn.Linear(gate_channels, gate_channels)
            )
        self.last = nn.Sequential(
            nn.Sigmoid(),
            # nn.Softmax(dim=1)
        )
    def forward(self, x, x_feature):
        ssim_list = []
        for kk in range(x.shape[1]):
            x_ind = x[:, kk, :, :].unsqueeze(1)
            ssim_feature_ind_list = []
            for ii in range(x_feature.shape[1]):
                x_feature_ind = x_feature[:, ii, :, :].unsqueeze(1)
                ssim_ind = ssim(x_feature_ind, x_ind, size_average=None, val_range=10)
                ssim_ind = ssim_ind.unsqueeze(1)
                ssim_feature_ind_list.append(ssim_ind)

            ssim_feature_ind = torch.cat(ssim_feature_ind_list, dim=1)
            ssim_feature_ind = ssim_feature_ind.unsqueeze(2)
            ssim_list.append(ssim_feature_ind)

        ssim_info = torch.cat(ssim_list, dim=2).unsqueeze(3)
        h = F.relu(self.spatial(ssim_info))
        h = self.mlp(h)
        h = self.last(h)
        return h, ssim_info.squeeze(3)

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1), torch.min(x,1)[0].unsqueeze(1)), dim=1 )

def ChannelNorm(x, mask):
    x_out = None
    for ii in range(mask.shape[0]):
        x_ind = x[ii, :, :, :]
        mask_ind = mask[ii, :, :]

        idxs1 = torch.nonzero(mask_ind == 1)
        idxs0 = torch.nonzero(mask_ind == 0)

        x_mask = x_ind[:, idxs1[:, 0], idxs1[:, 1]]
        x_mask_mean = torch.mean(x_mask, 1)
        x_mask_std = torch.std(x_mask, 1)

        x_mask_mean = x_mask_mean.unsqueeze(1).unsqueeze(2).expand_as(x_ind)
        x_mask_std = x_mask_std.unsqueeze(1).unsqueeze(2).expand_as(x_ind)

        x_mask_norm = (x_ind - x_mask_mean) / x_mask_std
        x_mask_norm[:, idxs0[:, 0], idxs0[:, 1]] = 0
        x_mask_norm = x_mask_norm.unsqueeze(0)

        if x_out is None:
            x_out = x_mask_norm
        else:
            x_out = torch.cat((x_out, x_mask_norm), dim=0)

    return x_out


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        self.compress = ChannelPool()
    def forward(self, x, mask):
        x_compress = self.compress(x)
        x_out = ChannelNorm(x_compress, mask)
        return x_out

class CBAMmax(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(CBAMmax, self).__init__()
        self.SpatialGate = SpatialGate()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio)
    def forward(self, x, mask):

        x_feature = self.SpatialGate(x, mask)
        h, ssim_info = self.ChannelGate(x, x_feature)

        # h_final = h.clone()
        # h_max = torch.argmax(h, 1, keepdim=True)
        # h = torch.zeros_like(h)
        # h.scatter_(1, h_max, 1)
        #
        # h_repeat = h.unsqueeze(2).unsqueeze(3).expand_as(x)
        # x_final = torch.sum(torch.mul(h_repeat, x), 1, keepdim=True)
        return h, ssim_info