import torch
import torch.nn as nn
import torch.nn.functional as F


class CBR(nn.Module):
    """Normal Conv with ReLU activation"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, bias=False):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class FSPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = CBR(c1, c_, 1, 1)
        self.cv2 = CBR(c_ * 6, c2, 1, 1)
        self.conv = nn.Conv2d(c_, c_ * 2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.a = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.cv1(x)

        x_avg = self.a(x)
        x_avg = self.conv(x_avg)
        x_avg = F.interpolate(x_avg, size=x.size()[2:], mode='bilinear', align_corners=True)

        y1 = self.m(x)
        y2 = self.m(y1)

        return self.cv2(torch.cat([x, x_avg, y1, y2, self.m(y2)], dim=1))


