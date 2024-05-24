import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F


class CoordAttention(nn.Module):
    def __init__(self, in_channels, reduction=32):
        super(CoordAttention, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(in_channels // reduction)
        self.relu = nn.ReLU()

        self.conv_h = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
        self.conv_w = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (C, H, W)
        identity = x
        batch_size, channels, height, width = x.size()

        # Coordinate Attention: Horizontal and Vertical Pooling
        x_h = self.pool_h(x).permute(0, 1, 3, 2)  # (B, C, W, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2) # (B, C, 1, H)
        
        # Concatenate
        y = torch.cat([x_h, x_w], dim=2)  # (B, C, W+1, H)

        # Convolution + BatchNorm + ReLU
        y = self.relu(self.bn1(self.conv1(y)))  # (B, C//r, W+1, H)

        # Split
        x_h, x_w = y.split([width, height], dim=2)
        
        # Transformations
        x_h = x_h.permute(0, 1, 3, 2)  # (B, C//r, 1, W)
        x_w = x_w  # (B, C//r, 1, H)

        # Generate attention maps
        a_h = self.sigmoid(self.conv_h(x_h))  # (B, C, 1, W)
        a_w = self.sigmoid(self.conv_w(x_w))  # (B, C, H, 1)

        # Apply attention maps
        out = identity * a_w * a_h  # (B, C, H, W)
        
        return out




