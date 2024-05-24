import torch
import torch.nn as nn
import torch.nn.functional as F

# class CoordAttention(nn.Module):
#     def __init__(self, inp, oup, reduction=32):
#         super(CoordAttention, self).__init__()
#         self.avg_pool_h = nn.AvgPool2d(kernel_size=(1, reduction), stride=(1, reduction), padding=(0, reduction // 2))
#         self.avg_pool_w = nn.AvgPool2d(kernel_size=(reduction, 1), stride=(reduction, 1), padding=(reduction // 2, 0))

#         mip = max(8, inp // reduction) 

#         self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
#         self.bn1 = nn.BatchNorm2d(mip)
#         self.relu = nn.ReLU(inplace=True)
#         self.sigmoid = nn.Sigmoid()
        
#         self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
#         self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        
#     def forward(self, x):
#         identity = x
        
#         n, c, h, w = x.size()
        
#         # Manually perform average pooling along the height and width dimensions
#         x_h = self.avg_pool_h(x)
#         x_w = self.avg_pool_w(x).permute(0, 1, 3, 2)

#         y = torch.cat([x_h, x_w], dim=2) 
#         y = self.conv1(y) 
#         y = self.bn1(y)  
#         y = self.relu(y) 
#         y = y * self.sigmoid(y)
        
#         x_h, x_w = torch.split(y, [h, w], dim=2)
#         x_w = x_w.permute(0, 1, 3, 2) 

#         a_h = self.conv_h(x_h)
#         a_h = self.sigmoid(a_h)
#         a_w = self.conv_w(x_w)
#         a_w = self.sigmoid(a_w)

#         out = identity * a_w * a_h

#         return out

# class CoordAttention(nn.Module):
#     def __init__(self, in_channels, reduction=32):
#         super(CoordAttention, self).__init__()
#         self.in_channels = in_channels
#         self.reduction = reduction
#         self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
#         self.pool_w = nn.AdaptiveAvgPool2d((1, None))

#         self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
#         self.bn1 = nn.BatchNorm2d(in_channels // reduction)
#         self.relu = nn.ReLU()

#         self.conv_h = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
#         self.conv_w = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         # x: (C, H, W)
#         identity = x
#         batch_size, channels, height, width = x.size()

#         # Coordinate Attention: Horizontal and Vertical Pooling
#         x_h = self.pool_h(x).permute(0, 1, 3, 2)  # (B, C, W, 1)
#         x_w = self.pool_w(x)  # (B, C, 1, H)
        
#         # Concatenate
#         y = torch.cat([x_h, x_w], dim=2)  # (B, C, W+1, H)

#         # Convolution + BatchNorm + ReLU
#         y = self.relu(self.bn1(self.conv1(y)))  # (B, C//r, W+1, H)

#         # Split
#         x_h, x_w = y.split([width, height], dim=2)
        
#         # Transformations
#         x_h = x_h.permute(0, 1, 3, 2)  # (B, C//r, 1, W)
#         x_w = x_w  # (B, C//r, 1, H)

#         # Generate attention maps
#         a_h = self.sigmoid(self.conv_h(x_h))  # (B, C, 1, W)
#         a_w = self.sigmoid(self.conv_w(x_w).permute(0, 1, 3, 2))  # (B, C, H, 1)

#         # Apply attention maps
#         out = identity * a_w * a_h  # (B, C, H, W)
        
#         return out

import torch
import torch.nn as nn
import torch.nn.functional as F

class CoordAttention(nn.Module):
    def __init__(self, in_channels, reduction=32):
        super(CoordAttention, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction

        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(in_channels // reduction)
        self.relu = nn.ReLU()

        self.conv_h = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
        self.conv_w = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def custom_avg_pool(self, x):
        # Perform average pooling along the vertical dimension (1 x H)
        avg_pool_vertical = torch.mean(x, dim=3, keepdim=True)  # C x H x 1

        # Perform average pooling along the horizontal dimension (W x 1)
        avg_pool_horizontal = torch.mean(x, dim=2, keepdim=True)  # C x 1 x W

        return avg_pool_vertical, avg_pool_horizontal

    def forward(self, x):
        # x: (B, C, H, W)
        identity = x
        batch_size, channels, height, width = x.size()

        # Custom average pooling along vertical and horizontal dimensions
        x_h, x_w = self.custom_avg_pool(x)
        
        # Convolution + BatchNorm + ReLU
        y = self.relu(self.bn1(self.conv1(torch.cat([x_h, x_w], dim=2))))  # (B, C//r, H, W)

        # Split
        split_size_h = channels // 2
        split_size_w = channels - split_size_h
        x_h, x_w = torch.split(y, [split_size_h, split_size_w], dim=1)
        
        # Transformations
        x_h = x_h.permute(0, 1, 3, 2)  # (B, C//r, W, 1)
        x_w = x_w.permute(0, 1, 3, 2)  # (B, C//r, H, 1)

        # Generate attention maps
        a_h = self.sigmoid(self.conv_h(x_h))  # (B, C, W, 1)
        a_w = self.sigmoid(self.conv_w(x_w))  # (B, C, H, 1)

        # Apply attention maps
        out = identity * a_w * a_h  # (B, C, H, W)
        
        return out

