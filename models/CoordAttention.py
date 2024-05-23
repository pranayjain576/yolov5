import torch
import torch.nn as nn
import torch.nn.functional as F

class CoordAttention(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAttention, self).__init__()
        self.avg_pool_h = nn.AvgPool2d(kernel_size=(1, reduction), stride=(1, reduction), padding=(0, reduction // 2))
        self.avg_pool_w = nn.AvgPool2d(kernel_size=(reduction, 1), stride=(reduction, 1), padding=(reduction // 2, 0))

        mip = max(8, inp // reduction) 

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.swish = lambda x: x * self.sigmoid(x)
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        identity = x
        
        n, c, h, w = x.size()
        
        # Manually perform average pooling along the height and width dimensions
        x_h = self.avg_pool_h(x)
        x_w = self.avg_pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2) 
        y = self.conv1(y) 
        y = self.bn1(y)  
        y = self.relu(y) 
        y = self.swish(y)
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2) 

        a_h = self.conv_h(x_h)
        a_h = self.sigmoid(a_h)
        a_w = self.conv_w(x_w)
        a_w = self.sigmoid(a_w)

        out = identity * a_w * a_h

        return out
