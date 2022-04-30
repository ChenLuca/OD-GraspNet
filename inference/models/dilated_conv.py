import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class DilatedConv(nn.Module):
    """
    A dilated convolution block
    """

    def __init__(self, in_channels, out_channels, dilation, stride=1, kernel_size=3, padding=1):
        super(DilatedConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)


    def forward(self, x_in):
        x = self.bn1(self.conv1(x_in))
        x = F.relu(x)
        return x

class DilatedNet(nn.Module):
    """
    A dilated convolution network
    """
    
    def __init__(self, in_channels, out_channels):
        super(DilatedNet, self).__init__()

        inner_channels = in_channels*4

        self.dc1 = DilatedConv(in_channels, inner_channels, dilation=5, stride=1, kernel_size=3, padding=5)
        self.dc2 = DilatedConv(inner_channels, inner_channels, dilation=2, stride=2, kernel_size=3, padding=1)
        self.dc3 = DilatedConv(inner_channels, out_channels, dilation=1, stride=2, kernel_size=3, padding=1)

    def forward(self, x_in):
        x = self.dc1(x_in)
        x = self.dc2(x)
        x = self.dc3(x)

        return x 

if __name__=="__main__":
    net = DilatedNet(10, 10).to("cuda")
    summary(net, (10, 224, 224))