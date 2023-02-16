import torch
import torch.nn as nn
import numpy as np



class DepthWiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias):
        super(DepthWiseSeparableConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.bias = bias

        if self.bias:
            self.const = nn.Parameter(torch.ones(self.out_channels, 1) * np.log(1 / 0.07))
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=self.in_channels, out_channels=self.in_channels, groups=self.in_channels, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2, bias=False),
            nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        )
        
    
    def forward(self, x):
        if self.bias:
            return self.conv(x) + self.const
        return self.conv(x)