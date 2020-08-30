import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import *

class BaseNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv_upsamp = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(),
            Interpolate(scale_factor=2, mode='bilinear')
        )
        
        self.conv_mid = nn.Conv2d(64, 3, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv_upsamp(x)
        x = self.conv_mid(x)
        x = self.sigmoid(x)
        return x

class MidLargeNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv_upsamp = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(),
            Interpolate(scale_factor=2, mode='bilinear')
        )
        
        self.conv_mid = nn.Conv2d(64, 3, 1)
        self.interp_large = Interpolate(scale_factor=2, mode='bilinear')
        self.conv_large = nn.Conv2d(64, 3, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv_upsamp(x)
        x_mid = self.conv_mid(x)
        x_mid = self.sigmoid(x_mid)
        x_large = self.interp_large(x)
        x_large = self.conv_large(x_large)
        x_large = self.sigmoid(x_large)
        return x_mid, x_large

class ResidualNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.first_block = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=1),
            Residual_Block(
                nn.Sequential(
                    nn.Conv2d(32, 32, kernel_size=3, padding=1),
                    nn.LeakyReLU(),
                    nn.Conv2d(32, 32, kernel_size=3, padding=1)
                )
            ),
            nn.LeakyReLU(),
            Residual_Block(
                nn.Sequential(
                    nn.Conv2d(32, 32, kernel_size=3, padding=1),
                    nn.LeakyReLU(),
                    nn.Conv2d(32, 32, kernel_size=3, padding=1)
                )
            ),
            nn.LeakyReLU(),
            Interpolate(scale_factor=2, mode='bilinear')
        )
        
        self.mid_block = nn.Conv2d(32, 3, kernel_size=1)
        
        self.large_block = nn.Sequential(
            Residual_Block(
                nn.Sequential(
                    nn.Conv2d(32, 32, kernel_size=3, padding=1),
                    nn.LeakyReLU(),
                    nn.Conv2d(32, 32, kernel_size=3, padding=1)
                )
            ),
            nn.LeakyReLU(),
            Interpolate(scale_factor=2, mode='bilinear'),
            nn.Conv2d(32, 3, kernel_size=1)
        )

        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.first_block(x)
        x_mid = self.mid_block(x)
        x_mid = self.sigmoid(x_mid)
        x_large = self.large_block(x)
        x_large = self.sigmoid(x_large)
        return x_mid, x_large

class DilatedNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.first_block = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=1),
            Dilated_Block(),
            Dilated_Block(),
            Interpolate(scale_factor=2, mode='bilinear')
        )
        
        self.mid_block = nn.Conv2d(32, 3, kernel_size=1)
        
        self.large_block = nn.Sequential(
            Dilated_Block(),
            Interpolate(scale_factor=2, mode='bilinear'),
            nn.Conv2d(32, 3, kernel_size=1)
        )

        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.first_block(x)
        x_mid = self.mid_block(x)
        x_mid = self.sigmoid(x_mid)
        x_large = self.large_block(x)
        x_large = self.sigmoid(x_large)
        return x_mid, x_large
    
class PreTrainedNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv_first = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU()
        )
        self.vgg19 = VGG19()
        self.interpolate = Interpolate(scale_factor=2, mode='bilinear')
        self.conv_mid = nn.Conv2d(128, 3, 1)
        self.conv_large = nn.Conv2d(128, 3, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x_f = self.vgg19(x)
        x = self.conv_first(x)
        x = torch.cat((x, x_f), dim=1)
        x = self.interpolate(x)
        x_mid = self.conv_mid(x)
        x_mid = self.sigmoid(x_mid)
        x_large = self.interpolate(x)
        x_large = self.conv_large(x_large)
        x_large = self.sigmoid(x_large)
        return x_mid, x_large