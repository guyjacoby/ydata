import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
from torchvision.transforms import Normalize

class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super().__init__()
        self.interpolate = F.interpolate
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        
    def forward(self, x):
        x = self.interpolate(x, size=self.size, scale_factor=self.scale_factor, 
                             mode=self.mode, align_corners=self.align_corners)
        return x

class Residual_Block(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        
    def forward(self, x):
        return self.module(x) + x

class Dilated_Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.dilate_one = nn.Conv2d(32, 32, kernel_size=3, padding=1, dilation=1)
        self.dilate_two = nn.Conv2d(32, 32, kernel_size=3, padding=2, dilation=2)
        self.dilate_four = nn.Conv2d(32, 32, kernel_size=3, padding=4, dilation=4)
        self.conv = nn.Conv2d(96, 32, kernel_size=3, padding=1)
        
    def forward(self, x):
        x1 = self.dilate_one(x)
        x1 = F.leaky_relu(x1)
        x2 = self.dilate_two(x)
        x2 = F.leaky_relu(x2)
        x3 = self.dilate_four(x)
        x3 = F.leaky_relu(x3)
        
        x = torch.cat((x1, x2, x3), dim=1)
        x = F.leaky_relu(x)
        x = self.conv(x)
        x = F.leaky_relu(x)
        return x
    
class VGG19(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model = vgg19(pretrained=True).features[:4]
        
        for p in self.model.parameters():
            p.requires_grad = False
        
        self.transform = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def forward(self, x):
        x = torch.cat([self.transform(x[i]).unsqueeze(0) for i in range(x.size(0))])
        return self.model(x)