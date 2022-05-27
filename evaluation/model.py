from torch import nn
import torch
from torchvision.models import mobilenet_v2


class CDomain(nn.Module):
    '''Binary classification: whether an image is in a domain'''
    def __init__(self):
        super().__init__()
        self.mobilenet = mobilenet_v2(pretrained=True) # 3.5M params
        self.classifier = nn.Linear(1000, 2)
        
    def forward(self, x):
        x = self.mobilenet(x)
        x = self.classifier(x)
        return x


class CPair(nn.Module):
    '''Binary classification: whether an image is in a domain'''
    def __init__(self):
        super().__init__()
        self.pre = nn.Conv2d(6, 3, kernel_size=1, stride=1, padding=0)
        self.mobilenet = mobilenet_v2(pretrained=True) # 3.5M params
        # self.mobilenet_b = mobilenet_v2(pretrained=False) # 3.5M params
        self.classifier = nn.Linear(1000, 2)
        
    def forward(self, x, y):
        x = torch.cat((x, y), dim=1)
        x = self.pre(x)
        x = self.mobilenet(x)
        # y = self.mobilenet_b(y)
        # x = torch.cat((x, y), dim=1)
        x = self.classifier(x)
        return x
