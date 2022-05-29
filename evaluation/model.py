from torch import nn
import torch
from torch.nn import functional as F
from torchvision.models import mobilenet_v2


class Classifier(nn.Module):
    def __init__(self, input_features: int=1, input_filters: int=16):
        super().__init__()
        
        # Input is (B, 1, 64, 64) 
        k = 4
        self.norm1 = nn.BatchNorm2d(input_features)
        self.conv1 = nn.Conv2d(input_features, input_filters, 
                               kernel_size=k, stride=2, padding=k//2-1)  # -> (B, 16, 32, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)                  # -> (B, 16, 16, 16)
        self.norm2 = nn.BatchNorm2d(input_filters)
        self.conv2 = nn.Conv2d(input_filters, input_filters * 2, 
                               kernel_size=k, stride=2, padding=k//2-1)  # -> (B, 32, 8, 8)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)                  # -> (B, 32, 4, 4)
        self.norm3 = nn.BatchNorm2d(input_filters * 2)
        self.conv3 = nn.Conv2d(input_filters * 2, input_filters * 4, 
                               kernel_size=k, stride=2, padding=k//2-1)  # -> (B, 64, 2, 2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)                  # -> (B, 64, 1, 1)
        self.fc1 = nn.Linear(input_filters * 4, 2)
    
    def forward(self, x):
        # input is (B, 1, 64, 64)
        x = self.norm1(x)
        x = self.conv1(x)   # -> (B, 16, 32, 32)
        x = self.pool1(x)   # -> (B, 16, 16, 16)
        x = F.dropout(x, p=0.2)
        x = torch.tanh(x)
        x = self.norm2(x)
        x = self.conv2(x)
        x = self.pool2(x)   # -> (B, 32, 4, 4)
        x = F.dropout(x, p=0.2)
        x = torch.tanh(x)
        x = self.norm3(x)
        x = self.conv3(x)   # -> (B, 64, 2, 2)
        x = self.pool3(x)   # -> (B, 64, 1, 1)
        x = F.dropout(x, p=0.2)
        x = torch.tanh(x)
        x = x.view(x.size(0), -1) # -> (B, 64)
        x = self.fc1(x)     # -> (B, 2)
        return x

class CDomain(nn.Module):
    '''Binary classification: whether an image is in a domain'''
    def __init__(self, num_features: int=1):
        super().__init__()
        self.linear = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0)
        self.mobilenet = mobilenet_v2(pretrained=True) # 3.5M params
        self.classifier = nn.Linear(1000, 2)
        
    def forward(self, x):
        x = self.linear(x)
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
