import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader


# LeNet
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        # x: (64, 1, 28, 28)
        x = self.conv1(x)  # (64, 6, 28, 28)
        x = F.relu(x)
        x = self.pool1(x)
#         x = F.max_pool2d(x, (2,2))  # (64, 6, 14, 14)
        
        x = self.conv2(x)  # (64, 16, 10, 10)
        x = F.relu(x)
        x = self.pool2(x)
#         x = F.max_pool2d(x, (2,2))  # (64, 16, 5, 5)
        
        x = x.view(x.size()[0], -1)  # (64, 256)
        x = self.fc1(x)  # (64, 120)
        x = F.relu(x)
        
        x = self.fc2(x)  # (64, 84)
        x = F.relu(x)
        
        x = self.fc3(x)  # (64, 10)
        
        return F.log_softmax(x)
    
    
# AlexNet
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        k = 2
        n = 5
        alpha = 1e-4
        beta = 0.75
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(n, alpha, beta, k),
            nn.MaxPool2d(3, 2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(n, alpha, beta, k),
            nn.MaxPool2d(3, 2)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2)
        )
        
        self.layer6 = nn.Sequential(
            nn.Linear(256*6*6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )
        
        self.layer7 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )
        self.layer8 = nn.Sequential(
            nn.Linear(4096, num_classes)
        )
        
    def forward(self, x):
        x = self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x)))))
        x = x.view(x.shape[0], -1)
        x = self.layer8(self.layer7(self.layer6(x)))
        return F.log_softmax(x)


# VGG-16
class VGG(nn.Module):
    def __init__(self, num_classes):
        super(VGG, self).__init__()    # (N, 3, 224, 224)
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),    # (N, 64, 224, 224)
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),    # (N, 64, 224, 224)
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),    # (N, 64, 112, 112)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),    # (N, 128, 112, 112)
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),    # (N, 128, 112, 112)
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),    # (N, 128, 56, 56)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),    # (N, 256, 56, 56)
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),    # (N, 256, 56, 56)
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),    # (N, 256, 56, 56)
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),    # (N, 256, 28, 28)          
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),    # (N, 512, 28, 28)
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),    # (N, 512, 28, 28)
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),    # (N, 512, 28, 28)
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)        
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),    # (N, 512, 14, 14)
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),    # (N, 512, 14, 14)
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),    # (N, 512, 14, 14)
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)    # (N, 512, 7, 7)      
        )
        
        self.layer6 = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout()
        )
        self.layer7 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout()     
        )     
        self.layer8 = nn.Sequential(
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, x):
        x = self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x)))))
        x = x.view(x.shape[0], -1)
        x = self.layer8(self.layer7(self.layer6(x)))
        return F.log_softmax(x)