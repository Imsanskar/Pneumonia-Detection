import torch.nn as nn
import torch

class Classifier(nn.Module):
    def generate_conv_module(self, in_channel, out_channel, kernel_size, stride, padding, pool_kernel):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=stride),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.MaxPool2d(pool_kernel),
        )
    
    def __init__(self):
        super().__init__()
        self.conv_1 = self.generate_conv_module(3, 12, 3, 1, 1, 2)
        self.conv_2 = nn.Sequential(
            nn.Conv2d(12, 20, 5, stride = 1, padding = 1),
            nn.ReLU()
        )
        self.generate_conv_module(12, 20, 5, 1, 1, 2)
        self.conv_3 = self.generate_conv_module(20, 32, 3, 1, 1, 2)
        self.conv_4 = nn.Sequential(
            nn.Conv2d(32, 48, 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(48*27*27, 2)
        )
        
    def forward(self, x):
        y = self.conv_4(self.conv_3(self.conv_2(self.conv_1(x))))
        y = torch.flatten(y, 1)
        y = self.fc(y)
        return y

    def get_feature_map(self, x):
        return self.conv_4(self.conv_3(self.conv_2(self.conv_1(x))))