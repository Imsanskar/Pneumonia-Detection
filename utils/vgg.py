import torch.nn as nn 
import torchvision

class VGGClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg19 = torchvision.models.vgg19(pretrained=True)
        for param in self.vgg19.parameters():
            param.require_grad = False 
        for param in self.vgg19.classifier.parameters():
            param.requires_grad = True
        self.vgg19.classifier = nn.Linear(
            25088, 2
        )
        
    def forward(self, x):
        return self.vgg19(x)