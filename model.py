import torch
from torch.nn import *
import torch.nn.functional as F
from torch import optim
import torchvision.models as models
from torch.utils.data import DataLoader

class ImgSpecModel(Module):
    def __init__(self):
        super(ImgSpecModel, self).__init__()

        self.spec_layers = Sequential(
            Conv2d(1, 4, kernel_size=3, stride=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(4, 8, kernel_size=3, stride=1),
            BatchNorm2d(8),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(8, 16, kernel_size=3, stride=1),
            BatchNorm2d(16),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )

        self.reduction = Sequential(Linear(14400, 2048))
        
        self.classifier = Sequential(
            Linear(4096, 512),
            Linear(512, 2),
        )
        
    def forward(self, img_feats, spec_imgs):
        img_feats = img_feats.view(img_feats.size(0), -1)
        x = self.spec_layers(spec_imgs)
        x = x.view(x.size(0), -1)
        spec_feats = self.reduction(x)
        feats = torch.cat((img_feats, spec_feats), dim=1)
        x = self.classifier(feats)
        return x
