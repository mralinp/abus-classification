"""The 2D-input ResNet34 of Yu et al., Applied Sciences 14(24):11611 (2024), Section 2.3.1."""
import torch
from torch import nn
from torchvision.models import ResNet34_Weights, resnet34

MEAN, STD = torch.tensor([0.485, 0.456, 0.406])[:, None, None], torch.tensor([0.229, 0.224, 0.225])[:, None, None]


class YuResNet(nn.Module):
    """ResNet34 with a two-class output. Grey slices (n, 1, h, w) in [0, 1] are repeated to three channels."""

    def __init__(self, pretrained=True):
        super().__init__()
        self.net = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        self.net.fc = nn.Linear(512, 2)

    def forward(self, x):
        return self.net((x.repeat(1, 3, 1, 1) - MEAN.to(x)) / STD.to(x))
