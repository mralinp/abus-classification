"""Modified Inception-v3 and multiview CNN B of Wang et al., Ultrasound Med. Biol. 46(5):1119-1132 (2020)."""
import torch
from torch import nn
from torchvision.models import Inception_V3_Weights, inception_v3

MEAN, STD = torch.tensor([0.485, 0.456, 0.406])[:, None, None], torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def backbone(pretrained):
    """Inception-v3 convolutional layers + GAP: 2048 features. The ImageNet weights expect ImageNet-normalised input."""
    net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None, init_weights=not pretrained)
    net.aux_logits, net.AuxLogits = False, None
    net.dropout, net.fc = nn.Identity(), nn.Identity()
    return net


def head(features, dropout=0.5):
    """FC 256, 128 and 2 with dropout after the first two; Xavier uniform weights."""
    fc = nn.Sequential(nn.Linear(features, 256), nn.ReLU(), nn.Dropout(dropout),
                       nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 2))
    for m in fc:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight), nn.init.zeros_(m.bias)
    return fc


class WangCNN(nn.Module):
    """3 input channels: one backbone (multiview CNN A, single view). 6 channels: two backbones (multiview CNN B)."""

    def __init__(self, channels=3, pretrained=True):
        super().__init__()
        self.backbones = nn.ModuleList(backbone(pretrained) for _ in range(channels // 3))
        self.fc = head(2048 * len(self.backbones))

    def forward(self, x):
        mean, std = MEAN.to(x), STD.to(x)
        features = [b((v - mean) / std) for b, v in zip(self.backbones, x.split(3, 1))]
        return self.fc(torch.cat(features, 1))
