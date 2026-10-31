"""The dCNN of Hejduk et al., Eur. Radiol. 32:4868-4878 (2022), "dCNN architecture"."""
import torch
from torch import nn

HEIGHT, WIDTH = 181, 151
BLOCKS = ((64, 64), (128, 128), (256, 256, 256), (512, 512, 512), (512, 512, 512))


class HejdukCNN(nn.Module):
    """13 convolutions, 5 max poolings, 2 ReLU dense layers with 50% dropout and a two-class output (VGG-16 layout)."""

    def __init__(self, dense=4096):
        super().__init__()
        layers, channels = [], 1
        for block in BLOCKS:
            for out in block:
                layers += [nn.Conv2d(channels, out, 3, padding=1), nn.ReLU()]
                channels = out
            layers.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*layers)
        size = 512 * (HEIGHT // 32) * (WIDTH // 32)
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(size, dense), nn.ReLU(), nn.Dropout(0.5),
                                nn.Linear(dense, dense), nn.ReLU(), nn.Dropout(0.5), nn.Linear(dense, 2))
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):      # TensorFlow's default initialisation
                nn.init.xavier_uniform_(m.weight), nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.fc(self.features(x))
