"""The three networks of T1. The paper names their families only (Fig. 3 is not specified further): see pipeline.py."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.video import r3d_18


def double_conv(a, b):
    return nn.Sequential(nn.Conv3d(a, b, 3, padding=1, bias=False), nn.BatchNorm3d(b), nn.ReLU(inplace=True),
                         nn.Conv3d(b, b, 3, padding=1, bias=False), nn.BatchNorm3d(b), nn.ReLU(inplace=True))


class UNet3D(nn.Module):
    """U-Net on single-channel volumes whose shape divides by 2 ** depth. `stop` leaves out that many top decoder levels."""

    def __init__(self, out, base=16, depth=4, stop=0):
        super().__init__()
        widths = [base * 2 ** i for i in range(depth + 1)]
        self.down = nn.ModuleList(double_conv(a, b) for a, b in zip([1] + widths, widths))
        self.up = nn.ModuleList(nn.ConvTranspose3d(b, a, 2, stride=2) for a, b in zip(widths[stop:], widths[stop + 1:]))
        self.merge = nn.ModuleList(double_conv(2 * a, a) for a in widths[stop:-1])
        self.head = nn.Conv3d(widths[stop], out, 1)

    def forward(self, x):
        skips = []
        for i, block in enumerate(self.down):
            x = block(x if i == 0 else F.max_pool3d(x, 2))
            skips.append(x)
        x = skips.pop()
        for up, merge in zip(reversed(self.up), reversed(self.merge)):
            x = merge(torch.cat([skips.pop(), up(x)], 1))
        return self.head(x)


class SEGNet(UNet3D):
    """Voxel logits of the lesion."""

    def __init__(self):
        super().__init__(1)

    def forward(self, x):
        return super().forward(x)[:, 0]


class DETNet(UNet3D):
    """Single-stage and anchor-free, after CPMNet: at stride 2, a centre heatmap logit, 3 offsets and 3 log sizes."""
    STRIDE = 2

    def __init__(self):
        super().__init__(7, depth=3, stop=1)
        nn.init.constant_(self.head.bias[0], -4.6)                    # start at p = 0.01, as focal losses do


class CLSNet(nn.Module):
    """3D ResNet-18 with softmax over benign and malignant; call `F.softmax(net(x), 1)[:, 1]` for p(malignant)."""

    def __init__(self):
        super().__init__()
        self.net = r3d_18(num_classes=2)
        self.net.stem[0] = nn.Conv3d(1, 64, (3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)

    def forward(self, x):
        return self.net(x)
