"""Zhou et al. 2021 losses: Dice (Eq. 4) and class-weighted focal (Eq. 2-3)."""
import torch


def dice_loss(seg, mask):
    p = torch.sigmoid(seg)
    return 1 - (2 * (p * mask).sum() + 1) / (p.sum() + mask.sum() + 1)


def focal_loss(logit, y, w_m, w_n, gamma=2.0):
    p = torch.sigmoid(logit).clamp(1e-6, 1 - 1e-6)
    return (-w_m * (1 - p) ** gamma * y * p.log() - w_n * p ** gamma * (1 - y) * (1 - p).log()).mean()
