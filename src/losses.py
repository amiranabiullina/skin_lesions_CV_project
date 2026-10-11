import torch
import torch.nn as nn


class BCEDiceLoss(nn.Module):
    """BCE on logits + soft Dice loss (computed per image, then averaged)."""

    def __init__(self, bce_weight=1.0, dice_weight=1.0, eps=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.eps = eps

    def forward(self, logits, target):
        prob = torch.sigmoid(logits)
        dims = (1, 2, 3)
        inter = (prob * target).sum(dims)
        dice = (2 * inter + self.eps) / (prob.sum(dims) + target.sum(dims) + self.eps)
        return self.bce_weight * self.bce(logits, target) + self.dice_weight * (1 - dice.mean())
