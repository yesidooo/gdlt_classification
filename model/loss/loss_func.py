"""
Cross Entropy w/ smoothing or soft targets
Hacked together by / Copyright 2021 Ross Wightman
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['NllLoss', 'CrossEntropy', 'SoftTargetCrossEntropy', 'LabelSmoothingCrossEntropy', 'FocalLoss']


class NllLoss(nn.Module):
    """NllLoss, 负对数似然损失, 输入的 x 已经经过 log_softmax.
    """
    def __init__(self):
        super(NllLoss).__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = x
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
        loss = nll_loss.squeeze(1)
        return loss.mean()


class CrossEntropy(nn.Module):
    """交叉熵损失, 输入的 x 未经过 log_softmax
    """
    def __init__(self):
        super(CrossEntropy).__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(x, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
        loss = nll_loss.squeeze(1)
        return loss.mean()


class SoftTargetCrossEntropy(nn.Module):
    """ CrossEntropy loss with SoftTarget.
    """
    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


class LabelSmoothingCrossEntropy(nn.Module):
    """ NLL loss with label smoothing and log_softmax.
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(x, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class FocalLoss(nn.Module):

    def __init__(self, alpha: list, gamma=2, reduction='mean'):
        super(FocalLoss).__init__()
        """
        :param alpha: 类别权重
        :param gamma: 困难样本挖掘的gamma
        :param reduction:
        """
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha[target]
        log_probs = torch.log_softmax(x, dim=1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        pt = torch.exp(nll_loss)
        loss = alpha * (1 - pt) ** self.gamma * nll_loss
        if self.reduction == "mean":
            return torch.mean(loss)
        if self.reduction == "sum":
            return torch.sum(loss)
        return loss
