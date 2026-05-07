import torch
import torch.nn as nn


def soft_iou(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1) - intersection
    return (intersection + eps) / (union + eps)


class BCESoftIoULoss(nn.Module):
    def __init__(self, iou_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCELoss()
        self.iou_weight = iou_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = self.bce(pred, target)
        iou = soft_iou(pred, target).mean()
        return bce + self.iou_weight * (1.0 - iou)
