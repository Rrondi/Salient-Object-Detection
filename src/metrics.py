import torch


def _flatten(pred: torch.Tensor, target: torch.Tensor):
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    return pred, target


def iou_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred, target = _flatten(pred, target)
    inter = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1) - inter
    return ((inter + eps) / (union + eps)).mean()


def precision_recall_f1(pred_bin: torch.Tensor, target_bin: torch.Tensor, eps: float = 1e-6):
    pred, target = _flatten(pred_bin, target_bin)
    tp = (pred * target).sum(dim=1)
    fp = (pred * (1 - target)).sum(dim=1)
    fn = ((1 - pred) * target).sum(dim=1)
    precision = ((tp + eps) / (tp + fp + eps)).mean()
    recall = ((tp + eps) / (tp + fn + eps)).mean()
    f1 = (2 * precision * recall) / (precision + recall + eps)
    return precision, recall, f1


def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.abs(pred - target).mean()
