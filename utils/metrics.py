import torch
import numpy as np


def rmse(inputs: torch.Tensor, targets: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((inputs - targets) ** 2)).item()

def rmse_numpy(inputs: np.ndarray, targets: np.ndarray) -> float:
    return np.sqrt(np.mean((inputs - targets) ** 2))

def mae(inputs: torch.Tensor, targets: torch.Tensor) -> float:
    return torch.mean(torch.abs(inputs - targets)).item()

def mae_numpy(inputs: np.ndarray, targets: np.ndarray) -> float:
    return np.mean(np.abs(inputs - targets))

def masked_rmse(inputs: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> float:
    return rmse(inputs[mask], targets[mask])

def masked_rmse_numpy(inputs: np.ndarray, targets: np.ndarray, mask: np.ndarray) -> float:
    return rmse_numpy(inputs[mask], targets[mask])

def masked_mae(inputs: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> float:
    return mae(inputs[mask], targets[mask])

def masked_mae_numpy(inputs: np.ndarray, targets: np.ndarray, mask: np.ndarray) -> float:
    return mae_numpy(inputs[mask], targets[mask])


def binary_f1_score(y_pred: torch.Tensor, y_true: torch.Tensor, threshold: float = 0.5, eps: float = 1e-8) -> float:
    """
    Computes the average binary F1 score:
      - First over each image (channel, height, width),
      - Then averaged over the batch.

    Args:
        y_pred (torch.Tensor): Predicted probabilities or logits with shape [B, C, H, W].
        y_true (torch.Tensor): Ground truth binary labels with shape [B, C, H, W].
        threshold (float): Threshold for converting probabilities to binary values.
        eps (float): Small epsilon to avoid division by zero.

    Returns:
        float: Average F1 score over the batch.
    """
    # Binarize predictions
    y_pred_bin = (y_pred > threshold).float()

    # Flatten per image: shape becomes [B, C*H*W]
    y_pred_flat = y_pred_bin.view(y_pred.size(0), -1)
    y_true_flat = y_true.view(y_true.size(0), -1)

    # Compute per-image TP, FP, FN
    tp = (y_pred_flat * y_true_flat).sum(dim=1)
    fp = (y_pred_flat * (1 - y_true_flat)).sum(dim=1)
    fn = ((1 - y_pred_flat) * y_true_flat).sum(dim=1)

    # Compute per-image precision and recall
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)

    # Per-image F1 scores
    f1 = 2 * (precision * recall) / (precision + recall + eps)

    # Average over batch
    return f1.mean().item()
