import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import utils.ssim as ssim

def temporal_gradient_matching_loss(depth_pred_t1, depth_gt_t, depth_gt_t1):
    """
    Compute temporal gradient matching loss between predictions and ground truth.

    Args:
        depth_pred_t (Tensor): Predicted depth at time t (B, 1, H, W)
        depth_pred_t1 (Tensor): Predicted depth at time t+1 (B, 1, H, W)
        depth_gt_t (Tensor): Ground truth depth at time t (B, 1, H, W)
        depth_gt_t1 (Tensor): Ground truth depth at time t+1 (B, 1, H, W)

    Returns:
        Tensor: Scalar loss value
    """
    pred_grad = depth_pred_t1 - depth_gt_t
    gt_grad = depth_gt_t1 - depth_gt_t
    return F.l1_loss(pred_grad, gt_grad)


def gradient_matching_loss(pred, target):
    """
    PyTorch implementation of Gradient-Matching Loss for depth estimation.
    Args:
        pred (torch.Tensor): Predicted depth map of shape (B, 1, H, W)
        target (torch.Tensor): Ground truth depth map of shape (B, 1, H, W)
    Returns:
        torch.Tensor: Scalar loss value
    """
    # Compute gradients in x and y directions for prediction
    grad_pred_x = pred[:, :, :, :-1] - pred[:, :, :, 1:]
    grad_pred_y = pred[:, :, :-1, :] - pred[:, :, 1:, :]

    # Compute gradients in x and y directions for target
    grad_target_x = target[:, :, :, :-1] - target[:, :, :, 1:]
    grad_target_y = target[:, :, :-1, :] - target[:, :, 1:, :]

    # Compute mean squared error between gradients
    loss_x = F.mse_loss(grad_pred_x, grad_target_x)
    loss_y = F.mse_loss(grad_pred_y, grad_target_y)

    return loss_x + loss_y


class MaskedSmoothL1Loss(nn.Module):
    """
    Custom loss function that combines Smooth L1 Loss with a mask.
    """
    def __init__(self, beta: float = 1.0):
        super(MaskedSmoothL1Loss, self).__init__()
        self.smooth_l1_loss = nn.SmoothL1Loss(beta=beta, reduction='mean')
        self.beta = beta

    def forward(self, output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.smooth_l1_loss(output[mask], target[mask])


class MaskedSmoothL1LossWithSSIM(nn.Module):
    """
    Custom loss function that combines Smooth L1 Loss with SSIM and a mask.
    """
    def __init__(self, beta: float = 1.0, lam: float = 1.0):
        super(MaskedSmoothL1LossWithSSIM, self).__init__()
        self.beta = beta
        self.smooth_l1_loss = nn.SmoothL1Loss(beta=beta, reduction='mean')
        self.lam = lam

    def forward(self, output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        ssim_loss = 1 - ssim.ssim(output * mask.float(), target * mask.float(), val_range=200)
        return self.smooth_l1_loss(output[mask], target[mask]) + self.lam * ssim_loss
    
    
class SmoothL1LossWithSSIM(nn.Module):
    """
    Custom loss function that combines Smooth L1 Loss with SSIM.
    """
    def __init__(self, beta: float = 1.0, lam: float = 1.0):
        super(SmoothL1LossWithSSIM, self).__init__()
        self.beta = beta
        self.lam = lam
        self.smooth_l1_loss = nn.SmoothL1Loss(beta=beta, reduction='mean')

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ssim_loss = 1 - ssim.ssim(output, target, val_range=200)
        return self.smooth_l1_loss(output, target) + self.lam * ssim_loss
    
    
def sobel_edge_detector(x: torch.Tensor) -> torch.Tensor:
    # Ensure float and same device as input
    device, dtype = x.device, x.dtype

    # Define Sobel kernels
    kx = torch.tensor([[1,  0, -1],
                       [2,  0, -2],
                       [1,  0, -1]], device=device, dtype=dtype).view(1, 1, 3, 3)

    ky = torch.tensor([[ 1,  2,  1],
                       [ 0,  0,  0],
                       [-1, -2, -1]], device=device, dtype=dtype).view(1, 1, 3, 3)

    # Stack into a (2,1,3,3) weight tensor
    weight = torch.cat([kx, ky], dim=0)

    # Convolve (no bias), stride=1, padding=1 to preserve H×W
    edges = F.conv2d(x, weight, bias=None, stride=1, padding=1)

    return edges


class GradientLoss(nn.Module):
    def __init__(self, device):
        super(GradientLoss, self).__init__()
        self.cos = nn.CosineSimilarity(dim=1, eps=0)
        self.device = device

    def forward(self, output: torch.Tensor, depth: torch.Tensor):
        depth_grad = sobel_edge_detector(depth)
        output_grad = sobel_edge_detector(output)

        depth_grad_dx = depth_grad[:, 0, :, :].contiguous().view_as(depth)
        depth_grad_dy = depth_grad[:, 1, :, :].contiguous().view_as(depth)
        output_grad_dx = output_grad[:, 0, :, :].contiguous().view_as(depth)
        output_grad_dy = output_grad[:, 1, :, :].contiguous().view_as(depth)

        loss_dx = torch.abs(output_grad_dx - depth_grad_dx).mean()
        loss_dy = torch.abs(output_grad_dy - depth_grad_dy).mean()
        
        loss_grad = loss_dx + loss_dy

        return loss_grad
    

class NormalLoss(nn.Module):
    def __init__(self, device):
        super(NormalLoss, self).__init__()
        self.cos = nn.CosineSimilarity(dim=1, eps=0)
        self.device = device

    def forward(self, output: torch.Tensor, depth: torch.Tensor):
        with torch.no_grad():
            ones = torch.ones(depth.size(0), 1, depth.size(2), depth.size(3)).float().to(self.device)

        depth_grad = sobel_edge_detector(depth)
        output_grad = sobel_edge_detector(output)

        depth_grad_dx = depth_grad[:, 0, :, :].contiguous().view_as(depth)
        depth_grad_dy = depth_grad[:, 1, :, :].contiguous().view_as(depth)
        output_grad_dx = output_grad[:, 0, :, :].contiguous().view_as(depth)
        output_grad_dy = output_grad[:, 1, :, :].contiguous().view_as(depth)

        depth_normal = torch.cat((-depth_grad_dx, -depth_grad_dy, ones), 1)
        output_normal = torch.cat((-output_grad_dx, -output_grad_dy, ones), 1)
        loss_normal = torch.abs(1 - self.cos(output_normal, depth_normal)).mean()

        return loss_normal
    
    
class MultiLoss(nn.Module):
    """
    Custom loss function that combines multiple losses.
    """
    def __init__(self, device, weights: dict = None):
        super(MultiLoss, self).__init__()
        self.dist_loss = nn.MSELoss(reduction='mean')
        self.gradient_loss = GradientLoss(device)
        self.normal_loss = NormalLoss(device)
        self.weights = weights if weights is not None else {
            "dist_loss": 1.0,
            "ssim": 1.0,
            "gradient": 1.0,
            "normal": 1.0
        }

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dist_loss = self.dist_loss(output, target)
        ssim_loss = 1 - ssim.ssim(output, target, val_range=200)
        gradient_loss = self.gradient_loss(output, target)
        normal_loss = self.normal_loss(output, target)

        return (self.weights["dist_loss"] * dist_loss +
                self.weights["ssim"] * ssim_loss +
                self.weights["gradient"] * gradient_loss +
                self.weights["normal"] * normal_loss)


class MSELossWithSSIM(nn.Module):
    """
    Custom loss function that combines MSE Loss with SSIM and a mask.
    """
    def __init__(self, lam: float = 1.0):
        super(MSELossWithSSIM, self).__init__()
        self.loss = nn.MSELoss(reduction='mean')
        self.lam = lam

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ssim_loss = 1 - ssim.ssim(output, target, val_range=200)
        return self.loss(output, target) + self.lam * ssim_loss
    
    
class MSELogLoss(nn.Module):
    def __init__(self):
        super(MSELogLoss, self).__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        error = torch.log(inputs + 1e-6) - torch.log(targets + 1e-6)
        is_nan = torch.isnan(error)
        return torch.mean(error[~is_nan] ** 2)
    
class ScaleInvariantMSELoss(nn.Module):
    def __init__(self):
        super(ScaleInvariantMSELoss, self).__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        error = inputs - targets
        return torch.mean(error ** 2) - torch.mean(error) ** 2


class ScaleInvariantLogMSELoss(nn.Module):
    def __init__(self):
        super(ScaleInvariantLogMSELoss, self).__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        error = torch.log(inputs + 1e-6) - torch.log(targets + 1e-6)
        is_nan = torch.isnan(error)
        return torch.mean(error[~is_nan] ** 2) - torch.mean(error[~is_nan]) ** 2
    

class MSEMaskedLoss(nn.Module):
    def __init__(self, lam=1.0):
        super(MSEMaskedLoss, self).__init__()
        self.lam = lam

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mask = (targets == 0)
        return self.lam * F.mse_loss(inputs[mask], targets[mask], reduction='mean') + (1 - self.lam) * F.mse_loss(inputs[~mask], targets[~mask], reduction='mean')
        


class ReverseHuberLoss(nn.Module):
    def __init__(self, c=None, fraction=0.2, reduction="mean"):
        """
        Reverse Huber (berHu) Loss.

        Args:
            c (float or None): fixed threshold. If None, threshold is set adaptively
                               as fraction * max|error| per batch.
            fraction (float): if c is None, use fraction * max|error| as threshold.
            reduction (str): "mean", "sum", or "none"
        """
        super().__init__()
        self.c = c
        self.fraction = fraction
        self.reduction = reduction

    def forward(self, input, target):
        error = torch.abs(input - target)
        
        # adaptive threshold if not fixed
        c = self.c if self.c is not None else self.fraction * error.max().detach()

        # piecewise definition
        l1_part = error <= c
        loss = torch.where(l1_part, error, (error ** 2 + c ** 2) / (2 * c))

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # "none"
            return loss


class ConfidenceAwareLoss(nn.Module):
    def __init__(self, alpha=1):
        """
        alpha: weight for the confidence regularization term
        loss_type: "l1" or "l2" regression loss
        """
        super().__init__()
        self.alpha = alpha

    def forward(self, preds, target, conf_logits):
        """
        preds: [B, H, W]   regression predictions (e.g., depth)
        target: [B, H, W]  ground truth
        conf_logits: [B, H, W] raw confidence values before transformation
        """
        # Ensure confidence > 1
        conf = 1.0 + torch.exp(conf_logits)  # [B, H, W]
        regr_loss = (preds - target) ** 2
        loss = (conf * regr_loss - self.alpha * torch.log(conf))

        denom = torch.numel(loss)
        return loss.sum() / denom