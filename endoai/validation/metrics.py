"""
Metrics Module

This script provides additional metrics for evaluating segmentation models.
"""

import torch
import numpy as np
from sklearn.metrics import jaccard_score

def compute_jaccard_index(y_true, y_pred):
    """
    Compute the Jaccard Index (Intersection over Union) for binary segmentation.

    Args:
        y_true (torch.Tensor): Ground truth tensor (binary).
        y_pred (torch.Tensor): Predicted tensor (binary).

    Returns:
        float: Jaccard Index.
    """
    y_true_np = y_true.cpu().numpy().flatten()
    y_pred_np = y_pred.cpu().numpy().flatten()
    return jaccard_score(y_true_np, y_pred_np, average="binary")

def compute_dice_coefficient(y_true, y_pred):
    """
    Compute the Dice Coefficient for binary segmentation.

    Args:
        y_true (torch.Tensor): Ground truth tensor (binary).
        y_pred (torch.Tensor): Predicted tensor (binary).

    Returns:
        float: Dice Coefficient.
    """
    intersection = (y_true * y_pred).sum().item()
    union = y_true.sum().item() + y_pred.sum().item()
    return (2.0 * intersection) / union if union > 0 else 0.0
