"""
Validation Module

This package contains scripts and utilities for evaluating and visualizing segmentation models.
"""

from .evaluate import evaluate_model
from .metrics import compute_jaccard_index, compute_dice_coefficient
from .visualize_predictions import visualize_prediction
