"""
Evaluation and validation script for segmentation models using MONAI.
"""

import torch
from monai.metrics import DiceMetric
from monai.data import DataLoader

import sys
import os

# Add src/preoperative to sys.path for import resolution
current_dir = os.path.dirname(__file__)
src_preoperative_path = os.path.abspath(os.path.join(current_dir, "..", "src", "preoperative"))
if src_preoperative_path not in sys.path:
    sys.path.insert(0, src_preoperative_path)

from model import get_unet_model

def evaluate_model(dataset, model_path, batch_size=2):
    """
    Evaluate a trained model on a validation dataset and print average Dice score.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = DataLoader(dataset, batch_size=batch_size)
    model = get_unet_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_scores = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = model(images)
            value = dice_metric(outputs, labels)
            dice_scores.append(value.item())
    avg_dice = sum(dice_scores) / len(dice_scores) if dice_scores else 0.0
    print(f"Average Dice Score: {avg_dice:.4f}")
    return avg_dice
