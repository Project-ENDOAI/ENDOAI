"""
Evaluation script for the trained segmentation model.
"""

import os
import sys
import torch
import argparse
import numpy as np
from monai.metrics import DiceMetric
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, ScaleIntensity, Resize
from monai.data import Dataset, DataLoader

# Add src/preoperative to sys.path for import resolution
current_dir = os.path.dirname(__file__)
src_preoperative_path = os.path.abspath(os.path.join(current_dir, "..", "src", "preoperative"))
if src_preoperative_path not in sys.path:
    sys.path.insert(0, src_preoperative_path)

try:
    from model import get_unet_model
except ImportError:
    print("Warning: Failed to import from relative path. Trying from absolute path.")
    from endoai.src.preoperative.model import get_unet_model

def create_validation_dataset(data_dir):
    """Create a validation dataset from the data directory."""
    # Use the first 10% of data for validation
    images_dir = os.path.join(data_dir, "imagesTr")
    labels_dir = os.path.join(data_dir, "labelsTr")
    
    transforms = Compose([
        LoadImage(image_only=True, reader="NibabelReader"),
        EnsureChannelFirst(),
        ScaleIntensity(),
        Resize((128, 128, 64)),
    ])
    
    label_transforms = Compose([
        LoadImage(image_only=True, reader="NibabelReader"),
        EnsureChannelFirst(),
        # Ensure labels are binary (0 or 1)
        lambda x: (x > 0).float(),
        Resize((128, 128, 64))
    ])
    
    # Skip hidden files (starting with '._')
    images = sorted([
        os.path.join(images_dir, f) 
        for f in os.listdir(images_dir) 
        if f.endswith(('.nii', '.nii.gz')) and not f.startswith('._')
    ])
    
    labels = sorted([
        os.path.join(labels_dir, f) 
        for f in os.listdir(labels_dir) 
        if f.endswith(('.nii', '.nii.gz')) and not f.startswith('._')
    ])
    
    # Use the last 10% of data for validation
    val_size = max(1, int(len(images) * 0.1))
    val_images = images[-val_size:]
    val_labels = labels[-val_size:]
    
    print(f"Using {len(val_images)} images for validation")
    
    val_data = []
    for img, lbl in zip(val_images, val_labels):
        if os.path.isfile(img) and os.path.isfile(lbl):
            val_data.append({"image": img, "label": lbl})
    
    class CustomDataset(Dataset):
        def __init__(self, data):
            self.data = data
            
        def __len__(self):
            return len(self.data)
            
        def __getitem__(self, index):
            item = self.data[index]
            image = transforms(item["image"])
            label = label_transforms(item["label"])
            return {"image": image, "label": label}
    
    return CustomDataset(val_data)

def evaluate_model(model_path, data_dir, batch_size=2):
    """
    Evaluate a trained model on a validation dataset and print metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model
    model = get_unet_model(out_channels=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    
    # Create validation dataset
    val_dataset = create_validation_dataset(data_dir)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize metrics
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    
    dice_scores = []
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > 0.5).float()
            dice_metric(y_pred=outputs, y=labels)
            dice_scores.append(dice_metric.aggregate().item())
            dice_metric.reset()
    
    avg_dice = np.mean(dice_scores) if dice_scores else 0.0
    print(f"Validation Dice Score: {avg_dice:.4f}")
    return avg_dice

def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained segmentation model.')
    parser.add_argument('--model-path', type=str, default='models/lesion_segmentation.pth', 
                        help='Path to the trained model')
    parser.add_argument('--data-dir', type=str, default='data/raw/Task04_Hippocampus', 
                        help='Path to the dataset directory')
    parser.add_argument('--batch-size', type=int, default=2, 
                        help='Batch size for evaluation')
    
    args = parser.parse_args()
    evaluate_model(args.model_path, args.data_dir, args.batch_size)

if __name__ == "__main__":
    main()
