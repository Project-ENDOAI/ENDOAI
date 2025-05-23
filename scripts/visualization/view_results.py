"""
Script for visualizing prediction results from the segmentation model.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from monai.transforms import LoadImage, Resize
import datetime

# Ensure project root is in Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def load_nifti_file(filepath):
    """Load a NIfTI file using MONAI's LoadImage transform."""
    loader = LoadImage(image_only=True)
    return loader(filepath)

def visualize_results(image_path, pred_path, label_path=None, output_dir=None, slices=None):
    """
    Visualize the original image, prediction, and optionally the ground truth label.
    """
    # Load the data
    image = load_nifti_file(image_path)
    pred = load_nifti_file(pred_path)
    label = load_nifti_file(label_path) if label_path else None
    
    # Get dimensions and ensure we have numpy arrays
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if label is not None and isinstance(label, torch.Tensor):
        label = label.detach().cpu().numpy()
        
    # Remove channel dimension if present
    if len(image.shape) > 3:
        image = image[0]
    if len(pred.shape) > 3:
        pred = pred[0]
    if label is not None and len(label.shape) > 3:
        label = label[0]
        
    # Automatically select slices if not provided
    if slices is None:
        # Find slices with segmentation
        if label is not None:
            nonzero_slices = np.where(np.sum(label, axis=(0, 1)) > 0)[0]
        else:
            nonzero_slices = np.where(np.sum(pred, axis=(0, 1)) > 0)[0]
            
        if len(nonzero_slices) > 0:
            # Choose 3 slices from segmented regions
            slice_indices = np.linspace(
                nonzero_slices[0], 
                nonzero_slices[-1], 
                min(3, len(nonzero_slices)), 
                dtype=int
            )
        else:
            # If no segmentation, choose middle slices
            slice_indices = [image.shape[2] // 4, image.shape[2] // 2, 3 * image.shape[2] // 4]
    else:
        slice_indices = slices
    
    # Create visualization for each slice
    for idx, slice_idx in enumerate(slice_indices):
        if slice_idx >= image.shape[2]:
            print(f"Warning: Slice index {slice_idx} out of bounds. Skipping.")
            continue
            
        plt.figure(figsize=(15, 5))
        
        # Add image info in suptitle
        image_name = os.path.basename(image_path)
        plt.suptitle(f"Visualization for {image_name} - Dice Score: {get_dice_score(pred, label) if label is not None else 'N/A'}", 
                    fontsize=14)
        
        # Original image
        plt.subplot(1, 3 if label is not None else 2, 1)
        plt.title(f"Original Image (Slice {slice_idx})")
        plt.imshow(image[:, :, slice_idx], cmap='gray')
        plt.axis('off')
        
        # Prediction overlay
        plt.subplot(1, 3 if label is not None else 2, 2)
        plt.title(f"Prediction Overlay (Slice {slice_idx})")
        plt.imshow(image[:, :, slice_idx], cmap='gray')
        plt.imshow(pred[:, :, slice_idx], alpha=0.5, cmap='hot')
        plt.axis('off')
        
        # Ground truth overlay (if available)
        if label is not None:
            plt.subplot(1, 3, 3)
            plt.title(f"Ground Truth Overlay (Slice {slice_idx})")
            plt.imshow(image[:, :, slice_idx], cmap='gray')
            plt.imshow(label[:, :, slice_idx], alpha=0.5, cmap='cool')
            plt.axis('off')
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f"visualization_slice_{slice_idx}.png")
            plt.savefig(save_path)
            print(f"Saved visualization to: {save_path}")
        
        # Always display the visualization
        plt.tight_layout()
        plt.show()
        
        plt.close()

def get_dice_score(pred, truth):
    """Calculate Dice similarity coefficient between prediction and truth."""
    if truth is None:
        return "N/A"
    
    # Ensure binary masks
    pred_binary = (pred > 0.5).astype(np.float32)
    
    # Resize truth to match pred shape if necessary
    if pred_binary.shape != truth.shape:
        resize_transform = Resize(spatial_size=pred_binary.shape, mode="nearest")
        truth = resize_transform(truth)
    
    truth_binary = (truth > 0).astype(np.float32)
    
    # Calculate Dice
    intersection = np.sum(pred_binary * truth_binary)
    union = np.sum(pred_binary) + np.sum(truth_binary)
    
    if union == 0:
        return 1.0  # Both empty means perfect match
    return 2.0 * intersection / union

def main():
    parser = argparse.ArgumentParser(description='Visualize prediction results from the segmentation model.')
    
    # Try to find the most recent prediction
    default_image = None
    default_prediction = None
    
    # Look for the most recently created prediction with _pred.nii.gz suffix
    data_dir = os.path.join(project_root, "data", "raw", "Task04_Hippocampus")
    images_dir = os.path.join(data_dir, "imagesTr")
    
    if os.path.isdir(images_dir):
        # Find all prediction files and sort by creation time (newest first)
        pred_files = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith("_pred.nii.gz"):
                    full_path = os.path.join(root, file)
                    pred_files.append((full_path, os.path.getmtime(full_path)))
        
        pred_files.sort(key=lambda x: x[1], reverse=True)
        
        if pred_files:
            default_prediction = pred_files[0][0]
            # Find the corresponding original image
            original_filename = os.path.basename(default_prediction).replace("_pred.nii.gz", ".nii.gz")
            original_path = os.path.join(images_dir, original_filename)
            if os.path.exists(original_path):
                default_image = original_path
    
    # Add arguments with defaults if found
    if default_image and os.path.isfile(default_image):
        parser.add_argument('--image', type=str, default=default_image,
                        help=f'Path to the input image (default: {os.path.basename(default_image)})')
    else:
        parser.add_argument('--image', type=str, required=True,
                        help='Path to the input image')
    
    if default_prediction and os.path.isfile(default_prediction):
        parser.add_argument('--prediction', type=str, default=default_prediction,
                        help=f'Path to the prediction mask (default: {os.path.basename(default_prediction)})')
    else:
        parser.add_argument('--prediction', type=str, required=True,
                        help='Path to the prediction mask')
    
    parser.add_argument('--label', type=str, help='Optional path to the ground truth label')
    
    default_output_dir = os.path.join(project_root, "output", "visualizations", 
                                      f"vis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument('--output-dir', type=str, default=default_output_dir, 
                        help=f'Directory to save visualization images (default: {default_output_dir})')
    parser.add_argument('--slices', type=int, nargs='+', help='Optional list of slice indices to visualize')
    
    args = parser.parse_args()
    
    # Automatically find label if not provided
    if args.label is None:
        # Try to find corresponding label based on image path
        image_dir = os.path.dirname(args.image)
        image_filename = os.path.basename(args.image)
        
        # Check if this is from the MSD dataset
        if "imagesTr" in image_dir:
            label_dir = image_dir.replace("imagesTr", "labelsTr")
            if os.path.exists(label_dir):
                potential_label = os.path.join(label_dir, image_filename)
                if os.path.exists(potential_label):
                    args.label = potential_label
                    print(f"Found matching label: {args.label}")
    
    visualize_results(args.image, args.prediction, args.label, args.output_dir, args.slices)
    
    if args.output_dir:
        print(f"\nVisualizations saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
