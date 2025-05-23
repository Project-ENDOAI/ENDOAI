"""
Visualization Module

This script provides utilities for visualizing model predictions and ground truth labels.
"""

import os
import matplotlib.pyplot as plt
import nibabel as nib

def visualize_prediction(image_path, label_path, prediction, output_dir="../reports/visualizations"):
    """
    Visualize the original image, ground truth label, and model prediction.

    Args:
        image_path (str): Path to the original image file.
        label_path (str): Path to the ground truth label file.
        prediction (torch.Tensor): Model prediction tensor.
        output_dir (str): Directory to save the visualization.

    Returns:
        str: Path to the saved visualization.
    """
    os.makedirs(output_dir, exist_ok=True)
    image = nib.load(image_path).get_fdata()
    label = nib.load(label_path).get_fdata()
    prediction_np = prediction.cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image[:, :, image.shape[2] // 2], cmap="gray")
    axes[0].set_title("Original Image")
    axes[1].imshow(label[:, :, label.shape[2] // 2], cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[2].imshow(prediction_np[:, :, prediction_np.shape[2] // 2], cmap="gray")
    axes[2].set_title("Prediction")

    for ax in axes:
        ax.axis("off")

    output_path = os.path.join(output_dir, f"visualization_{os.path.basename(image_path)}.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Visualization saved to {output_path}")
    return output_path
