"""
Script for running inference with the trained model.
"""

import os
import sys
import torch
import numpy as np
import argparse
import json
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, ScaleIntensity, Resize

# Ensure project root is in Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from endoai.src.preoperative.model import get_unet_model

def load_model(model_path):
    """Load the trained model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_unet_model(out_channels=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def preprocess_image(image_path):
    """Preprocess a single image for prediction."""
    transforms = Compose([
        LoadImage(image_only=True, reader="NibabelReader"),
        EnsureChannelFirst(),
        ScaleIntensity(),
        Resize((128, 128, 64))
    ])
    return transforms(image_path)

def predict(model, image):
    """Run prediction on a preprocessed image."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        image = image.to(device).unsqueeze(0)  # Add batch dimension
        prediction = model(image)
        prediction = torch.sigmoid(prediction)
        binary_mask = (prediction > 0.5).float()
        return binary_mask.cpu().numpy()

def save_prediction(prediction, output_path):
    """Save the prediction to disk."""
    import nibabel as nib
    # Create a NIfTI image and save it
    nii_image = nib.Nifti1Image(prediction[0, 0].astype(np.float32), np.eye(4))
    nib.save(nii_image, output_path)
    print(f"Prediction saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Run inference with the trained segmentation model.')
    parser.add_argument('--model-path', type=str, default='models/lesion_segmentation.pth', 
                        help='Path to the trained model')
    
    # Try to get default image path from settings or environment
    default_image_path = None
    vscode_settings_path = os.path.join(project_root, ".vscode", "settings.json")
    if os.path.exists(vscode_settings_path):
        try:
            with open(vscode_settings_path, "r") as f:
                settings = json.load(f)
                default_image_path = settings.get("endoai.default_image_path", None)
        except Exception:
            pass
            
    # If not found in settings, look for the default dataset location
    if not default_image_path:
        dataset_dir = os.path.join(project_root, "data", "raw", "Task04_Hippocampus")
        images_dir = os.path.join(dataset_dir, "imagesTr")
        if os.path.isdir(images_dir):
            nii_files = [f for f in os.listdir(images_dir) if f.endswith('.nii.gz') and not f.startswith('.')]
            if nii_files:
                default_image_path = os.path.join(images_dir, nii_files[0])
    
    # Add the argument, with default if available
    if default_image_path and os.path.isfile(default_image_path):
        parser.add_argument('--image-path', type=str, default=default_image_path,
                        help=f'Path to the input image (default: {os.path.basename(default_image_path)})')
    else:
        parser.add_argument('--image-path', type=str, required=True,
                        help='Path to the input image')
                        
    parser.add_argument('--output-path', type=str, default=None, 
                        help='Path to save the prediction (default: same as input with _pred suffix)')
    
    args = parser.parse_args()
    
    # Default output path if not specified
    if args.output_path is None:
        base_name = os.path.splitext(args.image_path)[0]
        if base_name.endswith('.nii'):
            base_name = os.path.splitext(base_name)[0]  # Handle double extension .nii.gz
        args.output_path = f"{base_name}_pred.nii.gz"
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path)
    
    # Preprocess image
    print(f"Preprocessing image {args.image_path}...")
    image = preprocess_image(args.image_path)
    
    # Run prediction
    print("Running prediction...")
    prediction = predict(model, image)
    
    # Save result
    save_prediction(prediction, args.output_path)
    
    # Visualization suggestion
    print("\n==== NEXT STEPS ====")
    print(f"1. Visualize your prediction with: python -m scripts.visualization.view_results --image={args.image_path} --prediction={args.output_path}")
    print("2. Compare with ground truth if available")
    print("3. Process more images or deploy your model")
    print("4. Fine-tune the model if needed to improve performance")
    print("====================\n")

if __name__ == "__main__":
    main()
