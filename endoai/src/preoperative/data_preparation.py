import os
import nibabel as nib
import numpy as np
from monai.transforms import (
    LoadImage, ScaleIntensity, Resize, EnsureChannelFirst, Compose
)

# Define preprocessing pipeline
preprocessing = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    ScaleIntensity(),
    Resize((128, 128, 64))  # Resize to standard shape
])

def preprocess_dataset(input_dir, output_dir):
    """
    Preprocess MRI dataset: Convert DICOM to NIfTI, normalize, and resize.
    """
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        image = preprocessing(os.path.join(input_dir, filename))
        nib.save(nib.Nifti1Image(image.numpy(), np.eye(4)), os.path.join(output_dir, filename.replace(".dcm", ".nii.gz")))

# Example usage
if __name__ == "__main__":
    preprocess_dataset("data/raw/images", "data/processed/images")
    preprocess_dataset("data/raw/labels", "data/processed/labels")
