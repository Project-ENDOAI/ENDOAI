"""
Preprocessing pipeline for medical imaging data using MONAI.
"""

import os
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, ScaleIntensity, Resize
from monai.data import Dataset

def create_preprocessed_dataset(images_dir, labels_dir, image_shape=(128, 128, 64)):
    """
    Create a MONAI Dataset with preprocessing transforms.
    """
    preprocess_transforms = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        ScaleIntensity(),
        Resize(image_shape)
    ])
    images = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir)])
    labels = sorted([os.path.join(labels_dir, f) for f in os.listdir(labels_dir)])
    data = [{"image": img, "label": lbl} for img, lbl in zip(images, labels)]
    return Dataset(data=data, transform=preprocess_transforms)
