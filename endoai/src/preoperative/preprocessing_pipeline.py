"""
Preprocessing pipeline for medical imaging data using MONAI.
"""

import os
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, ScaleIntensity, Resize
from monai.data import Dataset

def create_preprocessed_dataset(images_dir, labels_dir, image_shape=(128, 128, 64)):
    """
    Create a MONAI Dataset with preprocessing transforms.
    Only include files that exist in both images and labels directories.
    """
    # Define separate transforms for images and labels
    image_transforms = Compose([
        LoadImage(image_only=True, reader="NibabelReader"),
        EnsureChannelFirst(),
        ScaleIntensity(),
        Resize(image_shape)
    ])
    
    # Label transforms - ensure values are 0 or 1 for binary segmentation
    label_transforms = Compose([
        LoadImage(image_only=True, reader="NibabelReader"),
        EnsureChannelFirst(),
        # Ensure labels are binary (0 or 1) - this is important for DiceLoss
        lambda x: (x > 0).float(),
        Resize(image_shape)
    ])
    
    # Get all image and label files with full absolute paths
    images_dir = os.path.abspath(images_dir)
    labels_dir = os.path.abspath(labels_dir)
    
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

    # Match files by filename without extension
    def get_filename(path):
        # Extract filename without extension
        return os.path.splitext(os.path.basename(path))[0].split('.')[0]  # Handle double extensions like .nii.gz
    
    # Create dictionaries mapping filename to full path
    image_dict = {get_filename(img): img for img in images}
    label_dict = {get_filename(lbl): lbl for lbl in labels}
    
    # Find common filenames
    common_filenames = set(image_dict.keys()) & set(label_dict.keys())
    
    # Create dataset items
    data = []
    for name in sorted(common_filenames):
        img_path = image_dict[name]
        lbl_path = label_dict[name]
        # Verify files exist before including
        if os.path.isfile(img_path) and os.path.isfile(lbl_path):
            data.append({
                "image": img_path,
                "label": lbl_path,
                "image_transforms": image_transforms,
                "label_transforms": label_transforms
            })
        else:
            print(f"Warning: File missing for pair: image={img_path}, label={lbl_path}")
    
    if not data:
        print("Warning: No matching image/label pairs found. Check filenames and directory structure.")
    else:
        print(f"Found {len(data)} matching image/label pairs.")
        print("First 3 pairs:")
        for d in data[:3]:
            print({"image": d["image"], "label": d["label"]})
    
    # Custom dataset class that applies separate transforms
    class SeparateTransformDataset(Dataset):
        def __init__(self, data):
            self.data = data
            
        def __len__(self):
            return len(self.data)
            
        def __getitem__(self, index):
            item = self.data[index]
            image = item["image_transforms"](item["image"])
            label = item["label_transforms"](item["label"])
            return {"image": image, "label": label}
    
    return SeparateTransformDataset(data)
