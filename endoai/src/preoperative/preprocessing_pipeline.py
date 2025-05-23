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
    preprocess_transforms = Compose([
        LoadImage(image_only=True, reader="NibabelReader"),
        EnsureChannelFirst(),
        ScaleIntensity(),
        Resize(image_shape)
    ])
    images = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir)])
    labels = sorted([os.path.join(labels_dir, f) for f in os.listdir(labels_dir)])

    # Match files by stem (remove extension and possible _mask or _seg postfix)
    def stem(filename):
        return os.path.splitext(os.path.basename(filename))[0].replace("_mask", "").replace("_seg", "")

    image_stems = {stem(f): f for f in images}
    label_stems = {stem(f): f for f in labels}
    common_stems = sorted(set(image_stems.keys()) & set(label_stems.keys()))

    data = []
    for s in common_stems:
        img_path = image_stems[s]
        lbl_path = label_stems[s]
        # Extra check: ensure file extensions are .nii or .nii.gz
        if not (img_path.endswith(('.nii', '.nii.gz')) and lbl_path.endswith(('.nii', '.nii.gz'))):
            print(f"Skipping non-NIfTI file: image={img_path}, label={lbl_path}")
            continue
        if not (os.path.isfile(img_path) and os.path.isfile(lbl_path)):
            print(f"Warning: File missing for pair: image={img_path}, label={lbl_path}")
            continue
        data.append({"image": img_path, "label": lbl_path})

    if not data:
        print("Warning: No matching image/label pairs found. Check filenames and directory structure.")
    else:
        print(f"Found {len(data)} matching image/label pairs.")
        print("First 3 pairs:")
        for d in data[:3]:
            print(d)
    return Dataset(data=data, transform=preprocess_transforms)
