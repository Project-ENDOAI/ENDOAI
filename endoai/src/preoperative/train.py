"""
Training loop for 3D UNet segmentation model using MONAI.
"""

import torch
from monai.losses import DiceLoss
from torch.optim import Adam
from monai.data import DataLoader
import sys
import os

# Ensure the current file's directory is in sys.path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from model import get_unet_model

def train_model(dataset, epochs=10, batch_size=2, lr=1e-4, model_save_path="models/lesion_segmentation.pth"):
    """
    Train a UNet model on the given dataset.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = get_unet_model()
    optimizer = Adam(model.parameters(), lr=lr)
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch in dataloader:
            try:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
            except Exception as e:
                print(f"Error loading batch data: {e}")
                print("Batch content:", batch)
                raise
            try:
                outputs = model(images)
                loss = loss_function(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            except Exception as e:
                print(f"Error during forward/backward pass: {e}")
                raise
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    """
    Example usage: Run this script to train the model.
    You should import or create your dataset before calling train_model.
    """
    # Example: Import the preprocessing pipeline and create the dataset
    from preprocessing_pipeline import create_preprocessed_dataset

    # For early training, you can use a public dataset such as the Medical Segmentation Decathlon (MSD).
    # Download the "Task04_Hippocampus" or similar dataset from:
    # https://msd-for-monai.s3-us-west-2.amazonaws.com/Task04_Hippocampus.tar

    # After downloading and extracting, set the paths below to the images and labels folders.
    # For demonstration, we use the MSD Hippocampus task as an example.
    # Adjust these paths to match your extracted dataset structure.

    # Set the paths to the extracted dataset in data/raw/
    images_dir = "data/raw/Task04_Hippocampus/imagesTr"
    labels_dir = "data/raw/Task04_Hippocampus/labelsTr"

    if not (os.path.isdir(images_dir) and os.path.isdir(labels_dir)):
        print("Error: Public dataset not found. Please download and extract the Medical Segmentation Decathlon Task04_Hippocampus dataset.")
        print("Download link: https://msd-for-monai.s3-us-west-2.amazonaws.com/Task04_Hippocampus.tar")
        print("After extraction, set images_dir and labels_dir accordingly (e.g., data/raw/Task04_Hippocampus/imagesTr).")
        exit(1)

    # Print out the first few image/label paths to help debug file loading issues
    print("Sample image files:", os.listdir(images_dir)[:5])
    print("Sample label files:", os.listdir(labels_dir)[:5])
    dataset = create_preprocessed_dataset(images_dir, labels_dir)
    print("Dataset length:", len(dataset))
    if len(dataset) == 0:
        print("No valid image/label pairs found. Please check your data directory and file naming.")
        exit(1)
    # Print the first sample dict to verify file paths
    if hasattr(dataset, 'data') and len(dataset.data) > 0:
        print("First sample dict:", dataset.data[0])
        # Check if the files exist on disk
        img_path = dataset.data[0]["image"]
        lbl_path = dataset.data[0]["label"]
        print("Image file exists:", os.path.isfile(img_path), img_path)
        print("Label file exists:", os.path.isfile(lbl_path), lbl_path)
    try:
        sample = dataset[0]
        print("Loaded first sample successfully.")
        print("Sample image shape:", sample["image"].shape)
        print("Sample label shape:", sample["label"].shape)
    except Exception as e:
        print(f"Error loading first sample: {e}")
        print("First sample dict:", dataset.data[0] if hasattr(dataset, 'data') else "N/A")
        raise
    train_model(dataset)
