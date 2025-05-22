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
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = model(images)
            loss = loss_function(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
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

    images_dir = "data/public/Task04_Hippocampus/imagesTr"
    labels_dir = "data/public/Task04_Hippocampus/labelsTr"

    if not (os.path.isdir(images_dir) and os.path.isdir(labels_dir)):
        print("Error: Public dataset not found. Please download and extract the Medical Segmentation Decathlon Task04_Hippocampus dataset.")
        print("Download link: https://msd-for-monai.s3-us-west-2.amazonaws.com/Task04_Hippocampus.tar")
        print("After extraction, set images_dir and labels_dir accordingly.")
        exit(1)

    dataset = create_preprocessed_dataset(images_dir, labels_dir)
    train_model(dataset)
