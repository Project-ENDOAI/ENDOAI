import os
import numpy as np
import torch
from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst, ScaleIntensity, EnsureType
)
from monai.networks.nets import UNet
from monai.data import Dataset, DataLoader
from monai.losses import DiceLoss

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define data transforms
transforms = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    ScaleIntensity(),
    EnsureType()
])

# Load training data
def load_data(data_dir):
    images = [os.path.join(data_dir, "images", f) for f in os.listdir(data_dir + "/images")]
    labels = [os.path.join(data_dir, "labels", f) for f in os.listdir(data_dir + "/labels")]
    return [{"image": img, "label": lbl} for img, lbl in zip(images, labels)]

# Prepare dataset
data_dir = "data/processed/mri"
train_data = load_data(data_dir)
train_ds = Dataset(data=train_data, transform=transforms)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)

# Define U-Net model
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2)
).to(device)

# Define optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_function = DiceLoss(to_onehot_y=True, softmax=True)

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch in train_loader:
        images, labels = batch["image"].to(device), batch["label"].to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

# Save model
torch.save(model.state_dict(), "models/mri_segmentation.pth")

# This file is now superseded by the modular pipeline in preprocessing_pipeline.py, model.py, train.py, and validation/evaluate.py.
