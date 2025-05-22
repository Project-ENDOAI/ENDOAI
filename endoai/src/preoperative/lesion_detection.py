from monai.networks.nets import UNet
from monai.losses import DiceLoss
import torch

def train_lesion_segmentation(dataloader, epochs=10, lr=1e-4):
    """
    Train a 3D U-Net model for lesion segmentation.
    """
    # Define 3D U-Net model
    model = UNet(
        dimensions=3,
        in_channels=1,
        out_channels=2,  # Lesion vs non-lesion
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)

    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch in dataloader:
            images, labels = batch["image"], batch["label"]
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), "models/lesion_segmentation.pth")

# Example usage
# train_lesion_segmentation(dataloader)
