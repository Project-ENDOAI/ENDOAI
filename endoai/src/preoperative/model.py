"""
Defines a 3D UNet model for segmentation using MONAI.
"""

import torch
from monai.networks.nets import UNet

def get_unet_model(spatial_dims=3, in_channels=1, out_channels=2):
    """
    Returns a MONAI 3D UNet model.
    
    Args:
        spatial_dims (int): Number of spatial dimensions (2 for 2D, 3 for 3D).
        in_channels (int): Number of input channels (typically 1 for grayscale).
        out_channels (int): Number of output channels (1 for binary segmentation).
    """
    model = UNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device)
