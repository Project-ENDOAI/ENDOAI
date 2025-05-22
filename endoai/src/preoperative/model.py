"""
Defines a 3D UNet model for segmentation using MONAI.
"""

import torch
from monai.networks.nets import UNet

def get_unet_model(spatial_dims=3, in_channels=1, out_channels=2):
    """
    Returns a MONAI 3D UNet model.
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
