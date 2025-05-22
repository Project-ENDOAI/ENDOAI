"""
Export a trained PyTorch model to ONNX format for deployment.
"""

import torch
import os

def export_to_onnx(model, dummy_input, export_path):
    torch.onnx.export(model, dummy_input, export_path, opset_version=11)
    print(f"Model exported to {export_path}")

if __name__ == "__main__":
    # Example usage: export a trained UNet model
    from endoai.src.preoperative.model import get_unet_model

    model = get_unet_model()
    weights_path = "endoai/models/lesion_segmentation.pth"
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        model.eval()
        dummy_input = torch.randn(1, 1, 128, 128, 64)
        export_path = "endoai/models/lesion_segmentation.onnx"
        export_to_onnx(model, dummy_input, export_path)
    else:
        print(f"Trained weights not found at {weights_path}")
