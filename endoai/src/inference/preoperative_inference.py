import os
import json
import nibabel as nib
import torch

# Try importing the model and handle import errors
try:
    from model import get_trained_model
except ImportError as e:
    print(f"Error importing 'get_trained_model' from 'model': {e}")
    print("Ensure that 'model.py' exists and contains the 'get_trained_model' function.")
    raise

def load_patient_data(patient_id, base_dir="../data/preoperative"):
    """
    Load patient images and metadata.

    Args:
        patient_id (str): Unique identifier for the patient.
        base_dir (str): Base directory containing preoperative data.

    Returns:
        dict: Dictionary of images loaded as nibabel objects.
        dict: Metadata loaded as a dictionary.
    """
    # Load images
    image_dir = os.path.join(base_dir, "images", patient_id)
    images = {file: nib.load(os.path.join(image_dir, file)) for file in os.listdir(image_dir)}

    # Load metadata
    metadata_path = os.path.join(base_dir, "metadata", f"{patient_id}.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    return images, metadata

def run_inference(patient_id, model_path="../models/trained_model.pth"):
    """
    Run inference on patient data using a trained model.

    Args:
        patient_id (str): Unique identifier for the patient.
        model_path (str): Path to the trained model file.

    Returns:
        torch.Tensor: Predictions from the model.
        dict: Metadata associated with the patient.
    """
    # Load patient data
    images, metadata = load_patient_data(patient_id)

    # Preprocess images (example: convert to tensor)
    image_tensor = torch.tensor([img.get_fdata() for img in images.values()])

    # Load trained model
    model = get_trained_model()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Run inference
    with torch.no_grad():
        predictions = model(image_tensor)

    return predictions, metadata

if __name__ == "__main__":
    patient_id = "patient_001"
    predictions, metadata = run_inference(patient_id)
    print(f"Predictions for {patient_id}: {predictions}")
    print(f"Metadata: {metadata}")
