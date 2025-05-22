import SimpleITK as sitk
from monai.transforms import Compose, LoadImage, ScaleIntensity, GaussianSmooth

def preprocess_mri(input_path, output_path):
    """
    Preprocess MRI data: intensity normalization, noise removal, and save processed images.
    """
    # Load MRI data
    image = sitk.ReadImage(input_path)

    # Preprocessing pipeline
    transforms = Compose([
        LoadImage(image_only=True),
        ScaleIntensity(),
        GaussianSmooth(sigma=1.0)
    ])
    processed_image = transforms(image)

    # Save processed image
    sitk.WriteImage(sitk.GetImageFromArray(processed_image), output_path)

# Example usage
if __name__ == "__main__":
    preprocess_mri("data/raw/mri_sample.dcm", "data/processed/mri_sample_processed.nii")
