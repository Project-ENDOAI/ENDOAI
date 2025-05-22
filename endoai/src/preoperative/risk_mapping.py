import numpy as np
import scipy.ndimage
import nibabel as nib

def map_risk_zones(segmented_lesions, atlas_path, threshold=5):
    """
    Map risk zones by calculating proximity to critical structures in the anatomy atlas.
    """
    # Load anatomy atlas
    atlas = nib.load(atlas_path).get_fdata()

    # Calculate distance to nearest risk zone
    risk_mask = (atlas == 1)  # Assume '1' is the label for nerves
    distance_map = scipy.ndimage.distance_transform_edt(~risk_mask)

    # Threshold for high-risk zones
    high_risk_zones = distance_map < threshold

    # Combine with segmented lesions
    combined_risk = np.logical_and(segmented_lesions > 0.5, high_risk_zones)
    return combined_risk

# Example usage
# risk_zones = map_risk_zones(segmented_lesions, "data/atlas/anatomy_atlas.nii.gz")
