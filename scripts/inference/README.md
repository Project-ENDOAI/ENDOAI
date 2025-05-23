# Inference Utilities

This folder contains scripts for running inference with trained models in the ENDOAI project.

## Scripts

- `predict.py`  
  A script for running inference on a single image using a trained segmentation model.  
  Example usage:
  ```bash
  python predict.py --model-path models/lesion_segmentation.pth --image-path <path_to_image> --output-path <path_to_save_prediction>
  ```

## Purpose

The inference utilities are designed to:
- Load trained models and run predictions on new data.
- Preprocess input images to match the model's requirements.
- Save predictions in NIfTI format for further analysis or visualization.

## Output

Predictions are saved in the same directory as the input image by default, with `_pred.nii.gz` appended to the filename. Use the `--output-path` argument to specify a custom location.

## Next Steps

After running inference, you can:
1. Visualize the predictions using the `view_results.py` script in the `visualization` folder.
2. Compare predictions with ground truth labels if available.
3. Use the predictions for downstream tasks or deployment.
