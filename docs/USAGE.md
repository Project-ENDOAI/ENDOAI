# Usage Guide

This document provides instructions for using the ENDOAI project, including training, inference, and evaluation workflows.

## Training a Model

1. **Prepare the Dataset**:
   - Place raw data in the `data/raw/` directory.
   - Use the preprocessing pipeline to prepare the data:
     ```bash
     python src/preoperative/preprocessing_pipeline.py
     ```

2. **Train the Model**:
   - Run the training script:
     ```bash
     python src/preoperative/train.py
     ```

3. **Monitor Training**:
   - Use TensorBoard or logs to monitor training progress.

## Running Inference

1. **Prepare Input Data**:
   - Place input images in the `data/preoperative/images/` directory.

2. **Run Inference**:
   - Use the inference script:
     ```bash
     python src/inference/preoperative_inference.py --patient-id patient_001
     ```

3. **View Results**:
   - Check the output directory for predictions and visualizations.

## Evaluating a Model

1. **Prepare Validation Data**:
   - Place validation data in the `data/raw/Task04_Hippocampus/` directory.

2. **Run Evaluation**:
   - Use the evaluation script:
     ```bash
     python validation/evaluate.py --model-path models/lesion_segmentation.pth
     ```

3. **Check Metrics**:
   - Review the Dice score and other metrics printed in the console.

## See Also

- [../README.md](../README.md) — Project-level documentation.
- [TESTING.md](TESTING.md) — Guidelines for testing the project.
- [DEPLOYMENT.md](DEPLOYMENT.md) — Deployment instructions.
