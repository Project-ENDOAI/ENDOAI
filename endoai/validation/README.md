# validation

This folder contains scripts and utilities for evaluating and validating trained models in ENDOAI.

## Purpose

- Evaluate model performance on validation datasets.
- Compute metrics such as Dice score, IoU (Jaccard Index), and others.
- Visualize predictions and ground truth for qualitative analysis.
- Provide scripts for model selection and comparison.

## Structure

- `evaluate.py` — Main evaluation script for segmentation models.
- `metrics.py` — Contains additional metrics such as Dice Coefficient and Jaccard Index.
- `visualize_predictions.py` — Utilities for visualizing predictions and ground truth labels.
- `__init__.py` — Initializes the `validation` module for modular imports.

## Usage

### Evaluate a Model
Use `evaluate_model` from `evaluate.py` to compute metrics on your validation dataset:
```python
from evaluate import evaluate_model

model_path = "models/lesion_segmentation.pth"
data_dir = "data/raw/Task04_Hippocampus"
batch_size = 2

evaluate_model(model_path, data_dir, batch_size)
```

### Compute Metrics
Use `compute_dice_coefficient` or `compute_jaccard_index` from `metrics.py`:
```python
from metrics import compute_dice_coefficient, compute_jaccard_index

dice = compute_dice_coefficient(y_true, y_pred)
jaccard = compute_jaccard_index(y_true, y_pred)
```

### Visualize Predictions
Use `visualize_prediction` from `visualize_predictions.py` to generate visualizations:
```python
from visualize_predictions import visualize_prediction

visualize_prediction(image_path, label_path, prediction, output_dir="reports/visualizations")
```

## Guidelines

- Extend with additional metrics or visualization tools as needed.
- Ensure all scripts are well-documented and tested.
- Follow the project's coding standards and best practices.

## See Also

- [../README.md](../README.md) — Project-level documentation.
- [../src/](../src/) — Source code for model training and inference.
- [../reports/](../reports/) — Directory for storing evaluation reports and visualizations.
