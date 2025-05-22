# preoperative

This folder contains code and resources for **preoperative planning** in the ENDOAI project.

## Purpose

- Preprocessing, segmentation, and analysis of preoperative imaging data (e.g., MRI, ultrasound).
- Scripts and modules for lesion detection, risk mapping, and data preparation before surgery.
- Utilities for data normalization, augmentation, and conversion.

## Quickstart: What to Run First

1. **Preprocessing:**  
   Run the preprocessing pipeline to prepare your data:
   ```bash
   python -m endoai.src.preoperative.preprocessing_pipeline
   ```
   This will load, normalize, and resize your images and labels.

2. **Training:**  
   Train your segmentation model using the prepared dataset:
   ```bash
   python -m endoai.src.preoperative.train
   ```
   This will train the UNet model and save the weights to the `models/` directory.

3. **Validation/Evaluation:**  
   Evaluate your trained model on a validation set:
   ```bash
   python -m endoai.validation.evaluate
   ```
   This will compute metrics such as Dice score on your validation data.

- Adjust script paths and arguments as needed for your environment and data.
- See each script’s docstring or help message for more details.

## Running Training

You can run `train.py` directly:

```bash
python -m endoai.src.preoperative.train
```

Or, for more flexibility and reproducibility, you may create a separate script (e.g., `scripts/run_training.py`) to set up environment variables, logging, or experiment tracking, and then call the training module. This is especially useful if you want to automate multiple experiments or integrate with MLflow, Weights & Biases, or a job scheduler.

**For most users, running `train.py` directly is sufficient for initial experiments.**

## Example: Building a Simple Segmentation Model with MONAI

### 1. Data Preparation

- Organize your data into `images/` and `labels/` folders (labels are segmentation masks).
- Use MONAI transforms for preprocessing (see `data_preparation.py` and `preprocessing.py`).

### 2. Preprocessing Pipeline Example

```python
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, ScaleIntensity, Resize
from monai.data import Dataset

preprocess_transforms = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    ScaleIntensity(),
    Resize((128, 128, 64))
])

data = [
    {"image": "data/processed/images/img1.nii.gz", "label": "data/processed/labels/img1_mask.nii.gz"},
    # ...
]
dataset = Dataset(data=data, transform=preprocess_transforms)
```

### 3. Model Definition Example

```python
from monai.networks.nets import UNet
import torch

model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2)
).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
```

### 4. Training Loop Example

```python
from monai.losses import DiceLoss
import torch.optim as optim
from monai.data import DataLoader

dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_function = DiceLoss(to_onehot_y=True, softmax=True)

for epoch in range(10):
    model.train()
    epoch_loss = 0
    for batch in dataloader:
        images = batch["image"].to(model.device)
        labels = batch["label"].to(model.device)
        outputs = model(images)
        loss = loss_function(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
torch.save(model.state_dict(), "models/lesion_segmentation.pth")
```

### 5. Next Steps

- Add data augmentation to the preprocessing pipeline.
- Experiment with different architectures and hyperparameters.
- Integrate experiment tracking (e.g., MLflow, Weights & Biases).
- Add unit and integration tests for each module.
- Use MONAI’s metrics for evaluation and validation.

## Training Scripts Organization

- If you have multiple training scripts for different tasks or experiments, consider creating a `train/` directory at the project root (e.g., `ENDOAI/train/`) and placing your training scripts there (e.g., `train_preoperative.py`, `train_intraoperative.py`).
- If your training is specific to preoperative tasks, you can keep `train.py` and related scripts in this `preoperative` folder.
- Example structure for a general training folder:
  ```
  ENDOAI/
  ├── train/
  │   ├── train_preoperative.py
  │   ├── train_intraoperative.py
  │   └── ...
  ```

Organize your training scripts according to your project's needs for clarity and modularity.

## Structure

- `data_preparation.py` — Preprocessing and normalization of raw imaging data.
- `preprocessing.py` — Additional preprocessing utilities.
- `mri_segmentation.py` — MRI segmentation models and pipelines.
- `lesion_detection.py` — Lesion detection algorithms.
- `risk_mapping.py` — Risk mapping and visualization tools.
- Other scripts and modules related to preoperative workflows.

## Usage

Import modules as needed in your pipeline or run scripts directly for data processing and model training.

## Contributing

- Add new scripts or modules relevant to preoperative planning.
- Document all new files and functions.
- Follow the project’s [coding standards](../../../COPILOT.md).

## See Also

- [../intraoperative/](../intraoperative/) — For intraoperative (real-time) guidance.
- [../../README.md](../../README.md) — Project overview.
