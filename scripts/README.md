# scripts Directory

This directory contains utility scripts and automation tools for the ENDOAI project.

## Organization

Scripts are organized by category for clarity and scalability:

- `scripts/data/` — Scripts for downloading, preprocessing, or augmenting datasets.  
  Example: `download_public_dataset.py` automates the retrieval of public datasets like the Medical Segmentation Decathlon.
  
- `scripts/train/` — Scripts for orchestrating training, managing experiments, or performing hyperparameter searches.  
  Example: `run_training.py` sets up the environment, logging, and experiment tracking for training pipelines.

- `scripts/deploy/` — Scripts for exporting models, packaging them for deployment, or running inference.  
  Example: `export_model.py` converts trained models to ONNX format for deployment.

- `scripts/git/` — Scripts for automating git operations, managing credentials, or repository cleanup.  
  Example: `git_automation.py` simplifies repetitive git tasks like pushing changes or managing branches.

- `scripts/utils/` — Miscellaneous utility scripts for development, debugging, or maintenance.  
  Example: `list_python_files.py` lists all Python files in the project for quick reference.

## Example Structure
