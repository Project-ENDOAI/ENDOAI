# pipelines

This directory contains end-to-end machine learning and AI pipelines for ENDOAI.

## Purpose

- Automate data processing, model training, and evaluation workflows.
- Enable reproducible and scalable AI experiments.
- Integrate preoperative and intraoperative data processing tasks.

## Guidelines

- Keep pipelines modular and configurable.
- Document pipeline steps, inputs, and outputs.
- Use logging to track pipeline execution and errors.
- Ensure compatibility with the project's data formats and models.

## Pipeline Overview

- **data_preprocessing.py**: Handles data cleaning, transformation, and augmentation.
- **model_training.py**: Automates model training and hyperparameter tuning.
- **model_evaluation.py**: Evaluates trained models and generates performance metrics.
- **deployment_pipeline.py**: Prepares models for deployment in production environments.

## See Also

- [../README.md](../README.md) — Project-level documentation.
- [../notebooks/](../notebooks/) — Jupyter notebooks for experimentation and prototyping.
- [../src/](../src/) — Source code for core functionalities.
