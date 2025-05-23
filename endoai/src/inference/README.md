# inference

This directory contains scripts and modules for running inference on preoperative and intraoperative data in the ENDOAI project.

## Purpose

The inference modules are designed to:
- Load patient-specific data (e.g., images and metadata).
- Preprocess data for compatibility with trained models.
- Run predictions using trained machine learning models.
- Output results for clinical decision-making.

## Modules Overview

- **preoperative_inference.py**: Handles inference on preoperative patient data, including medical images and metadata.

## Guidelines

- Ensure all inference scripts are modular and reusable.
- Document the input, output, and usage of each script.
- Follow the project's coding standards and best practices.

## See Also

- [../README.md](../README.md) — Source folder documentation.
- [../../models/](../../models/) — Directory containing trained models.
- [../../data/](../../data/) — Directory containing patient data.
