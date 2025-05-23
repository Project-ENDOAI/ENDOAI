# File Overview for ENDOAI Project

This document provides a summary of the key files and their purposes in the ENDOAI project. It is intended to help developers and contributors understand the structure and functionality of the project.

---

## Root Directory

### `README.md`
**Summary**: The main `README.md` is the gateway to the project. It typically includes a project description, setup instructions (dependencies, environment), how to run the code (training, inference, tests), contribution guidelines, and licensing information.
**Purpose**: Provides an overview of the ENDOAI project, including instructions for CI/CD, Docker usage, logging, testing, and documentation.
**Why Needed**: Serves as the entry point for understanding the project and its setup, crucial for new contributors and for maintaining a clear overview of the project's goals and usage.

---

## Scripts Directory

### `scripts/inference/predict.py`
**Summary**: This script is designed for using a pre-trained model to make predictions on new, unseen data. It loads the model, preprocesses the input image(s) to match the format expected by the model, performs the prediction, and saves the output (e.g., segmentation masks).
**Purpose**: Runs inference on a single image using a trained segmentation model. Handles preprocessing, model loading, prediction, and saving results.
**Why Needed**: Automates the process of applying the trained model to new data, making it easy to generate predictions for further analysis, clinical use, or evaluation.

### `scripts/inference/__init__.py`
**Summary**: An empty file that signifies to Python that the `inference` directory should be treated as a package. This allows for organized imports of modules within this directory.
**Purpose**: Initializes the `inference` module for the ENDOAI project.
**Why Needed**: Ensures the `inference` directory is treated as a Python package, allowing imports from this module (e.g., `from scripts.inference import predict_utils`).

### `scripts/inference/README.md`
**Summary**: Provides specific documentation for the inference scripts. It details how to run `predict.py`, explains its command-line arguments, expected input formats, and output formats.
**Purpose**: Documents the purpose and usage of the inference utilities, including example commands for running predictions.
**Why Needed**: Provides guidance to users on how to use the inference scripts effectively, ensuring they can easily make predictions with the trained models.

### `scripts/visualization/view_results.py`
**Summary**: This script helps in visually inspecting the model's performance. It typically loads an original image, its corresponding ground truth (if available), and the model's prediction, then displays them side-by-side or as overlays for comparison.
**Purpose**: Visualizes segmentation predictions alongside the original images and ground truth labels. Supports saving visualizations to disk and displaying them interactively.
**Why Needed**: Helps users analyze and debug model predictions by providing visual feedback, which is crucial for understanding model strengths and weaknesses.

### `scripts/visualization/__init__.py`
**Summary**: An empty file that signifies to Python that the `visualization` directory should be treated as a package.
**Purpose**: Initializes the `visualization` module for the ENDOAI project.
**Why Needed**: Ensures the `visualization` directory is treated as a Python package, allowing for structured imports of visualization utilities.

### `scripts/visualization/README.md`
**Summary**: Documents the visualization scripts, explaining how to use `view_results.py`, its arguments, and how to interpret the visual outputs.
**Purpose**: Documents the purpose and usage of the visualization utilities, including example commands for generating visualizations.
**Why Needed**: Provides guidance to users on how to use the visualization scripts effectively for model evaluation and result interpretation.

### `scripts/utils/setup_project_structure.py`
**Summary**: A utility script that automates the creation of the standard directory layout for the project. This might include creating `data`, `output`, `models`, `logs` folders, etc.
**Purpose**: Sets up the directory structure for the ENDOAI project, including creating output directories and adding README files.
**Why Needed**: Simplifies the process of initializing the project structure for new users or contributors, ensuring consistency across different development environments.

---

## Output Directory

### `output/visualizations/README.md`
**Summary**: Explains the organization of the `visualizations` sub-directory. It details where different types of visualizations (e.g., training progress, validation results, inference outputs) are stored.
**Purpose**: Explains the structure and purpose of the `visualizations` directory, including subdirectories for training, validation, and inference results.
**Why Needed**: Helps users understand where visualizations are stored and how to organize them, making it easier to find and interpret results.

---

## Core Project Files

### `endoai/src/preoperative/model.py`
**Summary**: Contains the Python code defining the neural network architecture(s) used in the project, such as a UNet for segmentation. This includes the layers, forward pass logic, and any model-specific helper functions.
**Purpose**: Defines the UNet model architecture used for segmentation tasks.
**Why Needed**: Central to the project, as it provides the model used for training and inference. This is the core of the machine learning component.

### `endoai/src/preoperative/preprocessing_pipeline.py`
**Summary**: This module includes functions and classes for preparing raw data for model training and inference. This involves loading images/labels, data augmentation (e.g., rotations, flips), normalization, resizing, and formatting data into tensors.
**Purpose**: Handles data preprocessing, including loading, resizing, and normalizing images and labels.
**Why Needed**: Ensures consistent preprocessing for training, validation, and inference. Proper data preparation is critical for model performance.

### `endoai/src/preoperative/train.py`
**Summary**: Contains the main script or functions for training the model. This typically includes setting up the dataset and dataloaders, defining the optimizer and loss function, the training loop (iterating over epochs and batches), validation loop, and saving model checkpoints.
**Purpose**: Implements the training loop for the segmentation model, including data loading, loss calculation, and model optimization.
**Why Needed**: Automates the training process, allowing users to train the model on new datasets and experiment with different hyperparameters.

### `endoai/src/preoperative/__init__.py`
**Summary**: An empty file that makes the `preoperative` directory a Python package.
**Purpose**: Initializes the `preoperative` module for the ENDOAI project.
**Why Needed**: Ensures the `preoperative` directory is treated as a Python package, allowing for modular code organization and imports like `from endoai.src.preoperative import model`.

### `endoai/src/intraoperative/__init__.py`
**Summary**: An empty file that makes the `intraoperative` directory a Python package.
**Purpose**: Initializes the `intraoperative` module for the ENDOAI project.
**Why Needed**: Ensures the `intraoperative` directory is treated as a Python package, allowing for organized imports like `from endoai.src.intraoperative import sensor_fusion`.

### `endoai/src/intraoperative/README.md`
**Summary**: A documentation file explaining the purpose and components of the intraoperative module, including sensor integration and real-time processing features.
**Purpose**: Provides an overview of the intraoperative module's capabilities and usage examples.
**Why Needed**: Helps users and developers understand how to use the intraoperative components effectively.

### `endoai/src/intraoperative/sensor_fusion.py`
**Summary**: Provides a framework for integrating data from multiple sensor types in real-time during surgery. Implements different fusion strategies and data processing pipelines.
**Purpose**: Combines data from various sensors to create a unified representation for surgical guidance.
**Why Needed**: Essential for providing comprehensive information to surgeons by leveraging multiple data sources simultaneously.

### `endoai/src/intraoperative/visual_processing.py`
**Summary**: Processes and enhances visual data from cameras and imaging systems used in surgical procedures. Includes image enhancement, segmentation, and feature extraction.
**Purpose**: Improves the quality and information content of visual data for better surgical guidance.
**Why Needed**: Raw visual data often needs processing to highlight important features or remove noise before presentation to surgeons.

### `endoai/src/intraoperative/force_feedback.py`
**Summary**: Processes data from force and tactile sensors to provide haptic feedback and tissue characterization. Includes safety monitoring for excessive pressure.
**Purpose**: Translates force data into useful information about tissue properties and potential risks.
**Why Needed**: Force data provides critical information about tissue resistance and characteristics that visual data alone cannot capture.

### `endoai/src/intraoperative/preop_fusion.py`
**Summary**: Integrates preoperative imaging data with intraoperative sensor data. Implements registration algorithms to align preoperative plans with the current surgical state.
**Purpose**: Bridges the gap between preoperative planning and intraoperative execution.
**Why Needed**: Allows surgeons to reference critical information identified during planning while operating, improving decision-making and precision.

---

## Documentation

### `docs/`
**Summary**: This directory is intended to hold all project-related documentation that isn't directly in README files. This could include detailed design documents, API documentation generated by tools like Sphinx, user manuals, or research papers related to the project.
**Purpose**: Contains project documentation, including API references and usage guides.
**Why Needed**: Provides detailed information for developers and users, ensuring the project is well-documented, maintainable, and understandable.

---

## Tests Directory

### `tests/README.md`
**Summary**: Provides an overview of the testing strategy for the project. It explains how to run tests, the types of tests included (unit, integration), and conventions for writing new tests.
**Purpose**: Documents the purpose and structure of the test suite for the ENDOAI project.
**Why Needed**: Provides guidance on running and writing tests to ensure code quality, correctness, and prevent regressions.

### `tests/test_preprocessing_pipeline.py`
**Summary**: Contains unit tests specifically for the `preprocessing_pipeline.py` module. These tests verify that data loading, transformations, and augmentations work correctly under various conditions.
**Purpose**: Tests the functionality of the preprocessing pipeline, including data loading, resizing, and normalization.
**Why Needed**: Ensures that the preprocessing pipeline works as expected and handles edge cases, which is critical for reliable model training.

### `tests/test_model.py`
**Summary**: Contains unit tests for the `model.py` module. These tests might check if the model can be instantiated correctly, if the forward pass runs without errors, and if output shapes are as expected for given input shapes.
**Purpose**: Tests the UNet model architecture, including forward passes and output shape validation.
**Why Needed**: Ensures that the model is implemented correctly and produces the expected outputs, catching potential bugs in the network architecture.

---

## Environment and Configuration

### `.vscode/settings.json`
**Summary**: Contains workspace-specific settings for Visual Studio Code users. This can include Python interpreter paths, linter configurations, debugger settings, and default values for script arguments to streamline development.
**Purpose**: Stores project-specific settings for the Visual Studio Code editor, such as default paths for datasets and models.
**Why Needed**: Simplifies development by pre-configuring paths and environment variables for the project, ensuring a consistent development experience for VS Code users.

### `requirements.txt`
**Summary**: A text file listing all external Python packages and their specific versions required to run the project. This file is used by `pip` to install dependencies.
**Purpose**: Lists all Python dependencies required for the project.
**Why Needed**: Ensures that all contributors and deployment environments have the same dependencies installed, promoting reproducibility and avoiding "works on my machine" issues.

### `venv/`
**Summary**: This directory typically contains a Python virtual environment. A virtual environment is an isolated Python installation that allows project-specific dependencies to be installed without affecting the global Python setup or other projects.
**Purpose**: Contains the virtual environment for the project, isolating dependencies from the global Python environment.
**Why Needed**: Prevents dependency conflicts between different projects and ensures that the project runs with the exact versions of libraries it was developed and tested with.

---

## Python Files Analysis

### Core Application Files

### `endoai/src/preoperative/model.py`
**Python Analysis**: Implements a 3D UNet architecture using PyTorch. The model is designed with an encoder-decoder structure specifically for volumetric medical image segmentation. Uses standard PyTorch modules including nn.Conv3d, nn.BatchNorm3d, and nn.ReLU. The forward method describes the complete flow of data through the network, implementing skip connections between encoder and decoder blocks.

### `endoai/src/preoperative/preprocessing_pipeline.py`
**Python Analysis**: Built on the MONAI framework for medical imaging preprocessing. Creates a preprocessing pipeline using the Compose transform, which chains multiple operations. Implements robust data loading that handles both NIfTI formats and other medical imaging formats. Uses spatial transforms like resizing and normalization to prepare images for the model. Includes checks for dataset validity and file existence.

### `endoai/src/preoperative/train.py`
**Python Analysis**: Implements a standard PyTorch training loop with MONAI integration. Uses the Adam optimizer and DiceLoss, which is specialized for segmentation tasks. Contains both training and validation phases, with metrics being tracked for each. Implements model checkpoint saving based on validation performance. Uses PyTorch's DataLoader for efficient batch processing.

### `endoai/src/preoperative/__init__.py`
**Python Analysis**: An empty file that makes the `preoperative` directory a Python package.

### `endoai/src/intraoperative/__init__.py`
**Python Analysis**: An empty file that makes the `intraoperative` directory a Python package.

### `endoai/src/intraoperative/README.md`
**Python Analysis**: A documentation file explaining the purpose and components of the intraoperative module, including sensor integration and real-time processing features.

### `endoai/src/intraoperative/sensor_fusion.py`
**Python Analysis**: Implements a modular sensor fusion pipeline using Python classes for different fusion strategies. Uses NumPy and PyTorch for efficient data processing. Features a plugin architecture where different sensor types can be registered dynamically. Includes both feature-level and decision-level fusion techniques for combining multimodal sensor data.

### `endoai/src/intraoperative/visual_processing.py`
**Python Analysis**: Provides classes for processing various types of visual sensor data including laparoscopic images and stereoscopic views. Uses OpenCV for image enhancement and feature extraction. Implements several image processing techniques such as CLAHE for contrast enhancement and adaptive thresholding. Contains a stereo vision processor for depth estimation from dual-camera setups.

### `endoai/src/intraoperative/force_feedback.py`
**Python Analysis**: Contains multiple classes for handling force sensor data, including calibration and real-time processing. Implements tissue characterization based on force measurements. Features a haptic feedback management system that can operate in continuous or discrete modes. Uses threading for real-time feedback processing without blocking the main application.

### `endoai/src/intraoperative/preop_fusion.py`
**Python Analysis**: Implements registration algorithms for aligning preoperative imaging data with intraoperative views. Uses SimpleITK for medical image processing and registration tasks. Contains methods for rigid, affine, and deformable registration. Includes classes for displaying preoperative annotations as overlays on live surgical views and fiducial marker detection for registration.

---

## Summary

This file serves as a quick reference for understanding the purpose of each file in the ENDOAI project. It ensures that contributors can easily navigate the project and understand its components.
