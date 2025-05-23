# Project Goals and Source Folder Overview

This document outlines the primary goals of the ENDOAI project and provides a detailed overview of the Python files in the `src` folder. It is intended to help Copilot and contributors stay aligned with the project's mission and maintain focus on its objectives.

---

## Project Goals

1. **Enhance Diagnosis and Treatment of Endometriosis**:
   - Develop AI-driven tools to assist in the diagnosis and treatment planning for endometriosis.
   - Leverage medical imaging and sensor data to provide actionable insights.

2. **Preoperative Planning**:
   - Create tools for analyzing preoperative imaging data to assist in surgical planning.
   - Implement segmentation models for identifying critical structures in medical images.

3. **Intraoperative Guidance**:
   - Develop real-time tools for integrating sensor data during surgery.
   - Provide visual and haptic feedback to improve surgical precision and safety.

4. **Postoperative Analysis**:
   - Build tools for analyzing postoperative outcomes and generating reports.
   - Use data to refine surgical techniques and improve patient recovery.

5. **Scalability and Reproducibility**:
   - Ensure all tools and pipelines are modular, reusable, and well-documented.
   - Focus on reproducibility to enable deployment in clinical settings.

---

## Source Folder Overview

### `src/preoperative/`

- **`model.py`**:
  - **Purpose**: Defines the neural network architecture (e.g., UNet) for medical image segmentation.
  - **Details**: Implements an encoder-decoder structure with skip connections for volumetric segmentation tasks.

- **`preprocessing_pipeline.py`**:
  - **Purpose**: Handles data preprocessing, including loading, resizing, and normalizing medical images.
  - **Details**: Uses MONAI for medical imaging preprocessing and supports data augmentation.

- **`train.py`**:
  - **Purpose**: Implements the training loop for the segmentation model.
  - **Details**: Includes data loading, loss calculation, model optimization, and checkpoint saving.

- **`__init__.py`**:
  - **Purpose**: Initializes the `preoperative` module for modular imports.

---

### `src/intraoperative/`

- **`sensor_fusion.py`**:
  - **Purpose**: Integrates data from multiple sensors in real-time during surgery.
  - **Details**: Implements feature-level and decision-level fusion strategies for multimodal data.

- **`visual_processing.py`**:
  - **Purpose**: Processes and enhances visual data from surgical cameras.
  - **Details**: Includes image enhancement, segmentation, and depth estimation using stereoscopic views.

- **`force_feedback.py`**:
  - **Purpose**: Processes force and tactile sensor data to provide haptic feedback.
  - **Details**: Includes tissue characterization and safety monitoring for excessive pressure.

- **`preop_fusion.py`**:
  - **Purpose**: Aligns preoperative imaging data with intraoperative sensor data.
  - **Details**: Implements registration algorithms for rigid, affine, and deformable transformations.

- **`__init__.py`**:
  - **Purpose**: Initializes the `intraoperative` module for modular imports.

---

### `src/postoperative/`

- **`store_notes.py`**:
  - **Purpose**: Saves and manages postoperative notes for patients.
  - **Details**: Stores notes in JSON format for easy retrieval and analysis.

- **`analyze_outcomes.py`**:
  - **Purpose**: Analyzes postoperative outcomes and generates summary statistics.
  - **Details**: Processes CSV data to provide insights into recovery and complications.

- **`__init__.py`**:
  - **Purpose**: Initializes the `postoperative` module for modular imports.

---

### `src/inference/`

- **`preoperative_inference.py`**:
  - **Purpose**: Runs inference on preoperative imaging data using trained models.
  - **Details**: Loads patient data, preprocesses images, and generates predictions.

- **`__init__.py`**:
  - **Purpose**: Initializes the `inference` module for modular imports.

---

### `src/decision_support/`

- **`risk_assessment.py`**:
  - **Purpose**: Calculates risk scores for patients based on clinical data.
  - **Details**: Uses weighted sums of clinical features to generate risk scores.

- **`decision_tree_support.py`**:
  - **Purpose**: Implements decision tree-based support systems for clinical workflows.
  - **Details**: Trains and visualizes decision trees for classification tasks.

- **`patient_prioritization.py`**:
  - **Purpose**: Prioritizes patients based on risk scores and other criteria.
  - **Details**: Sorts and ranks patients to assist in resource allocation.

- **`__init__.py`**:
  - **Purpose**: Initializes the `decision_support` module for modular imports.

---

### `src/reporting/`

- **`generate_report.py`**:
  - **Purpose**: Generates detailed reports based on model predictions and metadata.
  - **Details**: Combines predictions and metadata into summary reports.

- **`visualize_results.py`**:
  - **Purpose**: Visualizes model predictions and performance metrics.
  - **Details**: Creates plots for prediction distributions and confusion matrices.

- **`__init__.py`**:
  - **Purpose**: Initializes the `reporting` module for modular imports.

---

## Guidelines for Copilot

- **Stay Aligned with Project Goals**:
  - Focus on enhancing diagnosis, treatment, and analysis for endometriosis.
  - Prioritize modular, reusable, and well-documented code.

- **Follow the Source Folder Structure**:
  - Ensure new files are placed in the appropriate subfolder.
  - Update this document whenever new files are added or existing files are modified.

- **Maintain Consistency**:
  - Use consistent naming conventions and follow the project's coding standards.
  - Add docstrings and comments to all new functions and classes.

---

## See Also

- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) — Guide for maintaining the project structure.
- [RELATED_FILES.md](RELATED_FILES.md) — Guide for understanding file relationships.
- [SYMBOLS.md](SYMBOLS.md) — Guide for naming and documenting symbols.
