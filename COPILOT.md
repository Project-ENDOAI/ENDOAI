---
applyTo: '**'
---

# Copilot Coding Standards and Preferences

This document defines coding standards, domain knowledge, and project-specific preferences for GitHub Copilot and other AI coding assistants.  
**Apply these guidelines to all code in this repository.**

---

## 1. General Coding Standards

- **Language:** Python 3.8+ (unless otherwise specified)
- **Formatting:** Use [Black](https://black.readthedocs.io/) for code formatting.
- **Linting:** Code must pass [flake8](https://flake8.pycqa.org/) and [pylint](https://pylint.org/).
- **Naming Conventions:**
  - Variables/functions: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_SNAKE_CASE`
- **Type Hints:** Use type annotations for all function signatures.
- **Docstrings:** Use [Google-style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) or [NumPy-style](https://numpydoc.readthedocs.io/en/latest/format.html) docstrings for all public functions, classes, and modules.
- **Imports:** Group standard library, third-party, and local imports separately. Use absolute imports.
- **Error Handling:** Use exceptions, not return codes. Log errors where appropriate.

---

## 2. Project Structure

- Each Python package directory must contain an `__init__.py`.
- Each directory (except for virtual environments and cache folders) should have a `README.md` describing its purpose.
- Place all source code in the `src/` directory, organized by domain (e.g., `preoperative`, `intraoperative`).
- Use a `tests/` directory for unit and integration tests.
- Use a `notebooks/` directory for Jupyter notebooks and experimentation.

---

## 3. Domain Knowledge

- **Medical Imaging:** Use [MONAI](https://monai.io/) for medical image processing and PyTorch for deep learning.
- **Segmentation:** Prefer U-Net or Swin-UNet architectures for segmentation tasks.
- **Data Formats:** Use NIfTI (`.nii.gz`) for MRI/CT, DICOM for raw medical images, and standard formats for annotations.
- **Reproducibility:** Set random seeds where possible and document all data preprocessing steps.

---

## 4. Preferences

- **Comments:** Write clear, concise comments explaining non-obvious logic.
- **Logging:** Use the `logging` module for non-trivial scripts and applications.
- **Configuration:** Use `.env` files or config files for environment-specific settings.
- **Dependencies:** List all dependencies in `requirements.txt` and keep them up to date.
- **Version Control:** Do not commit secrets, large data, or environment files. Use `.gitignore` appropriately.

---

## 5. AI Assistant Instructions

- Prefer concise, readable code over clever or overly compact solutions.
- When in doubt, add docstrings and comments.
- Suggest test cases for new modules or functions.
- When generating new files, follow the project structure above.
- Always check for existing code before suggesting new code.
- If a coding standard or preference is unclear, prefer explicitness and maintainability.

---

## 6. Example Directory Structure

```
endoai/
├── src/
│   ├── preoperative/
│   ├── intraoperative/
│   └── ...
├── tests/
├── notebooks/
├── data/
├── models/
├── requirements.txt
├── .env.example
└── README.md
```

---

## 7. References

- [PEP8](https://www.python.org/dev/peps/pep-0008/)
- [Black](https://black.readthedocs.io/)
- [MONAI](https://monai.io/)
- [PyTorch](https://pytorch.org/)