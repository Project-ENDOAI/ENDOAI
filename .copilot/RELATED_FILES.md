# Copilot Related Files Guide

This file helps Copilot and developers understand how files in this project are related.

## What are Related Files?

- Files that are linked by imports, usage, or naming conventions (e.g., `module.py` and `test_module.py`).
- Configuration files that affect code (e.g., `.env`, `pyproject.toml`, `requirements.txt`).
- Data files or assets used by scripts or modules.

## How to Use

- Keep related files in the same directory or use clear naming conventions.
- Reference related files in docstrings or comments.
- When adding a new file, update this document if it is closely tied to another file.

## Examples

- `src/preoperative/mri_segmentation.py` ↔ `tests/test_mri_segmentation.py`
- `src/preoperative/data_preparation.py` ↔ `src/preoperative/preprocessing.py`
- `scripts/git_automation.py` ↔ `.vscode/settings.json`
- `src/inference/preoperative_inference.py` ↔ `models/trained_model.pth`
- `src/reporting/visualize_results.py` ↔ `reports/visualizations/`

---

## See Also

- [SYMBOLS.md](SYMBOLS.md) — Guide for naming and documenting symbols.
- [PROBLEMS.md](PROBLEMS.md) — Guide for handling and documenting problems.
- [INSTRUCTIONS.md](INSTRUCTIONS.md) — Additional instructions for Copilot and AI assistants.
