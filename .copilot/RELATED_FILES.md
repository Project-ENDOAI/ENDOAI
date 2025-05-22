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

---

<!-- filepath: /home/kevin/Projects/Development/ENDOAI/.copilot/SYMBOLS.md -->
# Copilot Symbols Guide

This file provides guidance on naming and documenting symbols (functions, classes, variables) for Copilot.

## Best Practices

- Use descriptive, consistent names for all symbols.
- Add type hints and docstrings to all public functions and classes.
- Avoid ambiguous or single-letter variable names except in short loops.
- Group related functions and classes in the same module.

## Example

```python
def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize and preprocess a medical image.
    """
    ...
```

---

<!-- filepath: /home/kevin/Projects/Development/ENDOAI/.copilot/PROBLEMS.md -->
# Copilot Problems Guide

This file explains how to handle and document problems (errors, warnings, test failures) in the project.

## What are Problems?

- Issues detected by your IDE, linter, or test runner (e.g., syntax errors, failed tests, warnings).
- Problems Copilot can help fix or explain.

## How to Use

- Regularly check the "Problems" panel in your IDE.
- Document recurring or complex problems here for team awareness.
- When fixing a problem, add a note or link to the solution if helpful.

## Example

- **Problem:** `ModuleNotFoundError: No module named 'monai'`
  - **Solution:** Add `monai[all]` to `requirements.txt` and run `pip install -r requirements.txt`.

---

<!-- filepath: /home/kevin/Projects/Development/ENDOAI/.copilot/INSTRUCTIONS.md -->
# Copilot Instructions

This file provides additional instructions and preferences for Copilot and AI assistants.

- Follow all coding standards in `COPILOT.md`.
- Use the context and related files described in this folder.
- Prefer explicit, maintainable code.
- When in doubt, add comments and docstrings.
- Ask for clarification if requirements are ambiguous.

---

# How to Use This Folder

- Place all Copilot and AI assistant guidance files here.
- Update these files as the project evolves.
- Reference this folder in your main `COPILOT.md` and `COPILOT_CONTEXT.md`.

---

# See Also

- [COPILOT.md](../COPILOT.md)
- [COPILOT_CONTEXT.md](../COPILOT_CONTEXT.md)
