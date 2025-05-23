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

- **Problem:** `ValueError: Mismatched input dimensions`
  - **Solution:** Check the preprocessing pipeline to ensure input dimensions match the model's expected input.

## See Also

- [RELATED_FILES.md](RELATED_FILES.md) — Guide for understanding file relationships.
- [SYMBOLS.md](SYMBOLS.md) — Guide for naming and documenting symbols.
- [INSTRUCTIONS.md](INSTRUCTIONS.md) — Additional instructions for Copilot and AI assistants.
