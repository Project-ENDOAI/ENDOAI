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

## Guidelines for Naming

- **Functions**: Use verbs or verb phrases (e.g., `load_data`, `train_model`).
- **Classes**: Use nouns or noun phrases (e.g., `ImageProcessor`, `RiskCalculator`).
- **Variables**: Use descriptive names that reflect their purpose (e.g., `patient_data`, `risk_score`).

## See Also

- [RELATED_FILES.md](RELATED_FILES.md) — Guide for understanding file relationships.
- [PROBLEMS.md](PROBLEMS.md) — Guide for handling and documenting problems.
- [INSTRUCTIONS.md](INSTRUCTIONS.md) — Additional instructions for Copilot and AI assistants.
