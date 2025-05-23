# tests

This directory contains unit tests, integration tests, and other testing utilities for the ENDOAI project.

## Purpose

- Ensure the correctness and reliability of the codebase.
- Provide automated testing for all major components, including preprocessing, models, training, and utilities.
- Facilitate continuous integration and deployment workflows.

## Structure

- `test_basic.py` — Basic tests to verify the environment and setup.
- `test_preprocessing.py` — Tests for data preprocessing pipelines.
- `test_models.py` — Tests for model architectures and initialization.
- `test_training.py` — Tests for training loops and loss functions.
- `test_utils.py` — Tests for utility functions and scripts.

## Guidelines

- Use `pytest` as the primary testing framework.
- Follow the naming convention `test_<module>.py` for test files.
- Write descriptive test names and include docstrings for clarity.
- Mock external dependencies (e.g., file I/O, API calls) to ensure tests are isolated and reproducible.

## Running Tests

To run all tests:
```bash
pytest
```

To run a specific test file:
```bash
pytest tests/test_<module>.py
```

To generate a coverage report:
```bash
pytest --cov=endoai --cov-report=html
```

## Best Practices

- Add tests for every new feature or bug fix.
- Ensure high test coverage for critical components.
- Use fixtures to set up reusable test data and configurations.
- Avoid hardcoding paths; use relative paths or temporary directories.

## See Also

- [../README.md](../README.md) — Project-level documentation.
- [../dev-requirements.txt](../dev-requirements.txt) — Development dependencies for testing.
