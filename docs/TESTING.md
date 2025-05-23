# Testing Guidelines

This document provides guidelines for writing and running tests in the ENDOAI project.

## Purpose

Testing ensures that the code is reliable, maintainable, and free of regressions. All new features and bug fixes must include appropriate tests.

## Types of Tests

1. **Unit Tests**:
   - Test individual functions or classes in isolation.
   - Use mock data where necessary.

2. **Integration Tests**:
   - Test how different modules interact with each other.
   - Use real or simulated datasets.

3. **End-to-End Tests**:
   - Test the entire workflow, from data preprocessing to model inference.

## Running Tests

1. **Install Test Dependencies**:
   - Ensure all dependencies are installed:
     ```bash
     pip install -r requirements.txt
     ```

2. **Run All Tests**:
   - Use `pytest` to run the test suite:
     ```bash
     pytest tests/
     ```

3. **Run Specific Tests**:
   - Run a specific test file:
     ```bash
     pytest tests/test_preprocessing_pipeline.py
     ```

4. **Generate Coverage Report**:
   - Use `pytest-cov` to generate a test coverage report:
     ```bash
     pytest --cov=endoai tests/
     ```

## Writing Tests

- Place all test files in the `tests/` directory.
- Name test files as `test_<module_name>.py`.
- Use descriptive names for test functions.
- Use `pytest` fixtures for reusable test setups.

## See Also

- [CONTRIBUTING.md](CONTRIBUTING.md) — Guidelines for contributing to the project.
- [../README.md](../README.md) — Project-level documentation.
