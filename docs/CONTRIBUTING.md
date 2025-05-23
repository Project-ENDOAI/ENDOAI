# Contributing to ENDOAI

Thank you for your interest in contributing to the ENDOAI project! This document provides guidelines for contributing to the project.

## How to Contribute

1. **Fork the Repository**:
   - Create your own fork of the repository on GitHub.

2. **Clone the Repository**:
   - Clone your fork to your local machine:
     ```bash
     git clone https://github.com/your-username/ENDOAI.git
     ```

3. **Set Up the Environment**:
   - Install the required dependencies:
     ```bash
     pip install -r requirements.txt
     ```

4. **Create a Branch**:
   - Create a new branch for your feature or bug fix:
     ```bash
     git checkout -b feature/your-feature-name
     ```

5. **Make Changes**:
   - Add your changes and ensure they follow the project's coding standards.
   - Add tests for new features or bug fixes.

6. **Run Tests**:
   - Run the test suite to ensure everything works as expected:
     ```bash
     pytest tests/
     ```

7. **Commit Changes**:
   - Commit your changes with a descriptive commit message:
     ```bash
     git commit -m "Add feature: your-feature-name"
     ```

8. **Push Changes**:
   - Push your branch to your fork:
     ```bash
     git push origin feature/your-feature-name
     ```

9. **Create a Pull Request**:
   - Open a pull request on the main repository and describe your changes.

## Guidelines

- Follow the coding standards outlined in `COPILOT.md`.
- Write clear and concise commit messages.
- Ensure all new code is covered by tests.
- Update documentation if your changes affect existing functionality.

## See Also

- [../README.md](../README.md) — Project-level documentation.
- [TESTING.md](TESTING.md) — Guidelines for writing and running tests.
