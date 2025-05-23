---
name: Pull Request
about: Submit changes to the ENDOAI project
title: "[PR] "
labels: ''
assignees: ''

---

**Description**
A clear and concise description of the changes made in this pull request.

**Related Issues**
List any related issues or feature requests:
- Closes #...

**Type of Change**
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Refactoring
- [ ] Other (please specify):

**Checklist**
- [ ] My code follows the project's coding standards.
- [ ] I have added tests that prove my fix is effective or that my feature works.
- [ ] I have updated the documentation where necessary.
- [ ] I have run the test suite and verified that all tests pass.

**Additional Notes**
Add any additional context or screenshots about the pull request here.

# Debugging a Failed Commit or Pull Request

If you encounter a failure or "X" after pushing your commit, follow these steps to identify and resolve the issue:

---

## 1. Check the CI/CD Logs
- Review the logs from the Continuous Integration (CI) pipeline (e.g., GitHub Actions, Jenkins, etc.).
- Look for specific error messages or failed steps in the pipeline.

---

## 2. Common Issues and Fixes

### **Linting Errors**
- **Cause**: Code does not adhere to the project's linting standards (e.g., PEP8).
- **Fix**:
  ```bash
  flake8 .  # Run the linter locally to identify issues
  ```
  Correct the reported issues and commit the changes.

### **Test Failures**
- **Cause**: One or more tests failed during the CI pipeline.
- **Fix**:
  ```bash
  pytest tests/  # Run the test suite locally
  ```
  Review the failing tests, debug the code, and ensure all tests pass before pushing again.

### **Dependency Issues**
- **Cause**: Missing or incompatible dependencies.
- **Fix**:
  - Ensure all dependencies are installed:
    ```bash
    pip install -r requirements.txt
    ```
  - If a dependency is missing, add it to `requirements.txt` and commit the change.

### **Build Errors**
- **Cause**: Errors during the build process (e.g., Docker, Sphinx documentation).
- **Fix**:
  - For Docker:
    ```bash
    docker build -t endoai .
    ```
  - For Sphinx:
    ```bash
    sphinx-build docs/ docs/_build/
    ```

---

## 3. Reproduce the Issue Locally
- Clone a fresh copy of the repository and run the failing steps locally to reproduce the issue.

---

## 4. Update the Pull Request
- After fixing the issue, push the changes to your branch:
  ```bash
  git add .
  git commit -m "Fix: [describe the fix]"
  git push origin <branch-name>
  ```

---

## 5. Seek Help
- If you're unable to resolve the issue, provide the following details in the pull request:
  - Error logs or screenshots.
  - Steps to reproduce the issue.
  - Any debugging steps you've already tried.

---

## See Also
- [CONTRIBUTING.md](../docs/CONTRIBUTING.md) — Guidelines for contributing to the project.
- [TESTING.md](../docs/TESTING.md) — Instructions for running and writing tests.
