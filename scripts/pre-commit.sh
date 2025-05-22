#!/bin/bash
# Pre-commit hook script for ENDOAI project

# Exit on any error
set -e

echo "Running pre-commit checks..."

# 1. Format code with black
if command -v black &> /dev/null; then
    echo "Formatting Python files with black..."
    black --check .
else
    echo "Warning: black not installed. Skipping formatting check."
fi

# 2. Lint with flake8
if command -v flake8 &> /dev/null; then
    echo "Linting Python files with flake8..."
    flake8 .
else
    echo "Warning: flake8 not installed. Skipping lint check."
fi

# 3. Run tests (if pytest is available)
if command -v pytest &> /dev/null; then
    echo "Running tests with pytest..."
    pytest
else
    echo "Warning: pytest not installed. Skipping tests."
fi

# 4. Check for secrets (if detect-secrets is available)
if command -v detect-secrets &> /dev/null; then
    echo "Checking for secrets with detect-secrets..."
    detect-secrets scan > /dev/null
else
    echo "Warning: detect-secrets not installed. Skipping secrets check."
fi

echo "Pre-commit checks completed."
