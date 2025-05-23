"""
Find Issues Script

This script automates the process of identifying common issues in the project, such as test failures, dependency issues, and build errors.
"""

import os
import subprocess

def run_tests():
    """
    Run the test suite and return the results.
    """
    print("Running tests...")
    try:
        result = subprocess.run(["pytest", "tests/"], capture_output=True, text=True, check=False)
        print(result.stdout)
        if "collected 0 items" in result.stdout:
            print("No tests found. Ensure test files are present in the 'tests/' directory and follow the naming convention 'test_<module_name>.py'.")
            return False
        if result.returncode != 0:
            print("Test failures detected.")
            return False
        print("All tests passed.")
        return True
    except FileNotFoundError:
        print("pytest is not installed. Please install it using 'pip install pytest'.")
        return False

def check_dependencies():
    """
    Check if all dependencies are installed.
    """
    print("Checking dependencies...")
    try:
        result = subprocess.run(["pip", "check"], capture_output=True, text=True, check=False)
        print(result.stdout)
        if "No broken requirements found." not in result.stdout:
            print("Dependency issues detected.")
            return False
        print("All dependencies are installed and compatible.")
        return True
    except FileNotFoundError:
        print("pip is not installed or not in PATH.")
        return False

def build_docker():
    """
    Attempt to build the Docker image.
    """
    print("Building Docker image...")
    try:
        result = subprocess.run(["docker", "build", "-t", "endoai", "."], capture_output=True, text=True, check=False)
        print(result.stdout)
        if result.returncode != 0:
            print("Docker build failed.")
            return False
        print("Docker image built successfully.")
        return True
    except FileNotFoundError:
        print("Docker is not installed or not in PATH.")
        return False

def build_sphinx():
    """
    Attempt to build the Sphinx documentation.
    """
    print("Building Sphinx documentation...")
    try:
        result = subprocess.run(["sphinx-build", "docs/", "docs/_build/"], capture_output=True, text=True, check=False)
        print(result.stdout)
        if result.returncode != 0:
            print("Sphinx build failed.")
            return False
        print("Sphinx documentation built successfully.")
        return True
    except FileNotFoundError:
        print("Sphinx is not installed or not in PATH.")
        return False

def main():
    """
    Main function to run all checks.
    """
    print("Starting issue detection...")
    tests_passed = run_tests()
    dependencies_ok = check_dependencies()
    docker_built = build_docker()
    sphinx_built = build_sphinx()

    if tests_passed and dependencies_ok and docker_built and sphinx_built:
        print("\nNo issues detected. Everything is working as expected.")
    else:
        print("\nIssues detected. Please review the logs above for details.")

if __name__ == "__main__":
    main()
