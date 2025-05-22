"""
List all Python files in the project and print their paths to stdout.
"""

import os

def list_python_files(root_dir="."):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip hidden directories and virtual environments
        parts = dirpath.split(os.sep)
        if any(part.startswith(".") for part in parts) or "venv" in parts or "__pycache__" in parts:
            continue
        for filename in filenames:
            if filename.endswith(".py"):
                print(os.path.join(dirpath, filename))

def list_python_files_and_contents(root_dir=".", output_file="output/python_files_output.txt"):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as out:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # Skip hidden directories and virtual environments
            parts = dirpath.split(os.sep)
            if any(part.startswith(".") for part in parts) or "venv" in parts or "__pycache__" in parts:
                continue
            for filename in filenames:
                if filename.endswith(".py"):
                    filepath = os.path.join(dirpath, filename)
                    out.write(f"File: {filepath}\n")
                    out.write("-" * 30 + "\n")
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            out.write(f.read() + "\n")
                    except Exception as e:
                        out.write(f"Could not read file: {e}\n")
                    out.write("-" * 30 + "\n")

if __name__ == "__main__":
    # Start from the project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_path = os.path.join(project_root, "output", "python_files_output.txt")
    list_python_files_and_contents(project_root, output_file=output_path)
