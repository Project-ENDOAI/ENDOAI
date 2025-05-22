"""
Recursively add README.md to all directories and __init__.py to all directories
under the specified root (default: project root), except hidden, venv, and __pycache__ directories.
"""

import os

def ensure_readme_and_init(root_dir="."):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip hidden directories and virtual environments
        parts = dirpath.split(os.sep)
        if any(part.startswith(".") for part in parts) or "venv" in parts or "__pycache__" in parts:
            continue

        # Add README.md to all directories (including empty ones)
        readme_path = os.path.join(dirpath, "README.md")
        if not os.path.exists(readme_path):
            with open(readme_path, "w") as f:
                f.write(f"# {os.path.basename(dirpath) or 'Root'}\n")
            print(f"Created {readme_path}")

        # Add __init__.py to all directories (including root)
        init_path = os.path.join(dirpath, "__init__.py")
        if not os.path.exists(init_path):
            with open(init_path, "w") as f:
                f.write("# Auto-generated to make this directory a Python package.\n")
            print(f"Created {init_path}")

if __name__ == "__main__":
    # Only operate on folders inside the project directory (not the whole system)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    for folder in os.listdir(project_root):
        folder_path = os.path.join(project_root, folder)
        if os.path.isdir(folder_path) and not folder.startswith(".") and folder not in ("venv", "__pycache__"):
            ensure_readme_and_init(folder_path)
    # Flush file system buffers to disk (for Linux/Unix)
    try:
        os.sync()
    except AttributeError:
        # os.sync() may not be available on all platforms; ignore if not present
        pass

    # Notify VSCode to refresh file explorer (if running in VSCode)
    # This is a hint: VSCode will auto-detect new files, but you can force a refresh:
    print("All README.md and __init__.py files created.")
    print("If you do not see them in VSCode, right-click the folder in the Explorer and select 'Refresh', or restart VSCode.")
