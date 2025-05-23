"""
Script to set up the project directory structure for ENDOAI.
Creates output directories for models, logs, visualizations, etc.
"""

import os
import argparse

def create_directory_structure(base_dir):
    """
    Create the directory structure for ENDOAI project outputs.
    
    Args:
        base_dir: Base directory to create the structure in
    """
    # Define the directory structure
    structure = {
        "output": {
            "models": {},
            "logs": {},
            "visualizations": {
                "train_results": {},
                "validation_results": {},
                "inference_results": {
                    "hippocampus_001": {},
                },
                "case_comparisons": {},
            },
            "exported_models": {}
        }
    }
    
    # Create directories
    def create_recursive(path, struct):
        for name, substruct in struct.items():
            dir_path = os.path.join(path, name)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                print(f"Created directory: {dir_path}")
            else:
                print(f"Directory already exists: {dir_path}")
            
            if substruct:  # If there are subdirectories
                create_recursive(dir_path, substruct)
    
    create_recursive(base_dir, structure)
    
    # Create README files to explain the purpose of each directory
    readme_content = {
        "output": "This directory contains all outputs from the ENDOAI project.",
        "output/models": "This directory contains saved trained models.",
        "output/logs": "This directory contains training and evaluation logs.",
        "output/visualizations": "This directory contains visualization outputs.",
        "output/visualizations/train_results": "Training-related visualizations (loss curves, etc.)",
        "output/visualizations/validation_results": "Validation metrics and visualizations.",
        "output/visualizations/inference_results": "Inference visualizations for individual cases.",
        "output/visualizations/inference_results/hippocampus_001": "Case-specific visualizations for hippocampus_001.",
        "output/visualizations/case_comparisons": "Comparisons between multiple cases.",
        "output/exported_models": "Models exported for deployment."
    }
    
    for path, content in readme_content.items():
        readme_path = os.path.join(base_dir, path, "README.md")
        if not os.path.exists(readme_path):
            with open(readme_path, "w") as f:
                f.write(f"# {os.path.basename(path)}\n\n{content}")
            print(f"Created README: {readme_path}")

def main():
    parser = argparse.ArgumentParser(description='Set up ENDOAI project directory structure')
    parser.add_argument('--base-dir', type=str, default=os.getcwd(),
                       help='Base directory to create the structure in (default: current directory)')
    
    args = parser.parse_args()
    create_directory_structure(args.base_dir)
    print("Directory structure creation completed.")

if __name__ == "__main__":
    main()
