import subprocess
import sys

# List of required modules and their installation commands
modules = {
    "numpy": "pip install numpy",
    "pandas": "pip install pandas",
    "scikit-learn": "pip install scikit-learn",
    "matplotlib": "pip install matplotlib",
    "jupyterlab": "pip install jupyterlab",
    "torch": "pip install torch torchvision",
    "opencv-python": "pip install opencv-python",
    "pydicom": "pip install pydicom",
    "nibabel": "pip install nibabel",
    "tqdm": "pip install tqdm",
    "detectron2": "pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu117/torch2.0/index.html",
    "SimpleITK": "pip install SimpleITK",
    "monai": "pip install monai[all]",
    "plotly": "pip install plotly",
    "scipy": "pip install scipy"
}

# Function to install a module
def install_module(module, command):
    try:
        print(f"Installing {module}...")
        subprocess.check_call(command, shell=True)
        print(f"{module} installed successfully.")
    except subprocess.CalledProcessError:
        print(f"Error installing {module}. Retrying with alternative methods...")
        if module == "detectron2":
            print("Ensure you have the correct CUDA version or use 'cpu' if no GPU is available.")
        else:
            print(f"Please check the installation command for {module}.")

# Function to verify a module
def verify_module(module):
    try:
        __import__(module)
        print(f"{module} verified successfully.")
        return True
    except ImportError:
        print(f"Verification failed for {module}. Please check the installation.")
        return False

# Main installation process
def main():
    print("Starting installation of required modules...")
    all_verified = True
    for module, command in modules.items():
        install_module(module, command)
        if not verify_module(module):
            all_verified = False

    if all_verified:
        print("\nAll modules are installed and verified successfully. The environment is ready to use!")
    else:
        print("\nSome modules failed verification. Please review the output above and resolve any issues.")

if __name__ == "__main__":
    main()
