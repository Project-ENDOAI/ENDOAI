import torch
import importlib
import subprocess
import sys

# List of modules to verify
modules = [
    "numpy", "pandas", "scikit-learn", "matplotlib", "jupyterlab", "torch",
    "opencv-python", "pydicom", "nibabel", "tqdm", "detectron2", "SimpleITK",
    "monai", "plotly", "scipy"
]

def check_cuda():
    print("Checking CUDA availability...")
    if torch.cuda.is_available():
        print(f"CUDA is available. Device: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Please ensure your system has a compatible GPU and drivers installed.")

def check_torch():
    print("Checking PyTorch installation...")
    try:
        print(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            print("PyTorch is configured to use CUDA.")
        else:
            print("PyTorch is installed but not configured to use CUDA.")
    except Exception as e:
        print(f"Error checking PyTorch: {e}")

def install_pytorch_with_cuda():
    print("Installing PyTorch with CUDA support...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "torchaudio", 
            "--index-url", "https://download.pytorch.org/whl/cu118"
        ])
        print("PyTorch with CUDA support installed successfully.")
    except subprocess.CalledProcessError:
        print("Failed to install PyTorch with CUDA support. Please install it manually.")

def install_module(module):
    print(f"Attempting to install {module}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", module])
        print(f"{module} installed successfully.")
    except subprocess.CalledProcessError:
        print(f"Failed to install {module}. Please install it manually.")

def install_system_dependencies():
    print("Checking and installing system-level dependencies...")
    try:
        subprocess.check_call(["sudo", "apt", "install", "-y", "python-is-python3"])
        print("System-level dependency 'python-is-python3' installed successfully.")
    except subprocess.CalledProcessError:
        print("Failed to install 'python-is-python3'. Please install it manually.")

def check_modules():
    print("Checking installed modules...")
    all_verified = True
    for module in modules:
        try:
            imported_module = importlib.import_module(module)
            # Check version if the module has a __version__ attribute
            version = getattr(imported_module, "__version__", "Unknown version")
            print(f"{module} is installed and working. Version: {version}")
        except ImportError:
            print(f"{module} is not installed or not working.")
            install_module(module)
            all_verified = False
        except Exception as e:
            print(f"An error occurred while checking {module}: {e}")
            all_verified = False
    return all_verified

def main():
    print("Starting environment check...\n")

    install_system_dependencies()
    print()  # Add spacing for better readability
    install_pytorch_with_cuda()
    print()  # Add spacing for better readability
    check_cuda()
    print()  # Add spacing for better readability
    check_torch()
    print()  # Add spacing for better readability
    all_modules_verified = check_modules()

    if all_modules_verified:
        print("\nAll modules are installed and working correctly.")
    else:
        print("\nSome modules are missing or not working. Please review the output above and ensure all modules are installed.")

if __name__ == "__main__":
    main()
