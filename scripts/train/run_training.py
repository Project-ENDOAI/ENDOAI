"""
Script to run the preoperative training pipeline with environment setup, logging, and experiment tracking.

- Sets up logging to both file and console.
- Optionally integrates with Weights & Biases (wandb) for experiment tracking.
- Sets environment variables for reproducibility.
- Calls the main training module (endoai.src.preoperative.train).

Training parameters and experiment setup:
- Logging directory: output/logs/
- Experiment tracking: wandb (if installed)
- PYTHONHASHSEED set to 42 for reproducibility
- Training script called: python -m endoai.src.preoperative.train
- To adjust training parameters (epochs, batch size, etc.), edit endoai/src/preoperative/train.py or pass arguments as needed.
"""

import os
import sys
import logging
from datetime import datetime
import subprocess
import json
import shutil
import time

# Set up logging
log_dir = os.path.join(os.path.dirname(__file__), "..", "output", "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

# Check and install required dependencies
required_packages = ["nibabel", "pydicom", "itk", "nrrd"]
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        logging.info(f"{package} not installed. Attempting to install via pip...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            logging.info(f"{package} installed successfully.")
        except Exception as e:
            logging.error(f"Failed to install {package}: {e}")
            sys.exit(1)

# Set WANDB_API_KEY from VSCode settings.json if available
wandb_api_key = os.environ.get("WANDB_API_KEY")
if not wandb_api_key:
    # Try to load from .vscode/settings.json
    vscode_settings = os.path.join(os.path.dirname(__file__), "..", "..", ".vscode", "settings.json")
    if os.path.exists(vscode_settings):
        with open(vscode_settings, "r") as f:
            settings = json.load(f)
            envs = settings.get("terminal.integrated.env.linux", {})
            wandb_api_key = envs.get("WANDB_API_KEY")
            if wandb_api_key:
                os.environ["WANDB_API_KEY"] = wandb_api_key

# Try to import wandb, install if not present
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    logging.info("Weights & Biases (wandb) not installed. Attempting to install via pip...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
        import wandb
        WANDB_AVAILABLE = True
        logging.info("wandb installed successfully.")
    except Exception as e:
        WANDB_AVAILABLE = False
        logging.error(f"Failed to install wandb: {e}")

logging.info("Starting preoperative training pipeline.")

# Remove wandb runs older than 2 days
wandb_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "wandb"))
if os.path.isdir(wandb_dir):
    now = time.time()
    for entry in os.listdir(wandb_dir):
        entry_path = os.path.join(wandb_dir, entry)
        if os.path.isdir(entry_path) and entry.startswith("run-"):
            mtime = os.path.getmtime(entry_path)
            # 2 days = 172800 seconds
            if now - mtime > 172800:
                try:
                    shutil.rmtree(entry_path)
                    logging.info(f"Removed old wandb run: {entry_path}")
                except Exception as e:
                    logging.warning(f"Could not remove {entry_path}: {e}")

# Optionally set up experiment tracking with Weights & Biases
if WANDB_AVAILABLE:
    wandb.init(
        project="endoai-preoperative",
        name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        notes="Automated run via scripts/run_training.py"
    )
    logging.info("Weights & Biases tracking enabled.")
else:
    logging.info("Weights & Biases (wandb) not available. Skipping experiment tracking.")

# Optionally set environment variables (example)
os.environ["PYTHONHASHSEED"] = "42"

# Before calling the training module, ensure the public dataset is downloaded and extracted
dataset_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "Task04_Hippocampus")
images_dir = os.path.join(dataset_dir, "imagesTr")
labels_dir = os.path.join(dataset_dir, "labelsTr")

if not (os.path.isdir(images_dir) and os.path.isdir(labels_dir)):
    logging.info("Public dataset not found. Downloading and extracting Task04_Hippocampus to data/raw/ ...")
    # Dynamically import download_and_extract using an absolute path
    import importlib.util

    download_script_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "download_public_dataset.py")
    )
    spec = importlib.util.spec_from_file_location("download_public_dataset", download_script_path)
    download_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(download_module)
    url = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task04_Hippocampus.tar"
    dest_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
    download_module.download_and_extract(url, dest_dir)
    # Check again after extraction
    if not (os.path.isdir(images_dir) and os.path.isdir(labels_dir)):
        logging.error("Dataset extraction failed or folders not found. Please check data/raw/Task04_Hippocampus.")
        sys.exit(1)

# Call the training module
exit_code = os.system("python -m endoai.src.preoperative.train")

if exit_code == 0:
    logging.info("Training completed successfully.")
    
    # Post-training steps
    logging.info("Starting model evaluation...")
    
    # 1. Evaluate on validation set
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "lesion_segmentation.pth"))
    if os.path.exists(model_path):
        logging.info(f"Found trained model at {model_path}")
        
        # Run the evaluation script
        try:
            logging.info("Running evaluation script...")
            eval_script_path = os.path.join(os.path.dirname(__file__), "..", "..", "endoai", "validation", "evaluate.py")
            
            if os.path.exists(eval_script_path):
                eval_cmd = f"{sys.executable} {eval_script_path} --model-path={model_path} --data-dir={os.path.dirname(images_dir)}"
                eval_exit_code = os.system(eval_cmd)
                
                # Log performance metrics to wandb if available
                if WANDB_AVAILABLE and eval_exit_code == 0:
                    try:
                        # Log model artifact to wandb for versioning
                        artifact = wandb.Artifact(
                            name="lesion_segmentation_model", 
                            type="model"
                        )
                        artifact.add_file(model_path)
                        wandb.log_artifact(artifact)
                        
                        # You can also log metrics directly from evaluation results
                        # wandb.log({"val_dice": eval_dice_score, "val_loss": eval_loss})
                    except Exception as e:
                        logging.error(f"Error logging artifact to wandb: {e}")
            else:
                logging.warning(f"Evaluation script not found at {eval_script_path}")
        except Exception as e:
            logging.error(f"Error during model evaluation: {e}")
    else:
        logging.warning(f"Trained model not found at {model_path}")
    
    # Suggest next steps to the user
    print("\n==== NEXT STEPS ====")
    print("1. View model performance in the wandb dashboard")
    print("2. Test the model with new data using scripts/inference/predict.py")
    print("3. Fine-tune hyperparameters to improve performance")
    print("4. Export the model for deployment")
    print("====================\n")
else:
    logging.error(f"Training failed with exit code {exit_code}.")

if WANDB_AVAILABLE:
    wandb.finish()
