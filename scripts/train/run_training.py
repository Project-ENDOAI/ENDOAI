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

# Call the training module
exit_code = os.system("python -m endoai.src.preoperative.train")

if WANDB_AVAILABLE:
    wandb.finish()

if exit_code == 0:
    logging.info("Training completed successfully.")
else:
    logging.error(f"Training failed with exit code {exit_code}.")
