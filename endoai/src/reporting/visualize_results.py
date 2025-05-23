"""
Visualize Results Module

This script provides utilities for visualizing model predictions and performance metrics.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# Check if seaborn is installed, if not, install it
try:
    import seaborn as sns
except ImportError as e:
    print("Seaborn is not installed. Installing it now...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "seaborn"])
    import seaborn

def plot_prediction_distribution(predictions, output_dir="../reports/visualizations"):
    """
    Plot the distribution of predictions.

    Args:
        predictions (pd.Series): Series containing model predictions.
        output_dir (str): Directory to save the plot.

    Returns:
        str: Path to the saved plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "prediction_distribution.png")
    plt.figure(figsize=(8, 6))
    sns.countplot(x=predictions)
    plt.title("Prediction Distribution")
    plt.xlabel("Prediction")
    plt.ylabel("Count")
    plt.savefig(plot_path)
    plt.close()
    print(f"Prediction distribution plot saved to {plot_path}")
    return plot_path

def main():
    """
    Main function to demonstrate result visualization.
    """
    # Example data
    predictions = pd.Series(["High Risk", "Low Risk", "High Risk", "Medium Risk", "Low Risk"])
    plot_prediction_distribution(predictions)

if __name__ == "__main__":
    main()
