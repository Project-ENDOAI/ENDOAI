"""
Generate Report Module

This script generates detailed reports based on model predictions and metadata.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

def generate_summary_report(predictions, metadata, output_dir="../reports"):
    """
    Generate a summary report combining predictions and metadata.

    Args:
        predictions (pd.DataFrame): DataFrame containing model predictions.
        metadata (pd.DataFrame): DataFrame containing patient metadata.
        output_dir (str): Directory to save the generated report.

    Returns:
        str: Path to the generated report file.
    """
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "summary_report.csv")
    report = pd.concat([metadata, predictions], axis=1)
    report.to_csv(report_path, index=False)
    print(f"Summary report saved to {report_path}")
    return report_path

def main():
    """
    Main function to demonstrate report generation.
    """
    # Example data
    predictions = pd.DataFrame({
        "patient_id": [1, 2, 3],
        "prediction": ["High Risk", "Low Risk", "Medium Risk"]
    })
    metadata = pd.DataFrame({
        "patient_id": [1, 2, 3],
        "age": [45, 60, 30],
        "bmi": [25.4, 28.7, 22.1]
    })

    generate_summary_report(predictions, metadata)

if __name__ == "__main__":
    main()
