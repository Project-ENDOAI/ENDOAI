"""
Store Postoperative Notes

This script provides utilities for storing and managing postoperative notes.
"""

import os
import json

def save_postoperative_notes(patient_id, notes, output_dir="../data/postoperative/notes"):
    """
    Save postoperative notes for a patient.

    Args:
        patient_id (str): Unique identifier for the patient.
        notes (str): Postoperative notes to save.
        output_dir (str): Directory to save the notes.

    Returns:
        str: Path to the saved notes file.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{patient_id}_notes.json")
    with open(file_path, "w") as f:
        json.dump({"patient_id": patient_id, "notes": notes}, f, indent=4)
    print(f"Postoperative notes saved to {file_path}")
    return file_path

def main():
    """
    Main function to demonstrate saving postoperative notes.
    """
    patient_id = "patient_001"
    notes = "Patient is recovering well. No signs of infection. Scheduled for follow-up in 2 weeks."
    save_postoperative_notes(patient_id, notes)

if __name__ == "__main__":
    main()
