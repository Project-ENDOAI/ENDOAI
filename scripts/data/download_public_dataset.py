"""
Download and extract the Medical Segmentation Decathlon Task04_Hippocampus dataset
into the data/raw/ directory for use with ENDOAI.
"""

import os
import urllib.request
import tarfile

def download_and_extract(url, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    filename = url.split("/")[-1]
    filepath = os.path.join(dest_dir, filename)
    if not os.path.exists(filepath):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filepath)
    else:
        print(f"{filename} already downloaded.")
    if filepath.endswith(".tar"):
        print("Extracting...")
        with tarfile.open(filepath) as tar:
            tar.extractall(dest_dir)
        print("Extraction complete.")

if __name__ == "__main__":
    # Download to data/raw/
    url = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task04_Hippocampus.tar"
    dest_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
    download_and_extract(url, dest_dir)
