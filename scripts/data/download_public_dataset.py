"""
Download a public medical imaging dataset (e.g., Medical Segmentation Decathlon Task04_Hippocampus).
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
    # Example: Medical Segmentation Decathlon Task04_Hippocampus
    url = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task04_Hippocampus.tar"
    dest_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "public")
    download_and_extract(url, dest_dir)
