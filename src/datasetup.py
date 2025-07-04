import zipfile
import os
from pathlib import Path

def unzip_to_data(zip_path, extract_dir="data"):
    """
    Unzip the zip file to the data directory.
    """
    print(f"Unzipping '{zip_path}' to '{extract_dir}' ...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Extraction complete.")

if __name__ == "__main__":
    zip_file = "C:/Users/Lenovo/Downloads/creditcard.csv.zip"
    extract_dir = "data"
    
    # Ensure 'data/' exists
    os.makedirs(extract_dir, exist_ok=True)
    
    # Check if zip file exists
    if not Path(zip_file).exists():
        print(f"Error: '{zip_file}' not found in the current directory.")
    else:
        unzip_to_data(zip_file, extract_dir)
