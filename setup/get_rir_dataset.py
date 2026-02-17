import os
import requests
import zipfile
import sys

# --- Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))

# Where to save the zip and extract contents
download_dir = os.path.join(project_root, "datasets")
zip_filename = "rirs_noises.zip"
zip_path = os.path.join(download_dir, zip_filename)
url = "https://www.openslr.org/resources/28/rirs_noises.zip"

# Ensure download directory exists
os.makedirs(download_dir, exist_ok=True)

print(f"Project Root: {project_root}")
print(f"Download Target: {zip_path}")
print("-" * 40)


# --- Helper Functions ---
def download_file(url, destination):
    print(f"Downloading from {url}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            downloaded = 0
            chunk_size = 8192

            with open(destination, 'wb') as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    downloaded += len(chunk)
                    # Simple progress indicator
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        sys.stdout.write(f"\rProgress: {percent:.1f}%")
                        sys.stdout.flush()
        print("\nDownload complete.")
        return True
    except Exception as e:
        print(f"\nError downloading file: {e}")
        return False


def unzip_file(zip_filepath, extract_to):
    print(f"Extracting {zip_filepath}...")
    try:
        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("Extraction complete.")
        return True
    except Exception as e:
        print(f"Error unzipping file: {e}")
        return False


# --- Execution ---
# 1. Check if we need to download
if not os.path.exists(zip_path):
    success = download_file(url, zip_path)
    if not success:
        sys.exit(1)
else:
    print("Zip file already exists. Skipping download.")

# 2. Extract
# OpenSLR usually extracts to a folder named "RIRS_NOISES"
target_extraction_folder = os.path.join(download_dir, "RIRS_NOISES")

if not os.path.exists(target_extraction_folder):
    unzip_success = unzip_file(zip_path, download_dir)
    if not unzip_success:
        sys.exit(1)
else:
    print("Extracted folder already exists. Skipping extraction.")

# 3. Verify specific subfolder
final_path = os.path.join(download_dir, "RIRS_NOISES", "real_rirs_isotropic_noises")
if os.path.exists(final_path):
    print(f"Success. RIRs are ready at: {final_path}")
else:
    print("Warning: The expected subfolder 'real_rirs_isotropic_noises' was not found.")
    print(f"Check contents of: {os.path.join(download_dir, 'RIRS_NOISES')}")