import os
import sys
from huggingface_hub import snapshot_download

# --- Configuration ---
# We determine the Project Root relative to this script
# Assuming this script is placed in 'src' or 'scripts', we go up to the root.
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))

# Destination directories
# The snapshot_download will create the structure 'datasets/noise' inside base_dir
dataset_root = os.path.join(project_root, "datasets")

print(f"Project Root: {project_root}")
print(f"Target Directory: {dataset_root}")
print("-" * 40)

# --- Execution ---
try:
    print("Initializing download from Hugging Face (ltnghia/DNS-Challenge)...")

    # We download specifically the 'datasets/noise' folder from the repo
    # This preserves the repo structure, so it will download to:
    # {project_root}/datasets/noise
    snapshot_download(
        repo_id="ltnghia/DNS-Challenge",
        repo_type="dataset",
        allow_patterns="datasets/noise/*",
        local_dir=project_root,  # It appends the repo structure automatically
        local_dir_use_symlinks=False,
        resume_download=True
    )

    # Verify path
    downloaded_path = os.path.join(project_root, "datasets", "noise")
    if os.path.exists(downloaded_path):
        print(f"Success. Noise dataset located at: {downloaded_path}")
    else:
        print("Error: Download completed but the expected folder was not found.")

except Exception as e:
    print(f"An error occurred: {e}")
    print("Make sure you have installed the library: pip install huggingface_hub")