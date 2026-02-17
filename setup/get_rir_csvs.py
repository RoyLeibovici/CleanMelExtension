import os
import glob
import pandas as pd
import random
import sys

# --- CONFIGURATION ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))

# Path to the specific RIR folder (where the CSVs must be saved)
# This matches the path in your error log: ./datasets/RIRS_NOISES/real_rirs_isotropic_noises/
RIR_DIR = os.path.join(project_root, "datasets", "RIRS_NOISES", "real_rirs_isotropic_noises")

print(f"🚀 Generating RIR CSVs")
print(f"📂 Target Dir: {RIR_DIR}")
print("-" * 40)

if not os.path.exists(RIR_DIR):
    print(f"❌ Error: RIR directory not found at {RIR_DIR}")
    sys.exit(1)

# 1. Collect all RIR WAV files recursively
wav_files = glob.glob(os.path.join(RIR_DIR, "**", "*.wav"), recursive=True)

if len(wav_files) == 0:
    print("❌ No RIR wav files found! Check your download.")
    sys.exit(1)

print(f"📋 Found {len(wav_files)} RIR files.")

# 2. Shuffle and Split
random.seed(42)
random.shuffle(wav_files)

total = len(wav_files)
n_train = int(total * 0.8)
n_val = int(total * 0.1)
# Remaining 10% for test

train_files = wav_files[:n_train]
val_files = wav_files[n_train:n_train + n_val]
test_files = wav_files[n_train + n_val:]

print(f"   • Train: {len(train_files)}")
print(f"   • Val:   {len(val_files)}")
print(f"   • Test:  {len(test_files)}")


# 3. Create DataFrames and Save CSVs
# The dataloader expects columns: 'filename' and 't60'
# We use a dummy T60 value (0.3s) because we don't have the real measurements,
# and for training/testing this is mostly for logging, not the actual audio processing.

def save_csv(file_list, filename):
    # The dataloader reads relative or absolute paths.
    # We will save absolute paths to be safe.
    df = pd.DataFrame({
        'filename': file_list,
        't60': [0.3] * len(file_list)
    })

    save_path = os.path.join(RIR_DIR, filename)
    df.to_csv(save_path, index=False)
    print(f"✅ Saved {filename} to {save_path}")


save_csv(train_files, "train.csv")
save_csv(val_files, "val.csv")
save_csv(test_files, "test.csv")

print("\n✨ RIR preparation complete.")