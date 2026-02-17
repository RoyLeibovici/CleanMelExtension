import os
import glob
import shutil
import random
import sys

# --- 1. CONFIGURATION ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))

# Paths
SPEECH_DIR = os.path.join(project_root, "datasets", "he", "clean_hebrew_16k")
NOISE_DIR = os.path.join(project_root, "datasets", "noise")

# Split Ratios (Must sum to 1.0)
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
# TEST_RATIO is the remainder (0.10)

print(f"🚀 Starting 3-Way Dataset Organization (Train/Val/Test)")
print(f"📂 Speech Dir: {SPEECH_DIR}")
print(f"📂 Noise Dir:  {NOISE_DIR}")
print("-" * 40)


# --- 2. LOGIC ---
def reorganize_dataset(root_dir, name="Dataset"):
    print(f"\n🔍 Processing {name}...")

    if not os.path.exists(root_dir):
        print(f"❌ Error: Directory not found: {root_dir}")
        return

    train_dir = os.path.join(root_dir, "train")
    val_dir = os.path.join(root_dir, "val")
    test_dir = os.path.join(root_dir, "test")

    # A. RESET STEP: Check if we need to flatten previous splits
    # If train exists but test is missing, we likely have the old 2-way split.
    # We need to move everything back to root to re-shuffle properly.
    if os.path.exists(train_dir) and not os.path.exists(test_dir):
        print("   ⚠️ Found old 2-way split (train/val). Resetting structure...")
        for folder in [train_dir, val_dir]:
            if os.path.exists(folder):
                for f in os.listdir(folder):
                    shutil.move(os.path.join(folder, f), os.path.join(root_dir, f))
                os.rmdir(folder)  # Remove empty folder
        print("   ✅ Reset complete. Re-splitting...")

    # Check if already done (all 3 exist and have files)
    has_train = os.path.exists(train_dir) and len(os.listdir(train_dir)) > 0
    has_val = os.path.exists(val_dir) and len(os.listdir(val_dir)) > 0
    has_test = os.path.exists(test_dir) and len(os.listdir(test_dir)) > 0

    if has_train and has_val and has_test:
        print(f"✅ {name} is already split into train/val/test. Skipping.")
        return

    # B. COLLECT FILES
    extensions = ['*.wav', '*.flac', '*.WAV', '*.FLAC', '*.mp3']
    files = []
    for ext in extensions:
        # Recursive search in root (but ignore train/val/test if they exist to avoid double counting)
        found = glob.glob(os.path.join(root_dir, "**", ext), recursive=True)
        files.extend(found)

    # Filter out files already in destination folders (just in case)
    files = [f for f in files if
             "train" not in os.path.split(os.path.dirname(f))[-1] and
             "val" not in os.path.split(os.path.dirname(f))[-1] and
             "test" not in os.path.split(os.path.dirname(f))[-1]]

    total_files = len(files)
    if total_files == 0:
        print(f"⚠️  No audio files found in {root_dir}")
        return

    print(f"📋 Found {total_files} files to organize.")

    # C. CREATE DIRS
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # D. SPLIT
    random.seed(42)
    random.shuffle(files)

    n_train = int(total_files * TRAIN_RATIO)
    n_val = int(total_files * VAL_RATIO)
    # n_test is the rest

    train_files = files[:n_train]
    val_files = files[n_train: n_train + n_val]
    test_files = files[n_train + n_val:]

    print(f"   • Train: {len(train_files)}")
    print(f"   • Val:   {len(val_files)}")
    print(f"   • Test:  {len(test_files)}")
    print(f"   • Moving files...")

    # E. MOVE
    def move_safe(src, dest_folder):
        filename = os.path.basename(src)
        dest_path = os.path.join(dest_folder, filename)

        # Handle duplicate names
        counter = 1
        while os.path.exists(dest_path):
            name, ext = os.path.splitext(filename)
            dest_path = os.path.join(dest_folder, f"{name}_{counter}{ext}")
            counter += 1

        shutil.move(src, dest_path)

    for f in train_files: move_safe(f, train_dir)
    for f in val_files:   move_safe(f, val_dir)
    for f in test_files:  move_safe(f, test_dir)

    # Cleanup empty folders
    for root, dirs, _ in os.walk(root_dir, topdown=False):
        for name in dirs:
            d = os.path.join(root, name)
            if d not in [train_dir, val_dir, test_dir] and not os.listdir(d):
                try:
                    os.rmdir(d)
                except:
                    pass

    print(f"✅ {name} organized successfully!")


# --- 3. EXECUTION ---
if __name__ == "__main__":
    reorganize_dataset(SPEECH_DIR, "Hebrew Speech")
    reorganize_dataset(NOISE_DIR, "Noise Dataset")
    print("\n✨ All datasets prepared.")