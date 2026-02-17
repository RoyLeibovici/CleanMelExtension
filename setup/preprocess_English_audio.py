import os
import glob
import shutil
import random

# --- 1. CONFIGURATION ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))

# Path to the dataset you just redownloaded
# This should be the folder containing the Speaker ID folders (e.g., "1089", "121", etc.)
ENGLISH_DIR = os.path.join(project_root, "datasets", "test-clean", "LibriSpeech", "test-clean")

# Split Ratios
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

print(f"🚀 Starting Dataset Organization")
print(f"📂 Target Dir: {ENGLISH_DIR}")
print("-" * 40)


def reorganize_dataset(root_dir):
    if not os.path.exists(root_dir):
        print(f"❌ Error: Directory not found: {root_dir}")
        return

    # Define Destination Folders
    train_dir = os.path.join(root_dir, "train")
    val_dir = os.path.join(root_dir, "val")
    test_dir = os.path.join(root_dir, "test")

    # Create them if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 1. COLLECT AUDIO FILES
    print("🔍 Scanning for audio files...")
    # LibriSpeech is usually .flac, but we look for everything just in case
    extensions = ['*.flac', '*.wav', '*.mp3']
    files = []

    for ext in extensions:
        # Recursive search
        found = glob.glob(os.path.join(root_dir, "**", ext), recursive=True)
        files.extend(found)

    # Filter out files that are ALREADY in the destination folders (if you ran this partially)
    files = [f for f in files if
             "train" not in os.path.split(os.path.dirname(f))[-1] and
             "val" not in os.path.split(os.path.dirname(f))[-1] and
             "test" not in os.path.split(os.path.dirname(f))[-1]]

    total_files = len(files)
    if total_files == 0:
        print(f"⚠️  No audio files found to move.")
        return

    print(f"📋 Found {total_files} audio files.")

    # 2. SPLIT
    random.seed(42)
    random.shuffle(files)

    n_train = int(total_files * TRAIN_RATIO)
    n_val = int(total_files * VAL_RATIO)

    train_files = files[:n_train]
    val_files = files[n_train: n_train + n_val]
    test_files = files[n_train + n_val:]

    print(f"   • Train: {len(train_files)}")
    print(f"   • Val:   {len(val_files)}")
    print(f"   • Test:  {len(test_files)}")

    # 3. MOVE FILES
    print("🚚 Moving files...")

    def move_safe(src, dest_folder):
        filename = os.path.basename(src)
        dest_path = os.path.join(dest_folder, filename)

        # Handle duplicate filenames
        counter = 1
        while os.path.exists(dest_path):
            name, ext = os.path.splitext(filename)
            dest_path = os.path.join(dest_folder, f"{name}_{counter}{ext}")
            counter += 1

        shutil.move(src, dest_path)

    for f in train_files: move_safe(f, train_dir)
    for f in val_files:   move_safe(f, val_dir)
    for f in test_files:  move_safe(f, test_dir)

    # 4. AGGRESSIVE CLEANUP
    # This deletes the old folders (which now contain only text files)
    print("🧹 Cleaning up old folders and text files...")

    # List everything in the root directory
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)

        # Skip our new Train/Val/Test folders
        if item in ["train", "val", "test"]:
            continue

        # If it's a directory (like "1089", "121"), delete it and everything inside
        if os.path.isdir(item_path):
            try:
                shutil.rmtree(item_path)
                print(f"   - Deleted old folder: {item}")
            except Exception as e:
                print(f"   ⚠️ Could not delete {item}: {e}")

    print(f"\n✅ Success! Dataset is organized into {root_dir}/[train, val, test]")


if __name__ == "__main__":
    reorganize_dataset(ENGLISH_DIR)