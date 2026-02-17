import librosa
import soundfile as sf
import os
import sys
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# --- 1. CONFIGURATION ---
# Determine project root based on script location
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))

# Source: Where the raw MP3s are (datasets/he/clips)
SOURCE_FOLDER = os.path.join(project_root, "datasets", "he", "clips")

# Target: Where the cleaned WAVs will go (datasets/he/clean_hebrew_16k)
TARGET_FOLDER = os.path.join(project_root, "datasets", "he", "clean_hebrew_16k")

# Audio Settings
TARGET_SR = 16000
NUM_WORKERS = os.cpu_count() or 4  # Use all available CPU cores, fallback to 4

print(f"Starting Preprocessing")
print(f"Source Directory: {SOURCE_FOLDER}")
print(f"Target Directory: {TARGET_FOLDER}")
print(f"Workers: {NUM_WORKERS}")
print("-" * 40)

# --- 2. VALIDATION ---
if not os.path.exists(SOURCE_FOLDER):
    print(f"Error: Source folder not found at {SOURCE_FOLDER}")
    print("Please make sure you have extracted the Hebrew dataset correctly.")
    sys.exit(1)

os.makedirs(TARGET_FOLDER, exist_ok=True)


# --- 3. PROCESSING FUNCTION ---
def process_file(file_path):
    """
    Reads an MP3, resamples to 16kHz mono, and saves as WAV.
    """
    try:
        src_full = file_path
        filename = os.path.basename(src_full)
        tgt_filename = os.path.splitext(filename)[0] + ".wav"
        tgt_full = os.path.join(TARGET_FOLDER, tgt_filename)

        # Skip if already exists
        if os.path.exists(tgt_full):
            return None

            # Load audio (librosa automatically resamples and mixes to mono)
        # sr=16000 ensures it is 16kHz
        # mono=True ensures it is single channel
        audio, _ = librosa.load(src_full, sr=TARGET_SR, mono=True)

        # Save as WAV
        sf.write(tgt_full, audio, TARGET_SR)
        return None

    except Exception as e:
        return f"Error processing {os.path.basename(file_path)}: {str(e)}"


# --- 4. EXECUTION ---
if __name__ == "__main__":
    # A. Collect all MP3 files
    print("Scanning files...")
    mp3_files = []

    for root, _, files in os.walk(SOURCE_FOLDER):
        for file in files:
            if file.lower().endswith('.mp3'):
                mp3_files.append(os.path.join(root, file))

    total_files = len(mp3_files)
    print(f"Found {total_files} MP3 files to process.")

    if total_files == 0:
        print("No MP3 files found. Check your source directory.")
        sys.exit(0)

    # B. Run Parallel Processing
    print(f"Converting to 16kHz WAV...")

    errors = []
    # We use ProcessPoolExecutor for parallel CPU processing
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # map returns an iterator, list() consumes it to trigger processing
        # tqdm adds a progress bar
        results = list(tqdm(executor.map(process_file, mp3_files), total=total_files, unit="file"))

        # Collect errors
        for res in results:
            if res:
                errors.append(res)

    # --- 5. REPORT ---
    print("-" * 40)

    # Count output files
    wav_count = len([f for f in os.listdir(TARGET_FOLDER) if f.endswith('.wav')])

    print(f"Processing Complete.")
    print(f"Total Source Files: {total_files}")
    print(f"Total WAVs Created: {wav_count}")

    if errors:
        print(f"{len(errors)} files failed to convert.")
        print("First 5 errors:")
        for err in errors[:5]:
            print(f"- {err}")
    else:
        print("All files processed successfully.")