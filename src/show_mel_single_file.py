import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Ensure we don't need a GUI display
plt.switch_backend('Agg')

# ============================
# CONFIGURATION
# ============================

# ⚠️ PASTE YOUR FILE PATH HERE
NPY_FILE_PATH = "/home/enav/PycharmProjects/CleanMelExtension/my_output/finetune_hebrew_mask/version_0/logmel/HebrewAudioNoisy.npy"

# Output Directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "single_spectrogram_plots")

# Audio Config
SR = 16000
HOP_LENGTH = 128
N_MELS = 80
EPS = 1e-5
DYNAMIC_RANGE = 11.5  # Approx ln(1e-5)


# ============================
# HELPERS
# ============================

def normalize_inference_npy(mel: np.ndarray) -> np.ndarray:
    """Normalize the .npy inference output to match paper scaling."""
    # 1. Squeeze batch dim if present: (1, 80, T) -> (80, T)
    if mel.ndim == 3:
        mel = mel.squeeze()

    # 2. Transpose if needed: (T, M) -> (M, T)
    # We assume time is the longer dimension usually
    if mel.shape[0] > mel.shape[1]:
        mel = mel.T

    # 3. Log Check
    # If the data contains negative numbers, it is ALREADY log-scale.
    # If it is strictly positive, it is Linear scale.
    if np.min(mel) >= 0:
        # Convert to Log
        mel = np.log(np.maximum(mel, EPS))

    # 4. Peak Normalize to 0
    # This guarantees the "Red" color for the loudest sounds
    if mel.max() != 0:
        mel -= mel.max()

    return mel


def plot_spectrogram(data: np.ndarray, title: str, ax):
    """Plot single spectrogram with paper-style settings."""
    duration = (data.shape[1] * HOP_LENGTH) / SR
    extent = [0, duration, 0, N_MELS]

    im = ax.imshow(
        data,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap='jet',
        vmin=-DYNAMIC_RANGE,
        vmax=0,
        interpolation="nearest",
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("Mel bins")
    ax.set_xlabel("Time (s)")
    return im


# ============================
# MAIN
# ============================

def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(NPY_FILE_PATH):
        print(f"❌ Error: File not found: {NPY_FILE_PATH}")
        return

    filename = os.path.basename(NPY_FILE_PATH)
    stem = os.path.splitext(filename)[0]

    # Try to extract parent folder name for context (e.g. "1089_134686...")
    parent_folder = os.path.basename(os.path.dirname(NPY_FILE_PATH))
    full_name = f"{parent_folder}_{stem}"

    print(f"🔹 Processing: {filename}")

    try:
        # 1. Load Data
        raw_data = np.load(NPY_FILE_PATH)

        # 2. Normalize
        mel_data = normalize_inference_npy(raw_data)

        # 3. Plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        plot_spectrogram(mel_data, f"Mel Spectrogram - fine tuned mask", ax)

        # Colorbar
        cbar = plt.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("ln magnitude")
        cbar.set_ticks([-11.5, -8, -4, 0])
        cbar.set_ticklabels(['ln(1e-5)', '-8', '-4', '0'])

        # Save
        save_name = f"viz_{full_name}.png"
        save_path = os.path.join(OUTPUT_DIR, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✅ Saved plot to: {save_path}")

    except Exception as e:
        print(f"⚠️ Error processing {filename}: {e}")


if __name__ == "__main__":
    run()