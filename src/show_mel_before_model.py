import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import librosa

# Ensure we don't need a GUI display
plt.switch_backend('Agg')

# ============================
# CONFIGURATION
# ============================

# ⚠️ PASTE YOUR AUDIO FILE PATH HERE (.wav or .flac)
AUDIO_FILE_PATH = "/home/enav/PycharmProjects/CleanMelExtension/dev_utils/test_sample/HebrewAudioNoisy.wav"

# Output Directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "single_spectrogram_plots")

# Audio Config (MUST MATCH YOUR MODEL TRAINING)
SR = 16000
HOP_LENGTH = 128
N_MELS = 80
EPS = 1e-5
DYNAMIC_RANGE = 11.5  # Approx ln(1e-5)


# ============================
# HELPERS
# ============================

def compute_log_mel_from_audio(audio_path: str):
    """Load audio and compute log-Mel spectrogram matching the model's format."""
    try:
        # 1. Load Audio (Force 16kHz)
        y, _ = librosa.load(audio_path, sr=SR)

        # 2. Compute Mel Spectrogram
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=SR,
            n_fft=512,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            power=2.0,
        )

        # 3. Apply Paper-Style Normalization
        # Clip to epsilon (floor)
        mel = np.maximum(mel, EPS)

        # Convert to Natural Log (ln)
        log_mel = np.log(mel)

        # Peak Normalize to 0
        # (This ensures the loudest part is Red/0, same as your model output)
        if log_mel.max() != 0:
            log_mel -= log_mel.max()

        return log_mel

    except Exception as e:
        print(f"⚠️ Failed loading audio {os.path.basename(audio_path)}: {e}")
        return None


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

    if not os.path.exists(AUDIO_FILE_PATH):
        print(f"❌ Error: File not found: {AUDIO_FILE_PATH}")
        return

    filename = os.path.basename(AUDIO_FILE_PATH)
    stem = os.path.splitext(filename)[0]

    print(f"🔹 Processing Audio: {filename}")

    try:
        # 1. Compute Data from Audio
        mel_data = compute_log_mel_from_audio(AUDIO_FILE_PATH)

        if mel_data is None:
            return

        # 2. Plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        plot_spectrogram(mel_data, f"Original Audio - Before Model", ax)

        # Colorbar
        cbar = plt.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("ln magnitude")
        cbar.set_ticks([-11.5, -8, -4, 0])
        cbar.set_ticklabels(['ln(1e-5)', '-8', '-4', '0'])

        # Save
        save_name = f"viz_audio_{stem}.png"
        save_path = os.path.join(OUTPUT_DIR, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✅ Saved plot to: {save_path}")

    except Exception as e:
        print(f"⚠️ Error processing {filename}: {e}")


if __name__ == "__main__":
    run()