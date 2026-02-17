import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import librosa
import sys

# Ensure we don't need a GUI display
plt.switch_backend('Agg')

# ============================
# CONFIGURATION
# ============================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# Path to your test set outputs
INFERENCE_DIR = os.path.join(PROJECT_ROOT, "logs/dummy.ckpt_test_set/pretrained_hebrew_sisnr/examples")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "presentation_images_pretrained_hebrew")

# Audio Config
SR = 16000
HOP_LENGTH = 128
N_MELS = 80
EPS = 1e-5
DYNAMIC_RANGE = 11.5  # Approx ln(1e-5)


# ============================
# HELPERS
# ============================

def compute_log_mel_from_audio(audio_path: str):
    """Load audio (flac/wav) and compute log-Mel spectrogram."""
    try:
        y, _ = librosa.load(audio_path, sr=SR)

        mel = librosa.feature.melspectrogram(
            y=y,
            sr=SR,
            n_fft=512,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            power=2.0,
        )
        mel = np.maximum(mel, EPS)
        log_mel = np.log(mel)
        log_mel -= log_mel.max() # Peak Norm
        return log_mel
    except Exception as e:
        print(f"⚠️ Failed loading audio {os.path.basename(audio_path)}: {e}")
        return None


def normalize_inference_npy(mel: np.ndarray) -> np.ndarray:
    """Normalize the .npy inference output to match paper scaling."""
    if mel.ndim == 3: mel = mel.squeeze()
    if mel.shape[0] > mel.shape[1]: mel = mel.T
    if np.min(mel) >= 0: mel = np.log(np.maximum(mel, EPS))
    if mel.max() != 0: mel -= mel.max()
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
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_ylabel("Mel bins")
    ax.set_xlabel("Time (s)")
    return im


# ============================
# MAIN
# ============================

def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"🔹 Scanning: {INFERENCE_DIR}")

    if not os.path.exists(INFERENCE_DIR):
        print(f"❌ Directory not found: {INFERENCE_DIR}")
        return

    # 1. Gather all files
    all_files = glob.glob(os.path.join(INFERENCE_DIR, "**", "*"), recursive=True)

    # 2. Group by Sample ID
    # We assume filename format: {ID}_mis, {ID}_spk1, {ID}_spk1_p
    samples = {}

    for f in all_files:
        if os.path.isdir(f): continue

        filename = os.path.basename(f)
        name_no_ext = os.path.splitext(filename)[0]

        # Determine Role and ID
        sample_id = None
        role = None

        if name_no_ext.endswith("_mis") or name_no_ext.endswith("_mix"):
            role = "noisy"
            # Remove suffix to get ID
            sample_id = name_no_ext.replace("_mis", "").replace("_mix", "")

        elif name_no_ext.endswith("_spk1_p"):
            role = "denoised"
            sample_id = name_no_ext.replace("_spk1_p", "")

        elif name_no_ext.endswith("_spk1"):
            role = "clean"
            sample_id = name_no_ext.replace("_spk1", "")

        if sample_id:
            if sample_id not in samples:
                samples[sample_id] = {}
            samples[sample_id][role] = f

    print(f"📋 Found {len(samples)} unique sample groups.")

    # 3. Process and Plot
    count = 0
    for sample_id, files in samples.items():
        # We need at least Clean and Denoised to compare
        if "clean" not in files or "denoised" not in files:
            continue

        try:
            # A. Load Data
            # Clean
            clean_mel = compute_log_mel_from_audio(files["clean"])

            # Denoised (Check extension)
            denoised_path = files["denoised"]
            if denoised_path.endswith(".npy"):
                pred_raw = np.load(denoised_path)
                denoised_mel = normalize_inference_npy(pred_raw)
            else:
                denoised_mel = compute_log_mel_from_audio(denoised_path)

            # Noisy (Optional)
            noisy_mel = None
            if "noisy" in files:
                noisy_mel = compute_log_mel_from_audio(files["noisy"])

            # B. Plot
            if clean_mel is not None and denoised_mel is not None:
                # Setup: 3 rows if noisy exists, else 2
                rows = 3 if noisy_mel is not None else 2
                fig, axes = plt.subplots(rows, 1, figsize=(10, 4 * rows), sharex=True)

                # Ensure axes is iterable
                if rows == 1: axes = [axes]

                idx = 0
                # 1. Noisy
                if noisy_mel is not None:
                    plot_spectrogram(noisy_mel, f"Noisy Input: {sample_id}_mis", axes[idx])
                    idx += 1

                # 2. Clean
                plot_spectrogram(clean_mel, f"Target (Clean): {sample_id}_spk1", axes[idx])
                idx += 1

                # 3. Denoised
                im = plot_spectrogram(denoised_mel, f"Enhanced (Model): {sample_id}_spk1_p", axes[idx])

                # Colorbar (shared)
                cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
                cbar = fig.colorbar(im, cax=cbar_ax)
                cbar.set_ticks([-11.5, -8, -4, 0])
                cbar.set_ticklabels(['ln(1e-5)', '-8', '-4', '0'])

                # Save
                save_name = f"viz_{sample_id}.png"
                plt.savefig(os.path.join(OUTPUT_DIR, save_name), dpi=150, bbox_inches='tight')
                plt.close()

                print(f"   ✅ Generated: {save_name}")
                count += 1

        except Exception as e:
            print(f"   ⚠️ Error processing {sample_id}: {e}")

    print(f"\n✨ Done! Processed {count} comparisons.")
    print(f"   Check folder: {OUTPUT_DIR}")

if __name__ == "__main__":
    run()