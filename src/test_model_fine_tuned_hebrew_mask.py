import subprocess
import sys
import os

# --- 1. CONFIGURATION ---
# Hardcoded to match your Colab command
GPUS = "1"
MODE = "offline"
SIZE = "S"
OUTPUT = "mask"

# Checkpoint Paths (Relative to project root)
VOCOS_CKPT = "./pretrained/vocos/vocos_offline.pt"
ARCH_CKPT = "./logs/finetune_hebrew_mask/version_0/checkpoints/last.ckpt"

# Dataset Config
# IMPORTANT: You must ensure this file exists and points to your 'clean_hebrew_16k' folder
DATASET_CONFIG = "./configs/dataset/test_hebrew.yaml"

# --- 2. SETUP PATHS ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))

print(f"Starting Hebrew Test Run")
print(f"Project Root: {project_root}")
print(f"Dataset Config: {DATASET_CONFIG}")
print("-" * 40)

# --- 3. CONSTRUCT COMMAND ---
command = [
    sys.executable, "-m", f"model.CleanMelTrainer_{OUTPUT}", "test",

    # Configs
    "--config", "./configs/model/cleanmel_offline.yaml",
    "--config", DATASET_CONFIG,

    # Architecture Overrides (Size S)
    "--model.arch.init_args.n_layers=8",
    "--model.arch.init_args.dim_hidden=96",

    # Vocos (Vocoder) Settings
    "--model.vocos_ckpt", VOCOS_CKPT,
    "--model.vocos_config", "./configs/model/vocos_offline.yaml",

    # Trainer Settings
    f"--trainer.devices={GPUS}",
    #"--trainer.accelerator=gpu",  # Explicitly set gpu
    #"--trainer.strategy=auto",  # Force auto to avoid DDP crash

    # Experiment Name
    "--model.exp_name", "./test_finetune_hebrew_mask",

    # The Pretrained Checkpoint to Test
    "--model.arch_ckpt", ARCH_CKPT
]

# --- 4. ENVIRONMENT SETUP ---
env = os.environ.copy()
env["PYTHONPATH"] = f"{project_root}:{os.path.join(project_root, 'src')}:{env.get('PYTHONPATH', '')}"

# --- 5. EXECUTE ---
try:
    subprocess.run(command, check=True, cwd=project_root, env=env)
    print("\nTest finished successfully!")

except subprocess.CalledProcessError as e:
    print(f"\nError during testing (Exit Code: {e.returncode})")
except KeyboardInterrupt:
    print("\nStopped by user.")