import subprocess
import sys
import os

# --- CONFIGURATION ---
GPUS = "1"
MODE = "offline"
EXP_NAME = "finetune_english_mask"

# Checkpoints
PRETRAINED_CKPT = "./pretrained/enhancement/offline_CleanMel_S_mask.ckpt"
VOCOS_CKPT = f"./pretrained/vocos/vocos_{MODE}.pt"

# Config Files
# We use the NEW finetune config we just created
MODEL_CONFIG = "./configs/model/cleanmel_offline_hebrew_fine_tune.yaml"
DATA_CONFIG = "./configs/dataset/train_english.yaml"
VOCOS_CONFIG = f"./configs/model/vocos_{MODE}.yaml"

# --- EXECUTION ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))

print(f"🚀 Starting Fine-Tuning")
print(f"📦 Exp Name: {EXP_NAME}")
print(f"🔧 Model Config: {MODEL_CONFIG}")
print("-" * 40)

command = [
    sys.executable, "-m", "model.CleanMelTrainer_mask", "fit",

    "--config", MODEL_CONFIG,
    "--config", DATA_CONFIG,

    # Vocoder
    "--model.vocos_ckpt", VOCOS_CKPT,
    "--model.vocos_config", VOCOS_CONFIG,

    # Weights
    # This loads the weights BUT resets the optimizer (Perfect for fine-tuning)
    "--model.arch_ckpt", PRETRAINED_CKPT,

    # Logging
    "--model.exp_name", EXP_NAME
]

env = os.environ.copy()
env["PYTHONPATH"] = f"{project_root}:{os.path.join(project_root, 'src')}:{env.get('PYTHONPATH', '')}"

try:
    subprocess.run(command, check=True, cwd=project_root, env=env)
    print("\n✅ Training finished successfully!")
except subprocess.CalledProcessError as e:
    print(f"\n❌ Training failed (Exit Code: {e.returncode})")
except KeyboardInterrupt:
    print("\n🛑 Training stopped by user.")