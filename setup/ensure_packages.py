import importlib.metadata
import sys

# The target requirements (Adjusted for compatibility)
TARGETS = {
    "einops": "0.8.1", "encodec": "0.1.1", "huggingface_hub": "0.23.0",
    "jsonargparse": "4.28.0", "librosa": "0.10.2.post1",
    "lightning": "2.2.0",  # Downgraded from 2.5.0 to fix "Disabling PyTorch" error
    "lightning_utilities": "0.11.9", "matplotlib": "3.8.0",
    "numpy": "1.26.4",  # Downgraded from 2.2.3 for Torch compatibility
    "onnxruntime_gpu": "1.18.0", "pandas": "2.2.3", "pesq": "0.0.4",
    "pytorch_lightning": "2.0.3", "PyYAML": "6.0.2", "requests": "2.32.3",
    "scipy": "1.15.2", "soundfile": "0.12.1", "torch": "2.2.0+cu121",
    "torchaudio": "2.2.0+cu121", "torchinfo": "1.8.0", "torchmetrics": "1.1.2",
    "tqdm": "4.66.4", "transformers": "4.40.1", "mamba_ssm": "1.2.0.post1"
}

print(f"{'PACKAGE':<25} | {'STATUS':<10} | {'INSTALLED':<15} | {'TARGET':<15}")
print("-" * 75)
for pkg, target in TARGETS.items():
    try:
        current = importlib.metadata.version(pkg)
        status = "✅ OK" if current == target else "⚠️ DIFF"
        # Special check for Torch CUDA version
        if pkg == "torch" and "+cu121" in target and "+cu121" not in current:
            status = "❌ CUDA?"
    except importlib.metadata.PackageNotFoundError:
        current = "MISSING"
        status = "❌ MISSING"

    print(f"{pkg:<25} | {status:<10} | {current:<15} | {target:<15}")