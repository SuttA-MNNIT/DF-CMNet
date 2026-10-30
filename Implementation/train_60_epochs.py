"""
60-Epoch Full Training Entry Point for DF-CMNet v2.
Delegates to the unified H100/CUDA mixed-precision training pipeline in `train.py`.

Usage on Google Colab H100:
    python train_60_epochs.py --data_dir ./data/CASIA_v2 --epochs 60 --batch_size 16 --img_size 256
"""

import sys
from train import main

if __name__ == "__main__":
    if "--epochs" not in sys.argv:
        sys.argv.extend(["--epochs", "60"])
    if "--img_size" not in sys.argv:
        sys.argv.extend(["--img_size", "256"])
    if "--batch_size" not in sys.argv:
        sys.argv.extend(["--batch_size", "16"])
    main()
