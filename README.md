# DF-CMNet v2: Dual-Domain Frequency-Spatial Cross-Modulation Network with Boundary-Aware Supervision for Robust Passive Image Forgery Localization

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](Implementation/LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-brightgreen.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/Release-v2.0-orange.svg)](https://github.com/SuttA-MNNIT/DF-CMNet/releases/tag/v2.0)

This repository contains the complete official PyTorch codebase, evaluation benchmarks, and pre-trained weights for **DF-CMNet v2**, an end-to-end dual-domain deep forensic framework designed for passive image forgery localization across splicing, copy-move cloning, and generative diffusion inpainting.

---

## 📁 Repository Organization

The repository is organized under `Implementation/`:

```
DF-CMNet/
└── Implementation/        # Full PyTorch source code, models, and benchmarks
    ├── models/            # Architecture definitions (LHFRE, DACM, BARGM, DF-CMNet v2)
    ├── losses/            # Compound objective with Authentic Suppression Penalty (ASP)
    ├── dataset/           # Real benchmark loaders & on-the-fly synthetic generator
    ├── experiments/       # Genuine evaluation suite (benchmark_suite, generate_qualitative_suite)
    ├── checkpoints/       # Trained model weights directory (.gitkeep)
    ├── results/           # Benchmark metrics (JSON), heatmaps, and overlays (.gitkeep)
    ├── Colab_H100_Runner.ipynb # One-click Master Google Colab H100 execution notebook
    ├── download_datasets.py # Automated benchmark dataset downloader (Kaggle & mirrors)
    ├── train.py           # Training and validation pipeline with ablation support (V1..V8)
    ├── infer.py           # Single-image inference CLI with visual forensic artifact generation
    ├── verify_model.py    # Unit tests, gradient flow checks, and FLOP verification
    ├── requirements.txt   # Python dependencies
    └── README.md          # Detailed implementation documentation and quick start guide
```

---

## 🌟 Key Highlights & Architectural Innovations

1. **Learnable High-Frequency Residual Extractor (LHFRE):**
   - Pairs a $5 \times 5$ Bayar-constrained prediction-error convolution with affine instance standardization to eliminate camera ISO and exposure variance.
   - Applies orthonormal 2D Haar Discrete Wavelet Transform (2D-DWT) decomposition into directional sub-bands ($LL, LH, HL, HH$) without spatial aliasing.

2. **Dual-Attention Cross-Domain Modulation (DACM):**
   - Performs reciprocal point-wise spatial cross-attention to identify spatial-spectral coincidence at manipulation seams.
   - Executes global channel cross-attention with strictly linear $\mathcal{O}(H_l W_l d_k)$ complexity, avoiding quadratic transformer attention overhead.

3. **Boundary-Aware Residual Guidance Module (BARGM):**
   - Combines multi-rate dilated context ($r \in \{1, 2, 4\}$) with high-resolution Stage-1 skip refinement.
   - Enforces joint BCE + Soft-IoU morphological edge supervision against ground-truth seams ($B_{\text{gt}}$).

4. **Regime-Decoupled Authentic Suppression Penalty (ASP):**
   - Eliminates the vanishing gradient pathology of standard overlap losses on pristine authentic images ($Y = \mathbf{0}$).
   - Restores linear restoring backpropagation gradients on unmanipulated images.

---

## 🚀 Getting Started & Google Colab H100 Execution

### Option A: One-Click Execution on Google Colab (H100 / A100 / T4)
Open `Implementation/Colab_H100_Runner.ipynb` in Google Colab to run the entire pipeline with GPU acceleration:
1. Automated dataset acquisition (`CASIA v2.0`, `Columbia`, `COVERAGE`, `CocoGlide`).
2. Dual-stream architecture verification.
3. 60-epoch mixed-precision training (`--amp`).
4. Full benchmark evaluation across datasets and post-processing stress conditions.
5. Export of checkpoints, JSON logs, and qualitative overlay figures.

### Option B: Local Environment
To explore the code, train models, or run inference locally:

```bash
cd Implementation
pip install -r requirements.txt
python verify_model.py
python train.py --data_dir ./data/CASIA_v2 --epochs 60 --batch_size 32 --amp
```

For full setup, training instructions, and ablation experiments, see [Implementation/README.md](Implementation/README.md).

---

## 📜 License

This project is licensed under the MIT License - see the [Implementation/LICENSE](Implementation/LICENSE) file for details.
