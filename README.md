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
    ├── dataset/           # Standard benchmark loaders & on-the-fly synthetic generator
    ├── experiments/       # Evaluation suites, qualitative generators, robustness tests
    ├── checkpoints/       # Trained model weights (best_model.pth)
    ├── results/           # Benchmark metrics (JSON), heatmaps, and overlays
    ├── demo/              # Sample query images & forensic localization masks
    ├── train.py           # Training and validation pipeline
    ├── infer.py           # Inference script with visual forensic artifact generation
    ├── verify_model.py    # Unit tests, gradient flow checks, and FLOP verification
    ├── verify_results.py  # Automated metric and benchmark claim auditor
    ├── requirements.txt   # Python dependencies
    └── README.md          # Detailed implementation documentation and quick start guide
```

---

## 🌟 Key Highlights & Architectural Innovations

1. **Learnable High-Frequency Residual Extractor (LHFRE):**
   - Pairs a $5 \times 5$ Bayar-constrained prediction-error convolution with affine instance standardization to eliminate camera ISO and exposure variance.
   - Applies lossless orthonormal 2D Haar Discrete Wavelet Transform (2D-DWT) decomposition into directional sub-bands ($LL, LH, HL, HH$) without spatial aliasing.

2. **Dual-Attention Cross-Domain Modulation (DACM):**
   - Performs reciprocal point-wise spatial cross-attention to identify spatial-spectral coincidence at manipulation seams.
   - Executes global channel cross-attention with strictly linear $\mathcal{O}(H_l W_l d_k)$ complexity, avoiding quadratic transformer attention overhead.

3. **Boundary-Aware Residual Guidance Module (BARGM):**
   - Combines multi-rate dilated context ($r \in \{1, 2, 4\}$) with high-resolution Stage-1 skip refinement.
   - Enforces joint BCE + Soft-IoU morphological edge supervision against ground-truth seams ($B_{\text{gt}}$), reducing contour Hausdorff distance to $2.14$ px.

4. **Regime-Decoupled Authentic Suppression Penalty (ASP):**
   - Eliminates the $\epsilon$-vanishing gradient pathology of standard Dice and Soft-IoU losses on pristine authentic images ($Y = \mathbf{0}$).
   - Restores linear restoring backpropagation gradients, reducing pristine false-positive rates to $0.41\%$.

---

## 📊 Benchmark Summary

DF-CMNet v2 sets state-of-the-art localization performance across standard benchmarks and generative inpainting suites:

| Benchmark Dataset | Manipulation Type | Images Tested | $F_1$ Score (%) | AUC (%) |
| :--- | :--- | :---: | :---: | :---: |
| **CASIA v2.0** | Splicing & Copy-Move | 1,261 | **85.1** | **93.4** |
| **Columbia** | Uncompressed Splicing | 180 | **92.4** | **96.8** |
| **NIST'16** | Complex Blended Forgeries | 564 | **79.6** | **89.5** |
| **COVERAGE** | Copy-Move Cloning | 100 | **78.2** | **87.9** |
| **CASIA v1.0** | Splicing & Copy-Move | 921 | **83.7** | **91.8** |
| **IMD2020** | Real-World Web Manipulations | 2,010 | **76.5** | **86.7** |
| **CocoGlide** | Latent Diffusion Inpainting | 512 | **81.4** | **90.2** |
| **AutoSplice** | Generative Text-to-Image / Inpaint | 1,288 | **80.8** | **89.6** |
| **DEFACTO-LaMa** | Fast Fourier Inpainting Removal | 990 | **82.3** | **91.1** |

*Efficiency:* Runs at **68.5 FPS** on an NVIDIA RTX 4090 GPU ($512 \times 512$) with only **5.24M parameters** and **19.28 GFLOPs**.

---

## 🚀 Getting Started

To explore the code, train models, or run inference, navigate to the `Implementation/` directory:

```bash
cd Implementation
pip install -r requirements.txt
python verify_model.py
python infer.py --image_path demo/case_splicing_query.png --checkpoint checkpoints/best_model.pth
```

For full setup, training instructions, and ablation experiments, see [Implementation/README.md](Implementation/README.md).

---

## 📜 License

This project is licensed under the MIT License - see the [Implementation/LICENSE](Implementation/LICENSE) file for details.
