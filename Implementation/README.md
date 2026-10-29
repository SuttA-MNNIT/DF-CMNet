# DF-CMNet: Dual-Domain Frequency-Spatial Cross-Modulation Network for Passive Image Forgery Localization

An open-source PyTorch implementation of **DF-CMNet**, an end-to-end deep learning framework designed for passive (blind) image forgery detection and pixel-level localization across various manipulation types (splicing, copy-move cloning, and generative inpainting).

---

## 🌟 Key Architecture & Highlights

DF-CMNet resolves structural bottlenecks in passive image forensics (such as semantic edge bias, cross-domain decoupling, and vanishing gradients on authentic images) through a coordinated dual-stream design:

1. **Learnable High-Frequency Residual Extractor (LHFRE):**
   - Integrates Bayar constrained convolution kernels to suppress semantic content and extract low-level sensor noise footprints (PRNU, demosaicing artifacts).
   - Decomposes residuals into directional frequency sub-bands ($LH, HL, HH$) via multi-scale 2D Discrete Wavelet Transform (2D-DWT).

2. **Cross-Domain Bidirectional Modulation (CDBM / DACM):**
   - Coordinates reciprocal spatial and channel cross-attention between spatial representations and spectral sub-bands.
   - Selectively recalibrates spatial representations using high-frequency anomaly cues and vice versa.

3. **Boundary-Aware Residual Guidance Module (BARGM):**
   - Employs Atrous Spatial Pyramid Pooling (ASPP) to capture multi-scale context along subtle donor-host transition seams.
   - Enforces sharp localization along blended or feathered manipulation margins.

4. **Compound Multi-Task Objective with Authentic Suppression Penalty (ASP):**
   - Jointly optimizes Focal Loss, Soft Dice Loss, and morphological boundary supervision.
   - Integrates an authentic suppression penalty that completely prevents zero-gradient pathologies on negative pristine imagery.

---

## 📁 Repository Structure

```
├── models/                      # Deep learning architecture
│   ├── __init__.py
│   ├── bayar_conv.py            # Bayar constrained convolution (noise residual extraction)
│   ├── wavelet_dwt.py           # 2D Haar Discrete Wavelet Transform (sub-band decomposition)
│   ├── cdbm.py                  # Cross-Domain Bidirectional Modulation (reciprocal attention)
│   ├── bargm.py                 # Boundary-Aware Residual Guidance Module (ASPP edge head)
│   └── df_cmnet.py              # Full DF-CMNet pipeline & progressive multi-scale decoder
│
├── losses/                      # Loss formulations
│   ├── __init__.py
│   └── compound_loss.py         # Multi-task loss: Focal Loss + Dice Loss + Morphological Boundary BCE
│
├── dataset/                     # Data loading & augmentation
│   ├── __init__.py
│   ├── forgery_dataset.py       # Benchmark loader (CASIA, Columbia, NIST'16, COVERAGE)
│   └── synthetic_tamper.py      # On-the-fly synthetic tamper generator for out-of-the-box training
│
├── experiments/                 # Comprehensive evaluation & visualization suite
│   ├── benchmark_suite.py       # Multi-benchmark stress testing & ablation suite
│   ├── generate_qualitative_suite.py # Qualitative comparative visual generator
│   └── generate_robustness_visuals.py # Robustness analysis under noise, blur, & compression
│
├── checkpoints/                 # Pretrained weights
│   ├── best_model.pth           # Trained model weights (F1: 0.851, AUC: 0.934)
│   └── best_model_60epochs.pth  # 60-epoch convergence checkpoint
│
├── demo/                        # Sample test imagery & ground-truth masks
├── results/                     # Benchmark artifacts, metric logs, & forensic heatmaps
├── verify_model.py              # Unit tests, gradient flow checks, and throughput benchmarking
├── train.py                     # Standard training & evaluation pipeline (F1, AUC, IoU)
├── train_60_epochs.py           # Extended training pipeline with fast metric evaluations
├── infer.py                     # Inference script producing heatmaps, masks, and boundary overlays
├── requirements.txt             # Environment dependencies
├── LICENSE                      # MIT License
└── README.md                    # Project documentation
```

---

## 🔧 Installation & Requirements

### 1. Clone the Repository
```bash
git clone https://github.com/SuttA-MNNIT/DF-CMNet.git
cd DF-CMNet
```

### 2. Install Dependencies
Ensure Python 3.9+ and PyTorch 2.0+ are installed:
```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start Guide

### 1. Verify Architecture & Gradient Flow
Run the automated verification suite to test forward pass, backward pass, gradient propagation across both spatial and frequency streams, parameter count, and inference throughput:

```bash
python verify_model.py
```

### 2. Run Inference & Tampering Localization
Localize tampering on any test image and generate visual forensic artifacts (probability heatmaps, binary masks, boundary maps, and overlaid masks):

```bash
python infer.py --image_path demo/case_splicing_query.png --checkpoint checkpoints/best_model.pth --img_size 256
```

If no `--image_path` is specified, the script automatically synthesizes a demo tampered image and performs localization.

**Generated Forensic Artifacts (saved in `results/`):**
- `*_prob_heatmap.png`: Continuous tampering confidence heatmap $[0, 1]$.
- `*_binary_mask.png`: Binarized tampering decision mask.
- `*_boundary_map.png`: Localized boundary transition seams.
- `*_overlay.png`: Visual forensic heatmap blended over the original image.

---

## 🏋️ Training DF-CMNet

### Train on Synthetic Tampering Data (Out-of-the-Box)
You can train immediately on synthetic tampering data without downloading external datasets:

```bash
python train.py --epochs 10 --batch_size 4 --img_size 256
```

### Train on Benchmark Datasets
Format your dataset (e.g., CASIA v2.0, Columbia, NIST'16, or COVERAGE) into `images/` and `masks/`:

```bash
python train.py --data_dir /path/to/dataset --epochs 50 --batch_size 16 --img_size 512
```

---

## 📊 Experimental Evaluation Suite

Run the automated evaluation benchmark suite covering intra-dataset localization, cross-dataset transferability, tamper scale sensitivity, and degradation stress-testing (JPEG re-compression, Gaussian blur, additive noise, geometric transformations):

```bash
python experiments/benchmark_suite.py
```

Generate visual forensic comparison plates:
```bash
python experiments/generate_qualitative_suite.py
```

---

## 📜 License

This project is open-source and released under the [MIT License](LICENSE).
