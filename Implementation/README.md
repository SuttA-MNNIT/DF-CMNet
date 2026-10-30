# DF-CMNet v2: Dual-Domain Frequency-Spatial Cross-Modulation Network for Passive Image Forgery Localization

An open-source PyTorch implementation of **DF-CMNet v2**, an end-to-end deep learning framework designed for passive (blind) image forgery detection and pixel-level localization across various manipulation types (splicing, copy-move cloning, and generative inpainting).

---

## 🌟 Key Architecture & Highlights

DF-CMNet v2 resolves structural bottlenecks in passive image forensics (such as semantic edge bias, cross-domain decoupling, and vanishing gradients on authentic images) through a coordinated dual-stream design:

1. **Learnable High-Frequency Residual Extractor (LHFRE):**
   - Integrates Bayar constrained convolution kernels to suppress semantic content and extract low-level sensor noise footprints (PRNU, demosaicing artifacts).
   - Decomposes residuals into directional frequency sub-bands ($LH, HL, HH$) via multi-scale 2D Discrete Wavelet Transform (2D-DWT).

2. **Dual-Attention Cross-Domain Modulation (DACM / CDBM):**
   - Coordinates reciprocal spatial and channel cross-attention between spatial representations and spectral sub-bands.
   - Selectively recalibrates spatial representations using high-frequency anomaly cues and vice versa.

3. **Boundary-Aware Residual Guidance Module (BARGM):**
   - Employs Atrous Spatial Pyramid Pooling (ASPP) to capture multi-scale context along subtle donor-host transition seams.
   - Enforces sharp localization along blended or feathered manipulation margins.

4. **Compound Multi-Task Objective with Authentic Suppression Penalty (ASP):**
   - Jointly optimizes Focal Loss, Soft-IoU / Dice, and morphological boundary supervision.
   - Integrates an authentic suppression penalty that completely prevents zero-gradient pathologies on negative pristine imagery.

---

## 📁 Repository Structure

```
├── models/                      # Deep learning architecture
│   ├── __init__.py
│   ├── bayar_conv.py            # Bayar constrained convolution (prediction-error residual extraction)
│   ├── wavelet_dwt.py           # 2D Haar Discrete Wavelet Transform (sub-band decomposition)
│   ├── cdbm.py / dacm.py        # Dual-Attention Cross-Domain Modulation (reciprocal spatial & channel attention)
│   ├── bargm.py                 # Boundary-Aware Residual Guidance Module (ASPP edge head & seam gating)
│   └── df_cmnet.py              # Full DF-CMNet pipeline & progressive multi-scale decoder (V1..V8 ablation ready)
│
├── losses/                      # Loss formulations
│   ├── __init__.py
│   └── compound_loss.py         # Multi-task loss: Focal Loss + Soft-IoU + ASP + Boundary BCE
│
├── dataset/                     # Data loading & augmentation
│   ├── __init__.py
│   ├── forgery_dataset.py       # Real benchmark loader (auto-discovers CASIA, Columbia, COVERAGE, IMD2020)
│   └── synthetic_tamper.py      # Synthetic tamper generator for dry-run unit tests
│
├── experiments/                 # Comprehensive evaluation & visualization suite
│   ├── benchmark_suite.py       # Real multi-benchmark evaluation suite (Hausdorff 95, GFLOPs, robustness)
│   └── generate_qualitative_suite.py # Zero-leakage qualitative & robustness visual generator
│
├── checkpoints/                 # Model checkpoints directory (.gitkeep)
├── results/                     # Evaluation metrics, JSON logs, & qualitative heatmaps (.gitkeep)
├── Colab_H100_Runner.ipynb      # Master Google Colab H100 / A100 execution notebook
├── download_datasets.py         # Automated Kagglehub & mirror benchmark dataset downloader
├── verify_model.py              # Unit tests, gradient flow checks, and throughput benchmarking
├── train.py                     # Mixed-precision training pipeline with ablation switches (V1..V8)
├── infer.py                     # Single-image inference CLI producing heatmaps, masks, and boundary overlays
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

### 1. Google Colab Execution (H100 / A100 / T4)
For complete end-to-end training and evaluation, open `Colab_H100_Runner.ipynb` in Google Colab:
1. Open `Colab_H100_Runner.ipynb` in Colab.
2. Select an H100 or A100 GPU runtime.
3. Run all cells sequentially.

---

### 2. Verify Architecture & Gradient Flow
Run the automated verification suite to test forward pass, backward pass, gradient propagation across both spatial and frequency streams, parameter count, and inference throughput:

```bash
python verify_model.py
```

---

### 3. Training DF-CMNet v2

To train on a real forensic benchmark dataset (e.g. CASIA v2.0):
```bash
python train.py \
    --data_dir ./data/CASIA_v2 \
    --variant V8 \
    --epochs 60 \
    --batch_size 32 \
    --lr 1e-4 \
    --img_size 256 \
    --crop_mode native \
    --amp \
    --save_dir checkpoints \
    --ckpt_name colab_h100_best_model.pth
```

---

### 4. Experimental Benchmarks & Evaluation

Run the automated evaluation benchmark suite:
```bash
python experiments/benchmark_suite.py \
    --data_root ./data \
    --checkpoint checkpoints/colab_h100_best_model.pth \
    --img_size 256 \
    --batch_size 32 \
    --save_dir results
```

Generate held-out qualitative diagnostic visual panels:
```bash
python experiments/generate_qualitative_suite.py \
    --checkpoint checkpoints/colab_h100_best_model.pth \
    --data_dir ./data/CASIA_v2 \
    --output_dir results/qualitative \
    --num_cases 8
```

---

### 5. Single-Image Inference
Localize tampering on any test image and generate visual forensic artifacts:

```bash
python infer.py \
    --image_path ./data/CASIA_v2/Tp/example.jpg \
    --checkpoint checkpoints/colab_h100_best_model.pth \
    --output_dir results/inference
```

**Generated Forensic Artifacts (saved in `results/inference/`):**
- `*_prob_heatmap.png`: Continuous tampering confidence heatmap $[0, 1]$ (Jet colormap).
- `*_binary_mask.png`: Binarized tampering decision mask.
- `*_boundary_map.png`: Localized boundary transition seams.
- `*_overlay.png`: Visual forensic heatmap blended over the original query image.

---

## 📜 License

This project is open-source and released under the [MIT License](LICENSE).
