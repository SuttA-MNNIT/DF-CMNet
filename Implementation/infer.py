"""
Single-Image Forensic Inference CLI for DF-CMNet v2.

Runs pure forward inference on an input test image and exports:
  1. Continuous tampering probability heatmap (Jet colormap)
  2. Binarized localization mask (tau threshold)
  3. BARGM boundary transition seam map
  4. Visual forensic contour overlay on the original query image

Usage:
  python infer.py --image_path ./data/CASIA_v2/Tp/example.jpg --checkpoint checkpoints/colab_h100_best_model.pth --output_dir results/inference
"""

import os
import argparse
import numpy as np
from PIL import Image, ImageFilter
import torch
import torchvision.transforms.functional as TF

from models import DF_CMNet


def apply_jet_colormap(prob_2d):
    """Maps a 2D probability array in [0, 1] to an RGB heatmap (uint8 [H, W, 3])."""
    v = np.clip(prob_2d, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * v - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * v - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * v - 1.0), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255.0).astype(np.uint8)


def create_forensic_overlay(orig_img, bin_mask_2d):
    """Blends predicted tampering region and boundary contour over original RGB image."""
    base = np.array(orig_img.convert("RGB"), dtype=np.float32)
    mask_bool = bin_mask_2d > 0.5

    mask_img = Image.fromarray((mask_bool.astype(np.uint8) * 255))
    dil = np.array(mask_img.filter(ImageFilter.MaxFilter(5))) > 127
    ero = np.array(mask_img.filter(ImageFilter.MinFilter(5))) > 127
    contour = dil & (~ero)

    overlay = base.copy()
    overlay[mask_bool, 0] = 0.65 * overlay[mask_bool, 0] + 0.35 * 40.0
    overlay[mask_bool, 1] = 0.65 * overlay[mask_bool, 1] + 0.35 * 220.0
    overlay[mask_bool, 2] = 0.65 * overlay[mask_bool, 2] + 0.35 * 80.0

    overlay[contour, 0] = 20.0
    overlay[contour, 1] = 255.0
    overlay[contour, 2] = 60.0

    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


def run_inference(model, image_path, output_dir="results/inference", img_size=(256, 256), threshold=0.5, device="cpu"):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    orig_img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = orig_img.size

    resized_img = orig_img.resize(img_size, Image.BILINEAR)
    img_tensor = TF.to_tensor(resized_img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        mask_pred = outputs["mask_pred"].squeeze().float().cpu().numpy()
        edge_pred = outputs["edge_pred"].squeeze().float().cpu().numpy()

    # Bilinear interpolate predicted maps back to original resolution
    mask_full = np.array(Image.fromarray(mask_pred).resize((orig_w, orig_h), Image.BILINEAR))
    edge_full = np.array(Image.fromarray(edge_pred).resize((orig_w, orig_h), Image.BILINEAR))
    bin_mask = (mask_full >= threshold).astype(np.float32)

    # Save output artifacts
    heatmap_pil = Image.fromarray(apply_jet_colormap(mask_full))
    bin_mask_pil = Image.fromarray((bin_mask * 255).astype(np.uint8))
    boundary_pil = Image.fromarray(apply_jet_colormap(edge_full))
    overlay_pil = create_forensic_overlay(orig_img, bin_mask)

    p_heatmap = os.path.join(output_dir, f"{base_name}_prob_heatmap.png")
    p_bin = os.path.join(output_dir, f"{base_name}_binary_mask.png")
    p_edge = os.path.join(output_dir, f"{base_name}_boundary_map.png")
    p_overlay = os.path.join(output_dir, f"{base_name}_overlay.png")

    heatmap_pil.save(p_heatmap)
    bin_mask_pil.save(p_bin)
    boundary_pil.save(p_edge)
    overlay_pil.save(p_overlay)

    tampered_ratio = float(np.mean(bin_mask)) * 100.0
    print(f"\n[+] Forensic Localization Results for: {image_path}")
    print(f"  Dimensions         : {orig_w} x {orig_h}")
    print(f"  Tampered Coverage  : {tampered_ratio:.2f}% of pixels (tau={threshold})")
    print(f"  Artifacts saved to : {os.path.abspath(output_dir)}")
    print(f"    - Heatmap        : {p_heatmap}")
    print(f"    - Binary Mask    : {p_bin}")
    print(f"    - Boundary Seams : {p_edge}")
    print(f"    - Visual Overlay : {p_overlay}")


def main():
    parser = argparse.ArgumentParser(description="Run DF-CMNet v2 Forensic Inference")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input test image")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--output_dir", type=str, default="results/inference", help="Directory to save output masks")
    parser.add_argument("--threshold", type=float, default=0.5, help="Tampering detection threshold (default: 0.5)")
    parser.add_argument("--img_size", type=int, default=256, help="Inference resolution (default: 256)")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"[FATAL] Test image '{args.image_path}' not found!")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"[FATAL] Model checkpoint '{args.checkpoint}' not found!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing DF-CMNet v2 on {device}...")
    model = DF_CMNet(variant="V8").to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=False)

    if isinstance(ckpt, dict) and ckpt.get("is_real_training"):
        print(f"[✓] Checkpoint Provenance: Trained on {ckpt.get('num_train_samples')} real samples from {ckpt.get('dataset_path')}")

    run_inference(
        model=model,
        image_path=args.image_path,
        output_dir=args.output_dir,
        img_size=(args.img_size, args.img_size),
        threshold=args.threshold,
        device=device
    )


if __name__ == "__main__":
    main()
