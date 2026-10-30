"""
Genuine Zero-Leakage Qualitative & Robustness Visualization Suite for DF-CMNet v2.

IMPORTANT:
  - This script runs strictly in `model.eval()` / `torch.no_grad()` inference mode.
  - It NEVER trains or fine-tunes the model on test images.
  - It NEVER fabricates baseline masks from ground-truth erosion/noise.
  - All DF-CMNet v2 heatmaps, binary masks, BARGM boundary maps, and contour overlays
    are produced solely by forward passes of the trained checkpoint on held-out test images.

Usage on Google Colab H100:
    python experiments/generate_qualitative_suite.py --checkpoint checkpoints/best_model.pth --data_dir ./data/CASIA_v2
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image, ImageFilter

import torch
import torchvision.transforms.functional as TF

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.df_cmnet import DF_CMNet
from dataset.forgery_dataset import PassiveForgeryDataset, apply_test_distortion


def apply_jet_colormap(prob_2d):
    """Maps a 2D probability array in [0, 1] to an RGB heatmap (uint8 [H, W, 3])."""
    v = np.clip(prob_2d, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * v - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * v - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * v - 1.0), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255.0).astype(np.uint8)


def create_forensic_overlay(img_pil, bin_mask_2d, edge_prob_2d=None):
    """Overlays predicted binary region (translucent red/green) and boundary contour on original RGB image."""
    base = np.array(img_pil.convert("RGB"), dtype=np.float32)
    mask_bool = bin_mask_2d > 0.5

    # Extract morphological perimeter of predicted binary mask
    mask_img = Image.fromarray((mask_bool.astype(np.uint8) * 255))
    dil = np.array(mask_img.filter(ImageFilter.MaxFilter(5))) > 127
    ero = np.array(mask_img.filter(ImageFilter.MinFilter(5))) > 127
    contour = dil & (~ero)

    overlay = base.copy()
    # Translucent green tint inside predicted forged region
    overlay[mask_bool, 0] = 0.65 * overlay[mask_bool, 0] + 0.35 * 40.0
    overlay[mask_bool, 1] = 0.65 * overlay[mask_bool, 1] + 0.35 * 220.0
    overlay[mask_bool, 2] = 0.65 * overlay[mask_bool, 2] + 0.35 * 80.0

    # Crisp bright green boundary contour
    overlay[contour, 0] = 20.0
    overlay[contour, 1] = 255.0
    overlay[contour, 2] = 60.0

    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


@torch.no_grad()
def predict_single_image(model, img_pil, device, tau=0.5):
    """Runs pure forward inference on a single PIL image (padded to multiple of 32 and cropped back)."""
    model.eval()
    w, h = img_pil.size
    t_in = TF.to_tensor(img_pil).unsqueeze(0).to(device)

    pad_h = (32 - (h % 32)) % 32
    pad_w = (32 - (w % 32)) % 32
    if pad_h > 0 or pad_w > 0:
        t_in = torch.nn.functional.pad(t_in, (0, pad_w, 0, pad_h), mode="reflect")

    outputs = model(t_in)
    prob_map = outputs["mask_pred"][0, 0, :h, :w].float().cpu().numpy()
    edge_map = outputs["edge_pred"][0, 0, :h, :w].float().cpu().numpy()
    bin_mask = (prob_map >= tau).astype(np.float32)

    return prob_map, bin_mask, edge_map


def run_qualitative_inference(
    checkpoint_path="checkpoints/best_model.pth",
    data_dir=None,
    output_dir="results/qualitative",
    img_size=256,
    num_cases=8,
    tau=0.5
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DF_CMNet(variant="V8").to(device)
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)
        print(f"[+] Loaded trained checkpoint from {checkpoint_path} (pure inference mode)")
    else:
        print(f"[!] Warning: Checkpoint '{checkpoint_path}' not found. Using current weights.")
    model.eval()

    if not data_dir or not os.path.exists(data_dir):
        print(f"[!] No real dataset directory provided at '{data_dir}'. Pass --data_dir ./data/CASIA_v2 on Colab.")
        return

    test_ds = PassiveForgeryDataset(
        root_dir=data_dir,
        img_size=(img_size, img_size),
        is_train=False,
        split="test",
        crop_mode="resize",
        simulate_jpeg=False
    )
    if len(test_ds) == 0:
        print(f"[!] No test samples found in {data_dir}.")
        return

    # Select held-out tampered and authentic samples from the test split
    tamp_indices = [i for i, s in enumerate(test_ds.samples) if s[2] == 1]
    auth_indices = [i for i, s in enumerate(test_ds.samples) if s[2] == 0]
    selected = tamp_indices[:max(1, num_cases - 2)] + auth_indices[:2]

    print(f"[*] Running held-out qualitative inference on {len(selected)} test images...")
    for rank, idx in enumerate(selected):
        img_path, mask_path, is_tamp = test_ds.samples[idx]
        img_pil = Image.open(img_path).convert("RGB").resize((img_size, img_size), Image.BILINEAR)
        if mask_path and os.path.exists(mask_path):
            gt_pil = Image.open(mask_path).convert("L").resize((img_size, img_size), Image.NEAREST)
        else:
            gt_pil = Image.new("L", (img_size, img_size), 0)

        prob_map, bin_mask, edge_map = predict_single_image(model, img_pil, device, tau=tau)

        tag = f"test_{rank:02d}_{'tampered' if is_tamp else 'authentic'}"
        img_pil.save(os.path.join(output_dir, f"{tag}_query.png"))
        gt_pil.save(os.path.join(output_dir, f"{tag}_gt.png"))
        Image.fromarray(apply_jet_colormap(prob_map)).save(os.path.join(output_dir, f"{tag}_prob_heatmap.png"))
        Image.fromarray((bin_mask * 255).astype(np.uint8)).save(os.path.join(output_dir, f"{tag}_binary_mask.png"))
        Image.fromarray(apply_jet_colormap(edge_map)).save(os.path.join(output_dir, f"{tag}_boundary_map.png"))
        create_forensic_overlay(img_pil, bin_mask, edge_map).save(os.path.join(output_dir, f"{tag}_overlay.png"))

        # On the first tampered test sample, also generate the 5 real post-processing distortion outputs
        if rank == 0 and is_tamp == 1:
            distortions = [
                ("clean", None),
                ("jpeg50", ("jpeg", 50)),
                ("blur15", ("blur", 1.5)),
                ("noise15", ("noise", 15.0)),
                ("social", ("social", "whatsapp")),
            ]
            for dname, dspec in distortions:
                d_img, d_gt = apply_test_distortion(img_pil.copy(), gt_pil.copy(), dspec)
                d_prob, d_bin, d_edge = predict_single_image(model, d_img, device, tau=tau)
                d_img.save(os.path.join(output_dir, f"robust_{dname}_query.png"))
                d_gt.save(os.path.join(output_dir, f"robust_{dname}_gt.png"))
                Image.fromarray(apply_jet_colormap(d_prob)).save(os.path.join(output_dir, f"robust_{dname}_heatmap.png"))
                Image.fromarray((d_bin * 255).astype(np.uint8)).save(os.path.join(output_dir, f"robust_{dname}_mask.png"))

    print(f"[+] Saved genuine held-out qualitative & robustness panels to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Held-Out Qualitative Inference Suite for DF-CMNet v2")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Trained model checkpoint")
    parser.add_argument("--data_dir", type=str, default="./data/CASIA_v2", help="Dataset directory")
    parser.add_argument("--output_dir", type=str, default="results/qualitative", help="Directory to save visual panels")
    parser.add_argument("--img_size", type=int, default=256, help="Visualization resolution")
    parser.add_argument("--num_cases", type=int, default=8, help="Number of held-out test cases to export")
    args = parser.parse_args()

    run_qualitative_inference(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        img_size=args.img_size,
        num_cases=args.num_cases
    )
