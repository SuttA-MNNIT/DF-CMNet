import os
import argparse
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF

from models import DF_CMNet
from dataset.synthetic_tamper import SyntheticForgeryDataset

def run_inference(model, image_path, output_dir="results", img_size=(512, 512), threshold=0.5, device="cpu"):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Load and preprocess
    orig_img = Image.open(image_path).convert('RGB')
    orig_w, orig_h = orig_img.size
    
    resized_img = orig_img.resize(img_size, Image.BILINEAR)
    img_tensor = TF.to_tensor(resized_img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        mask_pred = outputs['mask_pred'].squeeze().cpu().numpy() # [H, W]
        edge_pred = outputs['edge_pred'].squeeze().cpu().numpy() # [H, W]

    # Resize predictions back to original image dimensions
    mask_img = Image.fromarray((mask_pred * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.BILINEAR)
    edge_img = Image.fromarray((edge_pred * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.BILINEAR)

    # Binary mask
    mask_np = np.array(mask_img) / 255.0
    bin_mask = ((mask_np > threshold) * 255).astype(np.uint8)
    bin_mask_img = Image.fromarray(bin_mask)

    # Create overlay visualization (red mask overlay on original image)
    orig_np = np.array(orig_img).astype(np.float32)
    overlay_np = orig_np.copy()
    tampered_indices = mask_np > threshold
    # Blend with red: [255, 0, 0]
    overlay_np[tampered_indices, 0] = overlay_np[tampered_indices, 0] * 0.4 + 255 * 0.6
    overlay_np[tampered_indices, 1] = overlay_np[tampered_indices, 1] * 0.4
    overlay_np[tampered_indices, 2] = overlay_np[tampered_indices, 2] * 0.4
    overlay_img = Image.fromarray(np.clip(overlay_np, 0, 255).astype(np.uint8))

    # Save outputs
    heatmap_path = os.path.join(output_dir, f"{base_name}_prob_heatmap.png")
    bin_path = os.path.join(output_dir, f"{base_name}_binary_mask.png")
    edge_path = os.path.join(output_dir, f"{base_name}_boundary_map.png")
    overlay_path = os.path.join(output_dir, f"{base_name}_overlay.png")

    mask_img.save(heatmap_path)
    bin_mask_img.save(bin_path)
    edge_img.save(edge_path)
    overlay_img.save(overlay_path)

    print(f"[+] Inference complete for: {image_path}")
    print(f"  - Probability Heatmap: {heatmap_path}")
    print(f"  - Binary Mask:        {bin_path}")
    print(f"  - Boundary Map:       {edge_path}")
    print(f"  - Tamper Overlay:     {overlay_path}")

def main():
    parser = argparse.ArgumentParser(description="Inference with DF-CMNet")
    parser.add_argument("--image_path", type=str, default=None, help="Path to input image to test")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained checkpoint (.pth)")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save output masks")
    parser.add_argument("--threshold", type=float, default=0.5, help="Tampering detection threshold")
    parser.add_argument("--img_size", type=int, default=512, help="Inference resolution")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running inference on device: {device}")

    # Load model
    model = DF_CMNet(pretrained=False).to(device)
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"[*] Loading model weights from: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
    else:
        print("[!] Warning: No checkpoint specified. Running with initialized weights.")

    # Image selection
    if args.image_path and os.path.exists(args.image_path):
        test_path = args.image_path
    else:
        # Generate a demo synthetic tampered image
        print("[*] No test image specified. Generating a synthetic tampered image demo...")
        os.makedirs("demo", exist_ok=True)
        syn_dataset = SyntheticForgeryDataset(num_samples=1, img_size=(args.img_size, args.img_size))
        img_tensor, mask_tensor, fname = syn_dataset[0]
        test_path = os.path.join("demo", "test_tampered_sample.png")
        TF.to_pil_image(img_tensor).save(test_path)
        TF.to_pil_image(mask_tensor).save(os.path.join("demo", "ground_truth_mask.png"))
        print(f"[*] Demo image saved to {test_path}")

    run_inference(model, test_path, output_dir=args.output_dir, 
                  img_size=(args.img_size, args.img_size), threshold=args.threshold, device=device)

if __name__ == "__main__":
    main()
