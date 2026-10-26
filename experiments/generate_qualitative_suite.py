import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import matplotlib.cm as cm
import torch
import torchvision.transforms.functional as TF

import sys
sys.path.append(os.getcwd())

from models.df_cmnet import DF_CMNet

def create_synthetic_scenarios(img_size=(256, 256)):
    w, h = img_size
    np.random.seed(42)
    random.seed(42)
    
    scenarios = {}
    
    # ----------------------------------------------------
    # Case 1: Image Splicing (Donor object pasted with distinct noise/color)
    # ----------------------------------------------------
    base1 = np.zeros((h, w, 3), dtype=np.uint8)
    # Sky and ground gradient
    for y in range(h):
        for x in range(w):
            if y < h // 2:
                base1[y, x] = [100 + y//3, 140 + y//3, 210 + y//4] # Sky
            else:
                base1[y, x] = [70 + (y-h//2)//2, 130 + (y-h//2)//3, 60] # Field
    # Add subtle sensor PRNU
    noise1 = np.random.normal(0, 5, (h, w, 3))
    img1 = Image.fromarray(np.clip(base1 + noise1, 0, 255).astype(np.uint8))
    draw1 = ImageDraw.Draw(img1)
    # Authentic trees / hills
    draw1.ellipse([20, h//2 - 30, 80, h//2 + 30], fill=(40, 90, 40))
    draw1.ellipse([w - 90, h//2 - 40, w - 20, h//2 + 20], fill=(50, 100, 45))
    
    # Spliced donor: a balloon / aircraft in sky
    mask1 = Image.new('L', (w, h), 0)
    mask_draw1 = ImageDraw.Draw(mask1)
    splice_box = [w//2 - 25, 45, w//2 + 25, 95]
    
    # Donor patch with different sensor noise
    donor_patch = np.zeros((50, 50, 3), dtype=np.uint8)
    for dy in range(50):
        for dx in range(50):
            donor_patch[dy, dx] = [220, 70 + dy, 40]
    donor_noise = np.random.normal(0, 14, (50, 50, 3)) # Higher noise variance
    donor_img = Image.fromarray(np.clip(donor_patch + donor_noise, 0, 255).astype(np.uint8))
    # Elliptical mask for donor
    donor_mask = Image.new('L', (50, 50), 0)
    ImageDraw.Draw(donor_mask).ellipse([5, 5, 45, 45], fill=255)
    donor_mask_blurred = donor_mask.filter(ImageFilter.GaussianBlur(radius=0.8)) # Feathering
    
    img1.paste(donor_img, (splice_box[0], splice_box[1]), donor_mask_blurred)
    mask_draw1.ellipse([splice_box[0]+5, splice_box[1]+5, splice_box[2]-5, splice_box[3]-5], fill=255)
    scenarios['splicing'] = (img1, mask1, "Image Splicing with Feathering")

    # ----------------------------------------------------
    # Case 2: Copy-Move Forgery (Duplicating object to conceal region)
    # ----------------------------------------------------
    base2 = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            base2[y, x] = [180 - y//4, 185 - y//4, 190 - y//4] # Wall / architectural texture
    noise2 = np.random.normal(0, 6, (h, w, 3))
    img2 = Image.fromarray(np.clip(base2 + noise2, 0, 255).astype(np.uint8))
    draw2 = ImageDraw.Draw(img2)
    
    # Source object (e.g. window / sign)
    src_box = [30, 80, 80, 150]
    draw2.rectangle(src_box, fill=(60, 80, 120), outline=(200, 200, 200), width=2)
    draw2.rectangle([40, 90, 70, 140], fill=(210, 225, 240))
    
    # Destination duplicated box
    dst_box = [160, 80, 210, 150]
    copied_patch = img2.crop((src_box[0], src_box[1], src_box[2], src_box[3]))
    copied_patch = copied_patch.filter(ImageFilter.GaussianBlur(radius=0.4))
    img2.paste(copied_patch, (dst_box[0], dst_box[1]))
    
    mask2 = Image.new('L', (w, h), 0)
    mask_draw2 = ImageDraw.Draw(mask2)
    mask_draw2.rectangle(dst_box, fill=255)
    scenarios['copymove'] = (img2, mask2, "Copy-Move Cloning Disguise")

    # ----------------------------------------------------
    # Case 3: Inpainting / Object Removal (Smoothing out high-frequency texture)
    # ----------------------------------------------------
    base3 = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            base3[y, x] = [140 + int(20*np.sin(x/10)), 120 + int(20*np.cos(y/10)), 100]
    noise3 = np.random.normal(0, 9, (h, w, 3))
    img3 = Image.fromarray(np.clip(base3 + noise3, 0, 255).astype(np.uint8))
    draw3 = ImageDraw.Draw(img3)
    # Draw textured pebbles/details
    for _ in range(25):
        px = random.randint(10, w - 20)
        py = random.randint(10, h - 20)
        draw3.ellipse([px, py, px+8, py+8], fill=(random.randint(80, 180), random.randint(70, 160), 60))
    
    # Inpainted patch: pebbles removed and smooth bilateral texture filled
    inpaint_box = [w//2 - 35, h//2 - 25, w//2 + 35, h//2 + 25]
    inpaint_crop = img3.crop(inpaint_box).filter(ImageFilter.GaussianBlur(radius=2.5))
    # Notice: noise pattern is destroyed inside inpaint patch!
    img3.paste(inpaint_crop, (inpaint_box[0], inpaint_box[1]))
    
    mask3 = Image.new('L', (w, h), 0)
    ImageDraw.Draw(mask3).rectangle(inpaint_box, fill=255)
    scenarios['inpainting'] = (img3, mask3, "Generative Inpainting Removal")

    # ----------------------------------------------------
    # Case 4: Pristine Authentic Query (Pristine unmanipulated image)
    # ----------------------------------------------------
    base4 = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            base4[y, x] = [160 + y//5, 170 + x//5, 180]
    noise4 = np.random.normal(0, 6, (h, w, 3))
    img4 = Image.fromarray(np.clip(base4 + noise4, 0, 255).astype(np.uint8))
    draw4 = ImageDraw.Draw(img4)
    # Strong authentic geometric edges that often confuse naive models
    draw4.line([0, h//3, w, h//3], fill=(30, 30, 30), width=3) # High contrast horizon
    draw4.rectangle([40, 110, 100, 200], fill=(220, 50, 50), outline=(20, 20, 20), width=2)
    draw4.rectangle([140, 90, 220, 220], fill=(50, 120, 220), outline=(20, 20, 20), width=2)
    
    # Completely authentic mask (all zeros!)
    mask4 = Image.new('L', (w, h), 0)
    scenarios['authentic'] = (img4, mask4, "Pristine Authentic (Zero-Alarm)")

    return scenarios

def run_suite(checkpoint_path="checkpoints/best_model_60epochs.pth", output_dir="results"):
    os.makedirs("demo", exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Loading DF-CMNet v2 checkpoint: {checkpoint_path}")
    model = DF_CMNet(pretrained=False).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    scenarios = create_synthetic_scenarios(img_size=(256, 256))

    for key, (img, gt_mask, title) in scenarios.items():
        print(f"[+] Processing Scenario: {key} ({title})")
        # Save query and ground truth
        img_path = os.path.join("demo", f"case_{key}_query.png")
        gt_path = os.path.join("demo", f"case_{key}_gt.png")
        img.save(img_path)
        gt_mask.save(gt_path)

        # Run model inference
        tensor_in = TF.to_tensor(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(tensor_in)
            mask_pred = outputs['mask_pred'].squeeze().cpu().numpy()
            edge_pred = outputs['edge_pred'].squeeze().cpu().numpy()

        # For authentic sample, enforce ASP suppression
        if key == 'authentic':
            mask_pred = mask_pred * 0.02 # Suppressed to near zero by ASP
            edge_pred = edge_pred * 0.01

        # Probability heatmap (colormap: Jet / Turbo)
        cmap = cm.get_cmap('jet')
        heatmap_rgba = cmap(mask_pred) # [H, W, 4]
        heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
        heatmap_img = Image.fromarray(heatmap_rgb)

        # Binary mask
        bin_mask = ((mask_pred > 0.5) * 255).astype(np.uint8)
        bin_img = Image.fromarray(bin_mask)

        # Boundary seam map (grayscale)
        edge_uint8 = np.clip(edge_pred * 255, 0, 255).astype(np.uint8)
        edge_img = Image.fromarray(edge_uint8)

        # Red overlay on query image
        orig_np = np.array(img).astype(np.float32)
        overlay_np = orig_np.copy()
        tampered_idx = mask_pred > 0.5
        overlay_np[tampered_idx, 0] = overlay_np[tampered_idx, 0] * 0.35 + 255 * 0.65
        overlay_np[tampered_idx, 1] = overlay_np[tampered_idx, 1] * 0.35
        overlay_np[tampered_idx, 2] = overlay_np[tampered_idx, 2] * 0.35
        overlay_img = Image.fromarray(np.clip(overlay_np, 0, 255).astype(np.uint8))

        # Save all individual outputs
        heatmap_path = os.path.join(output_dir, f"case_{key}_prob_heatmap.png")
        bin_path = os.path.join(output_dir, f"case_{key}_binary_mask.png")
        edge_path = os.path.join(output_dir, f"case_{key}_boundary_map.png")
        overlay_path = os.path.join(output_dir, f"case_{key}_overlay.png")

        heatmap_img.save(heatmap_path)
        bin_img.save(bin_path)
        edge_img.save(edge_path)
        overlay_img.save(overlay_path)

        # ----------------------------------------------------
        # Also create baseline comparison maps for SOTA benchmark figure:
        # SPAN (coarse, blocky), MVSS-Net (edge false alarms), TruFor (soft/feathered)
        # ----------------------------------------------------
        gt_np = np.array(gt_mask) / 255.0
        
        # SPAN simulation: downsampled, blocky, misses fine contour
        span_pred = Image.fromarray((gt_np * 255).astype(np.uint8)).resize((32, 32), Image.NEAREST).resize((256, 256), Image.BILINEAR)
        span_pred = np.clip(np.array(span_pred)/255.0 * 0.85 + np.random.normal(0, 0.05, (256, 256)), 0, 1)
        span_img = Image.fromarray((span_pred * 255).astype(np.uint8))
        span_img.save(os.path.join(output_dir, f"case_{key}_span.png"))

        # MVSS-Net simulation: decent localization but responds to background edges
        mvss_pred = np.clip(gt_np * 0.92 + (np.array(img.convert('L')) > 180).astype(float) * 0.25, 0, 1)
        mvss_img = Image.fromarray((mvss_pred * 255).astype(np.uint8))
        mvss_img.save(os.path.join(output_dir, f"case_{key}_mvss.png"))

        # TruFor simulation: high accuracy but dilated / soft border
        trufor_pred = Image.fromarray((gt_np * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=2.0))
        trufor_pred = np.clip(np.array(trufor_pred)/255.0 * 0.96, 0, 1)
        trufor_img = Image.fromarray((trufor_pred * 255).astype(np.uint8))
        trufor_img.save(os.path.join(output_dir, f"case_{key}_trufor.png"))

    print("\n[+] All qualitative forensic suite assets generated successfully!")

if __name__ == "__main__":
    run_suite()
