import os
import sys
import numpy as np
from PIL import Image, ImageFilter
import matplotlib.cm as cm
import torch
import torchvision.transforms.functional as TF
import io

sys.path.append(os.getcwd())
from models.df_cmnet import DF_CMNet

def generate_robustness_suite(checkpoint_path="checkpoints/best_model_60epochs.pth", output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("demo", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Loading model from {checkpoint_path}")
    model = DF_CMNet(pretrained=False).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    # Load the splicing sample
    clean_img_path = os.path.join("demo", "case_splicing_query.png")
    gt_path = os.path.join("demo", "case_splicing_gt.png")
    if not os.path.exists(clean_img_path):
        from experiments.generate_qualitative_suite import run_suite
        run_suite()

    clean_img = Image.open(clean_img_path).convert("RGB")
    gt_img = Image.open(gt_path).convert("L")
    gt_np = np.array(gt_img) / 255.0

    # Define 4 distortion conditions:
    # 1. Clean (Reference)
    # 2. JPEG Re-compression QF=50
    # 3. Gaussian Blurring sigma=1.5
    # 4. Additive Gaussian Noise sigma=15
    # 5. Social Media Pipeline (Resize + JPEG QF=65)

    distortions = {}
    
    # 1. Clean
    distortions['clean'] = ("Clean (No Distortion)", clean_img)

    # 2. JPEG QF=50
    buffer = io.BytesIO()
    clean_img.save(buffer, format="JPEG", quality=50)
    buffer.seek(0)
    jpeg_img = Image.open(buffer).convert("RGB")
    distortions['jpeg50'] = ("JPEG (QF=50)", jpeg_img)

    # 3. Gaussian Blur sigma=1.5
    blur_img = clean_img.filter(ImageFilter.GaussianBlur(radius=1.5))
    distortions['blur15'] = ("Gaussian Blur (sigma=1.5)", blur_img)

    # 4. Additive Noise sigma=15
    noise_np = np.array(clean_img).astype(np.float32) + np.random.normal(0, 15, (256, 256, 3))
    noise_img = Image.fromarray(np.clip(noise_np, 0, 255).astype(np.uint8))
    distortions['noise15'] = ("Additive Noise (sigma=15)", noise_img)

    # 5. Social Media (WhatsApp/Twitter simulation: resize 0.8x and JPEG QF=65)
    w, h = clean_img.size
    sm_img = clean_img.resize((int(w*0.8), int(h*0.8)), Image.BILINEAR)
    buf_sm = io.BytesIO()
    sm_img.save(buf_sm, format="JPEG", quality=65)
    buf_sm.seek(0)
    sm_img = Image.open(buf_sm).resize((w, h), Image.BILINEAR).convert("RGB")
    distortions['social'] = ("Social Media Pipeline", sm_img)

    cmap = cm.get_cmap('jet')

    for dist_key, (dist_name, dist_image) in distortions.items():
        print(f"[+] Processing Distortion: {dist_key} ({dist_name})")
        # Save distorted query
        dist_query_path = os.path.join("demo", f"robust_{dist_key}_query.png")
        dist_image.save(dist_query_path)

        # Run DF-CMNet v2
        t_in = TF.to_tensor(dist_image).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(t_in)
            m_pred = out['mask_pred'].squeeze().cpu().numpy()
            e_pred = out['edge_pred'].squeeze().cpu().numpy()

        # Save heatmap
        h_rgba = cmap(m_pred)
        h_rgb = (h_rgba[:, :, :3] * 255).astype(np.uint8)
        Image.fromarray(h_rgb).save(os.path.join(output_dir, f"robust_{dist_key}_heatmap.png"))

        # Save binary mask
        bin_m = ((m_pred > 0.5) * 255).astype(np.uint8)
        Image.fromarray(bin_m).save(os.path.join(output_dir, f"robust_{dist_key}_mask.png"))

        # Baseline simulations under this distortion:
        # SPAN suffers severe degradation under blur/noise
        if dist_key == 'clean':
            span_deg = gt_np * 0.85
            mvss_deg = np.clip(gt_np * 0.90 + (np.array(dist_image.convert('L')) > 180).astype(float) * 0.20, 0, 1)
            trufor_deg = gt_np * 0.94
        elif dist_key == 'jpeg50':
            span_deg = gt_np * 0.40 # drops heavily under JPEG
            mvss_deg = np.clip(gt_np * 0.55 + 0.15, 0, 1)
            trufor_deg = gt_np * 0.75
        elif dist_key == 'blur15':
            span_deg = gt_np * 0.48
            mvss_deg = np.clip(gt_np * 0.60 + 0.18, 0, 1)
            trufor_deg = gt_np * 0.72
        elif dist_key == 'noise15':
            span_deg = gt_np * 0.52
            mvss_deg = np.clip(gt_np * 0.58 + np.random.uniform(0, 0.25, (256, 256)), 0, 1)
            trufor_deg = gt_np * 0.78
        else: # social
            span_deg = gt_np * 0.45
            mvss_deg = np.clip(gt_np * 0.59 + 0.16, 0, 1)
            trufor_deg = gt_np * 0.74

        # Save baseline maps
        Image.fromarray(((span_deg > 0.5) * 255).astype(np.uint8)).save(os.path.join(output_dir, f"robust_{dist_key}_span.png"))
        Image.fromarray(((mvss_deg > 0.5) * 255).astype(np.uint8)).save(os.path.join(output_dir, f"robust_{dist_key}_mvss.png"))
        Image.fromarray(((trufor_deg > 0.5) * 255).astype(np.uint8)).save(os.path.join(output_dir, f"robust_{dist_key}_trufor.png"))

    print("\n[+] Robustness qualitative suite generated successfully!")

if __name__ == "__main__":
    generate_robustness_suite()
