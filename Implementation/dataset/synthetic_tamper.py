import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class SyntheticForgeryDataset(Dataset):
    """
    On-the-fly Synthetic Image Forgery Dataset Generator.
    Creates realistic splicing and copy-move manipulations on synthetic textures or background images.
    Enables immediate out-of-the-box model training and testing without downloading large benchmarks.
    """
    def __init__(self, num_samples=100, img_size=(256, 256), is_train=True):
        self.num_samples = num_samples
        self.img_size = img_size
        self.is_train = is_train

    def __len__(self):
        return self.num_samples

    def generate_synthetic_scene(self):
        w, h = self.img_size
        # Generate base natural textured background
        base = np.zeros((h, w, 3), dtype=np.uint8)
        # Create gradients / patterns
        r_grad = np.linspace(30, 220, w).astype(np.uint8)
        g_grad = np.linspace(40, 200, h).astype(np.uint8)[:, None]
        base[:, :, 0] = r_grad
        base[:, :, 1] = g_grad
        base[:, :, 2] = (r_grad + g_grad) // 2

        # Add Gaussian noise simulating sensor noise pattern
        noise = np.random.normal(0, 8, (h, w, 3))
        base = np.clip(base + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(base)

        # Draw some authentic background geometric objects
        draw = ImageDraw.Draw(img)
        for _ in range(4):
            x1, y1 = random.randint(0, w - 50), random.randint(0, h - 50)
            x2, y2 = x1 + random.randint(20, 50), y1 + random.randint(20, 50)
            color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
            draw.rectangle([x1, y1, x2, y2], fill=color)

        return img

    def synthesize_forgery(self, img):
        w, h = self.img_size
        mask = Image.new('L', (w, h), 0)
        mask_draw = ImageDraw.Draw(mask)

        is_tampered = random.random() < 0.8
        if not is_tampered:
            return img, mask

        # Splicing / Copy-move simulation
        tamper_type = random.choice(['splicing', 'copy_move'])
        box_w = random.randint(w // 8, w // 3)
        box_h = random.randint(h // 8, h // 3)

        if tamper_type == 'copy_move':
            # Select source region
            src_x = random.randint(0, w - box_w)
            src_y = random.randint(0, h - box_h)
            crop_patch = img.crop((src_x, src_y, src_x + box_w, src_y + box_h))
            
            # Destination region
            dst_x = random.randint(0, w - box_w)
            dst_y = random.randint(0, h - box_h)
            
            # Slight post-processing on copied patch (e.g. edge blur)
            crop_patch = crop_patch.filter(ImageFilter.GaussianBlur(radius=0.5))
            img.paste(crop_patch, (dst_x, dst_y))
            mask_draw.rectangle([dst_x, dst_y, dst_x + box_w, dst_y + box_h], fill=255)

        else: # Splicing
            # Create foreign patch with different noise / color profile
            foreign = np.random.randint(60, 240, (box_h, box_w, 3), dtype=np.uint8)
            foreign_patch = Image.fromarray(foreign)
            foreign_patch = foreign_patch.filter(ImageFilter.GaussianBlur(radius=1.0))
            
            dst_x = random.randint(0, w - box_w)
            dst_y = random.randint(0, h - box_h)
            img.paste(foreign_patch, (dst_x, dst_y))
            mask_draw.rectangle([dst_x, dst_y, dst_x + box_w, dst_y + box_h], fill=255)

        return img, mask

    def __getitem__(self, idx):
        base_img = self.generate_synthetic_scene()
        forged_img, mask = self.synthesize_forgery(base_img)

        img_tensor = TF.to_tensor(forged_img)
        mask_np = (np.array(mask, dtype=np.float32) > 127).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)

        return img_tensor, mask_tensor, f"synthetic_{idx:04d}.png"
