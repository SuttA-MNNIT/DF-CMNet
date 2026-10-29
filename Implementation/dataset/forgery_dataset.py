import os
import glob
import io
import random
import numpy as np
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class PassiveForgeryDataset(Dataset):
    """
    Standard dataset loader for passive image forgery detection and localization.
    Compatible with CASIA v1.0, CASIA v2.0, Columbia, NIST'16, COVERAGE, or custom datasets.
    
    Expected Directory Structure:
      root_dir/
        images/
          001.jpg, 002.png, ...
        masks/
          001.png, 002.png, ... (white pixels = forged, black pixels = authentic)
    """
    def __init__(self, root_dir=None, img_size=(512, 512), is_train=True, 
                 simulate_jpeg=False, simulate_blur=False):
        self.root_dir = root_dir
        self.img_size = img_size
        self.is_train = is_train
        self.simulate_jpeg = simulate_jpeg
        self.simulate_blur = simulate_blur
        
        self.samples = []
        if root_dir and os.path.exists(root_dir):
            img_dir = os.path.join(root_dir, 'images')
            mask_dir = os.path.join(root_dir, 'masks')
            if os.path.exists(img_dir) and os.path.exists(mask_dir):
                valid_exts = ('*.jpg', '*.jpeg', '*.png', '*.tif', '*.bmp')
                img_paths = []
                for ext in valid_exts:
                    img_paths.extend(glob.glob(os.path.join(img_dir, ext)))
                
                for ipath in img_paths:
                    base_name = os.path.splitext(os.path.basename(ipath))[0]
                    # Find matching mask
                    mask_candidate = os.path.join(mask_dir, base_name + '.png')
                    if not os.path.exists(mask_candidate):
                        mask_candidate = os.path.join(mask_dir, base_name + '_gt.png')
                    if not os.path.exists(mask_candidate):
                        mask_candidate = os.path.join(mask_dir, base_name + '.jpg')
                    
                    if os.path.exists(mask_candidate):
                        self.samples.append((ipath, mask_candidate))

    def __len__(self):
        return len(self.samples)

    def apply_distortions(self, image):
        # 1. JPEG Compression Simulation
        if self.simulate_jpeg and random.random() < 0.5:
            qf = random.randint(50, 95)
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=qf)
            buffer.seek(0)
            image = Image.open(buffer).convert('RGB')

        # 2. Gaussian Blur Simulation
        if self.simulate_blur and random.random() < 0.3:
            radius = random.uniform(0.5, 2.0)
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))

        return image

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        # Resize
        image = image.resize(self.img_size, Image.BILINEAR)
        mask = mask.resize(self.img_size, Image.NEAREST)

        # Distortions during training
        if self.is_train:
            image = self.apply_distortions(image)
            # Random horizontal flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            # Random vertical flip
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

        # Convert to tensors
        img_tensor = TF.to_tensor(image)  # [3, H, W] in [0, 1]
        mask_np = np.array(mask, dtype=np.float32)
        # Normalize mask to binary {0.0, 1.0}
        mask_binary = (mask_np > 127).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0)  # [1, H, W]

        return img_tensor, mask_tensor, os.path.basename(img_path)
