import os
import glob
import io
import random
import numpy as np
from PIL import Image, ImageFilter
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


VALID_EXTS = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')


def _list_images_in_dir(d):
    files = []
    if not os.path.isdir(d):
        return files
    for root, _, fnames in os.walk(d):
        for f in sorted(fnames):
            if os.path.splitext(f)[1].lower() in VALID_EXTS:
                files.append(os.path.join(root, f))
    return sorted(files)


def discover_forensic_samples(root_dir):
    """
    Auto-discovers (image_path, mask_path_or_None, label) tuples across standard forensic dataset layouts:
      1. Standard layout: root_dir/images/ + root_dir/masks/ (and optional root_dir/authentic/)
      2. CASIA v1/v2 layout: Au/ (authentic) + Tp/ (tampered) + Gt/ or 'CASIA 2 Groundtruth'/
      3. Columbia layout: 4cam_auth/ + 4cam_splc/ + edgemask/
      4. COVERAGE layout: image/ (1.tif, 1t.tif) + mask/ (1forged.tif)
    Returns list of (img_path, mask_path_or_None, is_tampered_int).
    """
    if not root_dir or not os.path.exists(root_dir):
        return []

    samples = []

    # Build a lookup of all potential ground-truth mask files in root_dir
    mask_lookup = {}
    for sub in os.listdir(root_dir):
        sub_path = os.path.join(root_dir, sub)
        if os.path.isdir(sub_path) and any(k in sub.lower() for k in ('mask', 'gt', 'groundtruth', 'edgemask')):
            for mpath in _list_images_in_dir(sub_path):
                stem = os.path.splitext(os.path.basename(mpath))[0].lower()
                mask_lookup[stem] = mpath
                # Also index without common suffixes (_gt, _mask, _edgemask, forged)
                for suf in ('_gt', '_mask', '_edgemask', '_edgemask_3', 'forged'):
                    if stem.endswith(suf):
                        mask_lookup[stem[:-len(suf)]] = mpath

    # Case 1: CASIA layout (Au and Tp directories anywhere under root_dir)
    au_dirs, tp_dirs = [], []
    for r, dirs, _ in os.walk(root_dir):
        for d in dirs:
            dl = d.lower()
            if dl in ('au', 'authentic', '4cam_auth', 'pristine', 'real'):
                au_dirs.append(os.path.join(r, d))
            elif dl in ('tp', 'tampered', '4cam_splc', 'spliced', 'forged', 'fake', 'inpainted'):
                tp_dirs.append(os.path.join(r, d))
            elif any(k in dl for k in ('mask', 'gt', 'groundtruth', 'edgemask')):
                for mpath in _list_images_in_dir(os.path.join(r, d)):
                    stem = os.path.splitext(os.path.basename(mpath))[0].lower()
                    mask_lookup[stem] = mpath
                    for suf in ('_gt', '_mask', '_edgemask', '_edgemask_3', 'forged'):
                        if stem.endswith(suf):
                            mask_lookup[stem[:-len(suf)]] = mpath

    if au_dirs or tp_dirs:
        seen = set()
        for ad in au_dirs:
            for ipath in _list_images_in_dir(ad):
                if ipath not in seen:
                    seen.add(ipath)
                    samples.append((ipath, None, 0))
        for td in tp_dirs:
            for ipath in _list_images_in_dir(td):
                if ipath not in seen:
                    seen.add(ipath)
                    stem = os.path.splitext(os.path.basename(ipath))[0].lower()
                    mpath = mask_lookup.get(stem, None)
                    if mpath is not None:
                        samples.append((ipath, mpath, 1))
        if samples:
            return samples

    # Case 2: COVERAGE layout (image/ with '1.tif' authentic and '1t.tif' tampered, mask/ with '1forged.tif')
    cov_img_dir = os.path.join(root_dir, 'image')
    cov_mask_dir = os.path.join(root_dir, 'mask')
    if os.path.isdir(cov_img_dir) and os.path.isdir(cov_mask_dir):
        for ipath in _list_images_in_dir(cov_img_dir):
            stem = os.path.splitext(os.path.basename(ipath))[0]
            if stem.lower().endswith('t'):
                base_id = stem[:-1].lower()
                mpath = mask_lookup.get(base_id + 'forged') or mask_lookup.get(stem.lower())
                if mpath:
                    samples.append((ipath, mpath, 1))
            else:
                samples.append((ipath, None, 0))
        if samples:
            return samples

    # Case 3: Standard images/ + masks/ layout
    img_dir = os.path.join(root_dir, 'images')
    mask_dir = os.path.join(root_dir, 'masks')
    if os.path.isdir(img_dir):
        for ipath in _list_images_in_dir(img_dir):
            stem = os.path.splitext(os.path.basename(ipath))[0].lower()
            mpath = mask_lookup.get(stem, None)
            if mpath and os.path.exists(mpath):
                # Check if stem indicates authentic or tampered
                is_tamp = 0 if 'auth' in stem or 'au_' in stem else 1
                samples.append((ipath, mpath, is_tamp))
            else:
                samples.append((ipath, None, 0))

    return samples


def apply_test_distortion(image, mask, distortion=None):
    """
    Applies deterministic post-processing distortion to a PIL image (and geometric transform to mask)
    for real robustness benchmarking:
      - ('jpeg', qf): JPEG compression with quality factor qf in [40, 100]
      - ('blur', sigma): Gaussian blur with radius/sigma in [0.5, 3.0]
      - ('noise', sigma_255): Additive white Gaussian noise with 8-bit std sigma_255 in [5, 25]
      - ('scale', factor): Bilinear downscaling/upscaling by factor then restoring size
      - ('rotate', deg): Bicubic rotation by deg degrees
      - ('social', platform): End-to-end social media re-encoding pipeline
        ('telegram' | 'wechat' | 'facebook' | 'whatsapp')
    """
    if distortion is None or distortion[0] in ('clean', None):
        return image, mask

    dtype, param = distortion[0].lower(), distortion[1]
    w, h = image.size

    if dtype == 'jpeg':
        qf = int(param)
        buf = io.BytesIO()
        image.save(buf, format='JPEG', quality=qf)
        buf.seek(0)
        image = Image.open(buf).convert('RGB')

    elif dtype == 'blur':
        sigma = float(param)
        if sigma > 0:
            image = image.filter(ImageFilter.GaussianBlur(radius=sigma))

    elif dtype == 'noise':
        sigma_255 = float(param)
        if sigma_255 > 0:
            rng = np.random.RandomState(101)
            arr = np.array(image, dtype=np.float32)
            noisy = arr + rng.normal(0.0, sigma_255, arr.shape).astype(np.float32)
            image = Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8))

    elif dtype == 'scale':
        factor = float(param)
        nw, nh = max(32, int(round(w * factor))), max(32, int(round(h * factor)))
        image = image.resize((nw, nh), Image.BILINEAR).resize((w, h), Image.BILINEAR)

    elif dtype == 'rotate':
        deg = float(param)
        image = image.rotate(deg, resample=Image.BICUBIC, expand=False)
        mask = mask.rotate(deg, resample=Image.NEAREST, expand=False)

    elif dtype == 'social':
        platform = str(param).lower()
        specs = {
            'telegram': (1280, 82),
            'wechat':   (1080, 75),
            'facebook': (2048, 71),
            'whatsapp': (1600, 68),
        }
        max_dim, qf = specs.get(platform, (1280, 75))
        scale = min(1.0, max_dim / float(max(w, h)))
        if scale < 1.0:
            sw, sh = max(32, int(round(w * scale))), max(32, int(round(h * scale)))
            image_small = image.resize((sw, sh), Image.BILINEAR)
        else:
            # Even when image is smaller than max_dim, social media apps re-sample/re-encode
            sw, sh = max(32, int(round(w * 0.85))), max(32, int(round(h * 0.85)))
            image_small = image.resize((sw, sh), Image.BILINEAR)
        buf = io.BytesIO()
        image_small.save(buf, format='JPEG', quality=qf, subsampling=2)  # 4:2:0 chroma subsampling
        buf.seek(0)
        image = Image.open(buf).convert('RGB').resize((w, h), Image.BILINEAR)

    return image, mask


class PassiveForgeryDataset(Dataset):
    """
    Real multi-dataset loader for passive image forgery localization.
    Supports CASIA v1.0, CASIA v2.0, Columbia, NIST'16, COVERAGE, IMD2020, CocoGlide, AutoSplice, LaMa.

    Features:
      - Automatic directory structure discovery (Au/Tp/Gt, 4cam_auth/4cam_splc, image/mask, images/masks).
      - Deterministic stratified train/val/test split (default 80% / 10% / 10% with seed=42).
      - Native-resolution random cropping during training (preserving PRNU, CFA, and 8x8 DCT grids).
      - Test-time distortion hooks for JPEG, Blur, AWGN, Geometric, and Social Media robustness benchmarks.
    """
    def __init__(
        self,
        root_dir=None,
        img_size=(256, 256),
        is_train=True,
        split='all',
        split_ratios=(0.8, 0.1, 0.1),
        split_seed=42,
        crop_mode='native',
        simulate_jpeg=True,
        simulate_blur=False,
        eval_distortion=None,
        samples_override=None
    ):
        self.root_dir = root_dir
        self.img_size = img_size if isinstance(img_size, tuple) else (int(img_size), int(img_size))
        self.is_train = is_train
        self.split = split.lower()
        self.crop_mode = crop_mode
        self.simulate_jpeg = simulate_jpeg
        self.simulate_blur = simulate_blur
        self.eval_distortion = eval_distortion

        if samples_override is not None:
            all_samples = list(samples_override)
        else:
            all_samples = discover_forensic_samples(root_dir)

        # Stratified train / val / test split
        if self.split in ('train', 'val', 'test') and len(all_samples) > 0:
            auth = [s for s in all_samples if s[2] == 0]
            tamp = [s for s in all_samples if s[2] == 1]
            rng = random.Random(split_seed)
            rng.shuffle(auth)
            rng.shuffle(tamp)

            def _slice_split(items):
                n = len(items)
                n_train = int(round(n * split_ratios[0]))
                n_val = int(round(n * split_ratios[1]))
                if self.split == 'train':
                    return items[:n_train]
                elif self.split == 'val':
                    return items[n_train:n_train + n_val]
                else:
                    return items[n_train + n_val:]

            self.samples = _slice_split(auth) + _slice_split(tamp)
            rng.shuffle(self.samples)
        else:
            self.samples = all_samples

    def __len__(self):
        return len(self.samples)

    def apply_train_distortions(self, image):
        if self.simulate_jpeg and random.random() < 0.5:
            qf = random.randint(60, 100)
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=qf)
            buffer.seek(0)
            image = Image.open(buffer).convert('RGB')

        if self.simulate_blur and random.random() < 0.25:
            radius = random.uniform(0.5, 1.5)
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))

        return image

    def _native_crop_or_pad(self, img_t, mask_t):
        """
        Extracts a native-resolution (crop_h, crop_w) patch without bilinear interpolation.
        Applies reflection padding if the raw image is smaller than the target crop size.
        """
        th, tw = self.img_size
        _, h, w = img_t.shape

        pad_h = max(0, th - h)
        pad_w = max(0, tw - w)
        if pad_h > 0 or pad_w > 0:
            # Safe reflection pad (fallback to replicate if image is smaller than pad size)
            mode = 'reflect' if (pad_h < h and pad_w < w) else 'replicate'
            img_t = F.pad(img_t.unsqueeze(0), (0, pad_w, 0, pad_h), mode=mode).squeeze(0)
            mask_t = F.pad(mask_t.unsqueeze(0), (0, pad_w, 0, pad_h), mode='constant', value=0.0).squeeze(0)
            _, h, w = img_t.shape

        if self.is_train:
            top = random.randint(0, h - th)
            left = random.randint(0, w - tw)
        else:
            top = (h - th) // 2
            left = (w - tw) // 2

        img_t = img_t[:, top:top + th, left:left + tw]
        mask_t = mask_t[:, top:top + th, left:left + tw]
        return img_t, mask_t

    def __getitem__(self, idx):
        entry = self.samples[idx]
        img_path, mask_path = entry[0], entry[1]

        image = Image.open(img_path).convert('RGB')
        w, h = image.size

        if mask_path is not None and os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')
            if mask.size != (w, h):
                mask = mask.resize((w, h), Image.NEAREST)
        else:
            mask = Image.new('L', (w, h), 0)

        # Apply evaluation distortion if benchmarking robustness
        if not self.is_train and self.eval_distortion is not None:
            image, mask = apply_test_distortion(image, mask, self.eval_distortion)

        # Apply training photometric augmentations prior to tensor conversion
        if self.is_train:
            image = self.apply_train_distortions(image)

        if self.crop_mode == 'resize':
            image = image.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
            mask = mask.resize((self.img_size[1], self.img_size[0]), Image.NEAREST)

        img_tensor = TF.to_tensor(image)  # [3, H, W] in [0, 1]
        mask_np = np.array(mask, dtype=np.float32)
        mask_binary = (mask_np > 127).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0)  # [1, H, W]

        if self.crop_mode == 'native':
            img_tensor, mask_tensor = self._native_crop_or_pad(img_tensor, mask_tensor)

        if self.is_train:
            # Random horizontal and vertical flips
            if random.random() > 0.5:
                img_tensor = torch.flip(img_tensor, dims=[2])
                mask_tensor = torch.flip(mask_tensor, dims=[2])
            if random.random() > 0.5:
                img_tensor = torch.flip(img_tensor, dims=[1])
                mask_tensor = torch.flip(mask_tensor, dims=[1])
            # Random 90-degree rotations (preserving 8x8 JPEG block grid)
            k_rot = random.randint(0, 3)
            if k_rot > 0:
                img_tensor = torch.rot90(img_tensor, k=k_rot, dims=[1, 2])
                mask_tensor = torch.rot90(mask_tensor, k=k_rot, dims=[1, 2])

        return img_tensor, mask_tensor, os.path.basename(img_path)
