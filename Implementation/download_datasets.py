"""
Automated Benchmark Dataset Downloader & Verifier for Google Colab (H100 / A100 / T4).

Downloads real passive image forensics benchmark datasets into `data/<DATASET_NAME>/`
using `kagglehub` (pre-installed on Google Colab, no API token needed for public datasets)
and official university mirrors (GRIP Unina for CocoGlide), then verifies sample counts
with `discover_forensic_samples`.

Usage on Google Colab:
    python download_datasets.py --data_root ./data --datasets CASIA_v2 Columbia COVERAGE CocoGlide
"""

import os
import sys
import shutil
import zipfile
import urllib.request
import argparse

from dataset.forgery_dataset import discover_forensic_samples


KAGGLE_DATASETS = {
    "CASIA_v2": [
        "divg07/casia-20-image-tampering-detection-dataset",
        "sophatvathana/casia-dataset",
    ],
    "CASIA_v1": [
        "sophatvathana/casia-dataset",
    ],
    "Columbia": [
        "ridamia/columbia-uncompressed-image-splicing-detection",
    ],
    "COVERAGE": [
        "saurabhbagchi/coverage-dataset",
    ],
    "IMD2020": [
        "zeyadkhalid/imd2020-dataset",
    ],
}

DIRECT_URLS = {
    "CocoGlide": "https://www.grip.unina.it/download/prog/TruFor/CocoGlide.zip",
}


def symlink_or_copy(src, dst):
    if os.path.exists(dst):
        return
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    try:
        os.symlink(os.path.abspath(src), dst, target_is_directory=True)
    except Exception:
        shutil.copytree(src, dst, dirs_exist_ok=True)


def download_via_kagglehub(dataset_name, slugs, target_dir):
    try:
        import kagglehub
    except ImportError:
        print("[!] Installing kagglehub...")
        os.system(f"{sys.executable} -m pip install -q kagglehub")
        import kagglehub

    for slug in slugs:
        try:
            print(f"[*] Downloading {dataset_name} from Kaggle ({slug})...")
            dl_path = kagglehub.dataset_download(slug)
            print(f"    Downloaded to cache: {dl_path}")
            samples = discover_forensic_samples(dl_path)
            if len(samples) > 0:
                symlink_or_copy(dl_path, target_dir)
                auth_n = sum(1 for s in samples if s[2] == 0)
                tamp_n = sum(1 for s in samples if s[2] == 1)
                print(f"[+] {dataset_name} ready at {target_dir}: {len(samples)} total ({auth_n} authentic, {tamp_n} tampered)")
                return True
        except Exception as e:
            print(f"[!] Kaggle slug {slug} failed: {e}")
    return False


def download_via_url(dataset_name, url, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    zip_path = os.path.join(target_dir, f"{dataset_name}.zip")
    try:
        print(f"[*] Downloading {dataset_name} from {url} ...")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(target_dir)
        os.remove(zip_path)
        samples = discover_forensic_samples(target_dir)
        print(f"[+] {dataset_name} extracted to {target_dir}: {len(samples)} samples discovered")
        return len(samples) > 0
    except Exception as e:
        print(f"[!] Direct download for {dataset_name} failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download real forensic benchmark datasets on Colab")
    parser.add_argument("--data_root", type=str, default="./data", help="Root folder to store datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["CASIA_v2", "Columbia", "COVERAGE"],
        choices=["CASIA_v2", "CASIA_v1", "Columbia", "COVERAGE", "IMD2020", "CocoGlide"],
        help="Which benchmark datasets to download"
    )
    args = parser.parse_args()

    os.makedirs(args.data_root, exist_ok=True)

    for ds_name in args.datasets:
        target_dir = os.path.join(args.data_root, ds_name)
        existing = discover_forensic_samples(target_dir)
        if len(existing) > 0:
            auth_n = sum(1 for s in existing if s[2] == 0)
            tamp_n = sum(1 for s in existing if s[2] == 1)
            print(f"[=] {ds_name} already present at {target_dir}: {len(existing)} samples ({auth_n} auth, {tamp_n} tamp)")
            continue

        if ds_name in KAGGLE_DATASETS:
            download_via_kagglehub(ds_name, KAGGLE_DATASETS[ds_name], target_dir)
        elif ds_name in DIRECT_URLS:
            download_via_url(ds_name, DIRECT_URLS[ds_name], target_dir)


if __name__ == "__main__":
    main()
