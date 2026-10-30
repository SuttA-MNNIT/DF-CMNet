import os
import json
import random
import argparse
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from models import DF_CMNet
from losses import CompoundForgeryLoss
from dataset import PassiveForgeryDataset, SyntheticForgeryDataset


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_optimal_threshold(preds, targets, thresholds=np.arange(0.1, 0.95, 0.05)):
    """
    Evaluates F1 score across candidate thresholds to identify the optimal operating point
    and record the full threshold stability curve.
    """
    best_f1 = -1.0
    best_tau = 0.5
    best_prec = 0.0
    best_rec = 0.0
    best_iou = 0.0
    curve = {}

    for tau in thresholds:
        f1, prec, rec, iou, _ = calculate_metrics_at_threshold(preds, targets, tau=float(tau))
        curve[round(float(tau), 2)] = round(float(f1 * 100.0), 2)
        if f1 > best_f1:
            best_f1 = f1
            best_tau = float(tau)
            best_prec = prec
            best_rec = rec
            best_iou = iou

    return best_f1, best_tau, best_prec, best_rec, best_iou, curve


def calculate_metrics_at_threshold(preds, targets, tau=0.5):
    """
    Computes image-wise Forgery F1, Precision, Recall, IoU on tampered images (||Y||_1 > 0)
    and False-Positive Rate (Auth_FPR) on pristine authentic images (||Y||_1 == 0).
    Falls back to corpus-level macro aggregation if all images share one class.
    """
    preds_bin = (preds >= tau).astype(np.uint8)
    targets_bin = (targets > 0.5).astype(np.uint8)

    f1_list, prec_list, rec_list, iou_list = [], [], [], []
    auth_fpr_list = []

    for i in range(len(targets_bin)):
        p_i = preds_bin[i]
        t_i = targets_bin[i]
        t_sum = int(np.sum(t_i))
        if t_sum == 0:
            auth_fpr_list.append(float(np.mean(p_i)))
        else:
            tp = float(np.sum((p_i == 1) & (t_i == 1)))
            fp = float(np.sum((p_i == 1) & (t_i == 0)))
            fn = float(np.sum((p_i == 0) & (t_i == 1)))
            prec = tp / (tp + fp + 1e-7)
            rec = tp / (tp + fn + 1e-7)
            f1 = 2.0 * prec * rec / (prec + rec + 1e-7)
            iou = tp / (tp + fp + fn + 1e-7)
            f1_list.append(f1)
            prec_list.append(prec)
            rec_list.append(rec)
            iou_list.append(iou)

    if len(f1_list) > 0:
        f1 = float(np.mean(f1_list))
        prec = float(np.mean(prec_list))
        rec = float(np.mean(rec_list))
        iou = float(np.mean(iou_list))
    else:
        tp = float(np.sum((preds_bin == 1) & (targets_bin == 1)))
        fp = float(np.sum((preds_bin == 1) & (targets_bin == 0)))
        fn = float(np.sum((preds_bin == 0) & (targets_bin == 1)))
        prec = tp / (tp + fp + 1e-7)
        rec = tp / (tp + fn + 1e-7)
        f1 = 2.0 * prec * rec / (prec + rec + 1e-7)
        iou = tp / (tp + fp + fn + 1e-7)

    auth_fpr = float(np.mean(auth_fpr_list)) if len(auth_fpr_list) > 0 else 0.0
    return f1, prec, rec, iou, auth_fpr


def evaluate_model(model, val_loader, criterion, device, compute_curve=False):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, masks, _ in val_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            outputs = model(images)
            loss, _ = criterion(outputs, masks)
            val_loss += loss.item() * images.size(0)

            preds = outputs['mask_pred'].squeeze(1).float().cpu().numpy()
            targets = masks.squeeze(1).float().cpu().numpy()

            all_preds.append(preds)
            all_targets.append(targets)

    val_loss /= max(len(val_loader.dataset), 1)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    f1_05, prec_05, rec_05, iou_05, auth_fpr = calculate_metrics_at_threshold(all_preds, all_targets, tau=0.5)

    if compute_curve:
        opt_f1, opt_tau, opt_prec, opt_rec, opt_iou, tau_curve = find_optimal_threshold(all_preds, all_targets)
    else:
        opt_f1, opt_tau, opt_prec, opt_rec, opt_iou, tau_curve = f1_05, 0.5, prec_05, rec_05, iou_05, {}

    flat_targets = (all_targets.flatten() > 0.5).astype(np.uint8)
    flat_preds = all_preds.flatten()
    if len(flat_targets) > 500000:
        rng = np.random.RandomState(42)
        subsample_idx = rng.choice(len(flat_targets), size=500000, replace=False)
        flat_targets = flat_targets[subsample_idx]
        flat_preds = flat_preds[subsample_idx]

    try:
        auc = float(roc_auc_score(flat_targets, flat_preds))
    except Exception:
        auc = 0.5

    return {
        'val_loss': val_loss,
        'f1_05': f1_05,
        'prec_05': prec_05,
        'rec_05': rec_05,
        'iou_05': iou_05,
        'auth_fpr': auth_fpr,
        'opt_f1': opt_f1,
        'opt_tau': opt_tau,
        'opt_iou': opt_iou,
        'auc': auc,
        'tau_curve': tau_curve
    }


def main():
    parser = argparse.ArgumentParser(description="Train DF-CMNet v2 on Real Forensic Datasets (Colab H100 Ready)")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to real dataset folder (e.g. ./data/CASIA_v2)")
    parser.add_argument("--variant", type=str, default="V8", choices=["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8"],
                        help="Architecture/loss ablation variant (default V8 = Full DF-CMNet v2)")
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (16 or 32 recommended on H100)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--img_size", type=int, default=256, help="Training crop dimension")
    parser.add_argument("--crop_mode", type=str, default="native", choices=["native", "resize"],
                        help="Native-resolution random cropping ('native') or bilinear resize ('resize')")
    parser.add_argument("--lambda_asp", type=float, default=0.5, help="Weight for Authentic Suppression Penalty")
    parser.add_argument("--lambda_bound", type=float, default=0.5, help="Weight for BARGM boundary loss")
    parser.add_argument("--lambda_aux", type=float, default=0.4, help="Weight for deep supervision loss")
    parser.add_argument("--boundary_kernel", type=int, default=5, help="Morphological structuring element size (2*r_d + 1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--amp", action="store_true", default=True, help="Use automatic mixed precision on CUDA")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save JSON logs")
    parser.add_argument("--ckpt_name", type=str, default="best_model.pth", help="Output checkpoint filename")
    parser.add_argument("--allow_synthetic", action="store_true", default=False,
                        help="Explicitly permit SyntheticForgeryDataset for local dry-run unit tests ONLY. Defaults to False.")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and (device.type == "cuda")
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    img_size = (args.img_size, args.img_size)
    is_real = False

    if args.data_dir and os.path.exists(args.data_dir):
        train_dataset = PassiveForgeryDataset(
            args.data_dir, img_size=img_size, is_train=True, split='train',
            split_seed=args.seed, crop_mode=args.crop_mode, simulate_jpeg=True
        )
        val_dataset = PassiveForgeryDataset(
            args.data_dir, img_size=img_size, is_train=False, split='val',
            split_seed=args.seed, crop_mode=args.crop_mode, simulate_jpeg=False
        )
        if len(train_dataset) < 50:
            raise RuntimeError(
                f"[FATAL INTEGRITY ERROR] Only {len(train_dataset)} training samples found at '{args.data_dir}'! "
                f"Real forensic datasets require hundreds/thousands of images. Execution aborted to prevent training on corrupt paths."
            )
        is_real = True
        print("=" * 80)
        print(f"[PROVENANCE: GENUINE REAL DATASET CONFIRMED]")
        print(f"  Dataset Root   : {os.path.abspath(args.data_dir)}")
        print(f"  Train Samples  : {len(train_dataset)} real images")
        print(f"  Val Samples    : {len(val_dataset)} real images")
        print(f"  Resolution/Crop: {args.crop_mode} ({args.img_size}x{args.img_size})")
        print("=" * 80)
    elif args.allow_synthetic:
        print("!" * 80)
        print("[!] CAUTION: --allow_synthetic explicitly passed. Using Synthetic dummy dataset for SMOKE TESTING ONLY.")
        print("!" * 80)
        train_dataset = SyntheticForgeryDataset(num_samples=160, img_size=img_size, is_train=True)
        val_dataset = SyntheticForgeryDataset(num_samples=40, img_size=img_size, is_train=False)
        is_real = False
    else:
        raise FileNotFoundError(
            f"\n[FATAL ERROR] Dataset path '{args.data_dir}' does not exist or was not specified!\n"
            f"Silent fallback to synthetic/dummy data has been completely DISABLED to guarantee zero experimental contamination.\n"
            f"Actions to resolve:\n"
            f"  1. On Google Colab / GPU: Run `python download_datasets.py --data_root ./data` first, then pass `--data_dir ./data/CASIA_v2`.\n"
            f"  2. For dry-run code testing only: Explicitly pass `--allow_synthetic`.\n"
        )

    print(f"[*] Training DF-CMNet v2 (Variant {args.variant}, Seed {args.seed}) on {device} (AMP={use_amp}, dtype={amp_dtype})")

    nw = args.num_workers if device.type == "cuda" else 0
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=nw, pin_memory=(device.type == "cuda"), drop_last=(len(train_dataset) > args.batch_size))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=nw, pin_memory=(device.type == "cuda"))

    # Configure loss mode for Ablation Variants V6 and V7
    overlap_mode = 'proposed_asp'
    lambda_asp = args.lambda_asp
    lambda_bound = args.lambda_bound
    if args.variant == 'V6':
        overlap_mode = 'naive_iou'
        lambda_asp = 0.0
    elif args.variant == 'V7':
        overlap_mode = 'dice'
        lambda_asp = 0.0
    elif args.variant in ('V1', 'V2', 'V3', 'V4'):
        lambda_bound = 0.0

    model = DF_CMNet(variant=args.variant).to(device)
    criterion = CompoundForgeryLoss(
        lambda_foc=1.0,
        lambda_iou=1.0,
        lambda_bound=lambda_bound,
        lambda_aux=args.lambda_aux,
        lambda_asp=lambda_asp,
        boundary_kernel_size=args.boundary_kernel,
        overlap_mode=overlap_mode
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    history = {
        'variant': args.variant,
        'seed': args.seed,
        'data_dir': args.data_dir,
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'f1': [],
        'iou': [],
        'auc': [],
        'auth_fpr': [],
        'lr': []
    }

    best_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        start_time = time.time()

        for images, masks, _ in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                outputs = model(images)
                loss, _ = criterion(outputs, masks)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_loss += loss.item()

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        train_loss /= max(len(train_loader), 1)
        epoch_time = time.time() - start_time

        compute_curve = (epoch == args.epochs)
        res = evaluate_model(model, val_loader, criterion, device, compute_curve=compute_curve)

        history['epoch'].append(epoch)
        history['train_loss'].append(round(train_loss, 4))
        history['val_loss'].append(round(res['val_loss'], 4))
        history['f1'].append(round(res['f1_05'] * 100.0, 2))
        history['iou'].append(round(res['iou_05'] * 100.0, 2))
        history['auc'].append(round(res['auc'] * 100.0, 2))
        history['auth_fpr'].append(round(res['auth_fpr'] * 100.0, 2))
        history['lr'].append(current_lr)
        if compute_curve:
            history['tau_curve'] = res['tau_curve']

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_time:.1f}s) | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {res['val_loss']:.4f} | "
              f"Val F1@0.5: {res['f1_05']*100:.2f}% | IoU: {res['iou_05']*100:.2f}% | "
              f"AUC: {res['auc']*100:.2f}% | Auth FPR: {res['auth_fpr']*100:.2f}%")

        checkpoint = {
            'model_name': 'DF-CMNet-v2',
            'is_real_training': is_real,
            'dataset_path': os.path.abspath(args.data_dir) if args.data_dir else 'SyntheticSmokeTest',
            'num_train_samples': len(train_dataset),
            'num_val_samples': len(val_dataset),
            'crop_mode': args.crop_mode,
            'epoch': epoch,
            'total_epochs': args.epochs,
            'variant': args.variant,
            'seed': args.seed,
            'model_state_dict': model.state_dict(),
            'best_f1': max(best_f1, res['f1_05']),
            'f1_05': res['f1_05'],
            'iou_05': res['iou_05'],
            'auc': res['auc'],
            'auth_fpr': res['auth_fpr'],
            'train_loss': train_loss,
            'val_loss': res['val_loss'],
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'device': str(device),
            'cuda_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        }

        if res['f1_05'] > best_f1:
            best_f1 = res['f1_05']
            ckpt_path = os.path.join(args.save_dir, args.ckpt_name)
            torch.save(checkpoint, ckpt_path)
            print(f"    --> [SAVED NEW BEST] Epoch {epoch:02d}: Val F1 improved to {best_f1*100:.2f}% (Saved to {ckpt_path})")

    log_name = f"convergence_{args.variant}_seed{args.seed}.json" if args.variant != "V8" else "convergence_60_epochs.json"
    log_path = os.path.join(args.results_dir, log_name)
    with open(log_path, "w") as f:
        json.dump(history, f, indent=4)
    print(f"[+] Training complete. Best Val F1@0.5: {best_f1*100:.2f}%. Saved log to {log_path}")


if __name__ == "__main__":
    main()
