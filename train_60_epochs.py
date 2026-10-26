import os
import argparse
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from models import DF_CMNet
from losses import CompoundForgeryLoss
from dataset import PassiveForgeryDataset, SyntheticForgeryDataset

def calculate_metrics_fast(preds, targets, threshold=0.5):
    preds_bin = (preds > threshold).astype(np.uint8)
    targets_bin = targets.astype(np.uint8)

    tp = np.sum((preds_bin == 1) & (targets_bin == 1))
    fp = np.sum((preds_bin == 1) & (targets_bin == 0))
    fn = np.sum((preds_bin == 0) & (targets_bin == 1))

    prec = tp / (tp + fp + 1e-7)
    rec = tp / (tp + fn + 1e-7)
    f1 = 2 * prec * rec / (prec + rec + 1e-7)
    iou = tp / (tp + fp + fn + 1e-7)

    return f1, prec, rec, iou

def evaluate_fast(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, masks, _ in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss, _ = criterion(outputs, masks)
            val_loss += loss.item() * images.size(0)

            preds = outputs['mask_pred'].squeeze(1).cpu().numpy()
            targets = masks.squeeze(1).cpu().numpy()

            all_preds.append(preds)
            all_targets.append(targets)

    val_loss /= len(val_loader.dataset)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    f1, prec, rec, iou = calculate_metrics_fast(all_preds, all_targets, threshold=0.5)

    flat_targets = all_targets.flatten()
    flat_preds = all_preds.flatten()
    if len(flat_targets) > 200000:
        sub_idx = np.random.choice(len(flat_targets), size=200000, replace=False)
        flat_targets = flat_targets[sub_idx]
        flat_preds = flat_preds[sub_idx]

    try:
        auc = roc_auc_score(flat_targets, flat_preds)
    except Exception:
        auc = 0.5

    return val_loss, f1, iou, auc

def run_60_epoch_training():
    parser = argparse.ArgumentParser(description="60-Epoch Full Training Experiment for DF-CMNet v2")
    parser.add_argument("--epochs", type=int, default=60, help="Total training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--img_size", type=int, default=128, help="Resolution for fast CPU/GPU profiling")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to real dataset (optional)")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Checkpoint directory")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Starting 60-epoch training on device: {device}")

    img_size = (args.img_size, args.img_size)
    if args.data_dir and os.path.exists(args.data_dir):
        print(f"[*] Loading dataset from {args.data_dir}")
        train_ds = PassiveForgeryDataset(args.data_dir, img_size=img_size, is_train=True)
        val_ds = PassiveForgeryDataset(args.data_dir, img_size=img_size, is_train=False)
    else:
        print("[*] Generating synthetic multi-scale benchmark dataset...")
        train_ds = SyntheticForgeryDataset(num_samples=160, img_size=img_size, is_train=True)
        val_ds = SyntheticForgeryDataset(num_samples=40, img_size=img_size, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = DF_CMNet().to(device)
    criterion = CompoundForgeryLoss(lambda_foc=1.0, lambda_iou=1.0, lambda_bound=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'f1': [],
        'iou': [],
        'auc': [],
        'lr': []
    }

    best_f1 = 0.0
    start_total = time.time()

    print("\n" + "="*80)
    print(f"{'Epoch':^8}|{'Train Loss':^12}|{'Val Loss':^12}|{'F1 (%)':^10}|{'IoU (%)':^10}|{'AUC (%)':^10}|{'Time':^8}")
    print("="*80)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for imgs, masks, _ in train_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss, _ = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        train_loss /= len(train_loader)
        ep_duration = time.time() - t0

        # Evaluate
        val_loss, f1, iou, auc = evaluate_fast(model, val_loader, criterion, device)

        history['epoch'].append(epoch)
        history['train_loss'].append(round(train_loss, 4))
        history['val_loss'].append(round(val_loss, 4))
        history['f1'].append(round(f1 * 100, 2))
        history['iou'].append(round(iou * 100, 2))
        history['auc'].append(round(auc * 100, 2))
        history['lr'].append(current_lr)

        print(f"{epoch:^8}|{train_loss:^12.4f}|{val_loss:^12.4f}|{f1*100:^10.2f}|{iou*100:^10.2f}|{auc*100:^10.2f}|{ep_duration:^7.1f}s")

        if f1 > best_f1:
            best_f1 = f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'auc': auc
            }, os.path.join(args.save_dir, "best_model_60epochs.pth"))

    total_time_min = (time.time() - start_total) / 60.0
    print("="*80)
    print(f"[+] 60-Epoch Training Complete in {total_time_min:.2f} minutes! Best F1: {best_f1*100:.2f}%")

    # Save convergence history
    hist_path = os.path.join("results", "convergence_60_epochs.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=4)
    print(f"[+] Saved 60-epoch convergence log to: {hist_path}")

if __name__ == "__main__":
    run_60_epoch_training()
