import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from models import DF_CMNet
from losses import CompoundForgeryLoss
from dataset import PassiveForgeryDataset, SyntheticForgeryDataset

def find_optimal_threshold(preds, targets, thresholds=np.arange(0.1, 0.95, 0.05)):
    """
    Evaluates F1 score across multiple candidate thresholds to identify the optimal operating point.
    """
    best_f1 = 0.0
    best_tau = 0.5
    best_prec = 0.0
    best_rec = 0.0
    best_iou = 0.0

    targets_bin = targets.astype(np.uint8)

    for tau in thresholds:
        preds_bin = (preds > tau).astype(np.uint8)
        tp = np.sum((preds_bin == 1) & (targets_bin == 1))
        fp = np.sum((preds_bin == 1) & (targets_bin == 0))
        fn = np.sum((preds_bin == 0) & (targets_bin == 1))

        prec = tp / (tp + fp + 1e-7)
        rec = tp / (tp + fn + 1e-7)
        f1 = 2 * prec * rec / (prec + rec + 1e-7)
        iou = tp / (tp + fp + fn + 1e-7)

        if f1 > best_f1:
            best_f1 = f1
            best_tau = tau
            best_prec = prec
            best_rec = rec
            best_iou = iou

    return best_f1, best_tau, best_prec, best_rec, best_iou

def calculate_metrics_at_threshold(preds, targets, tau=0.5):
    preds_bin = (preds > tau).astype(np.uint8)
    targets_bin = targets.astype(np.uint8)

    tp = np.sum((preds_bin == 1) & (targets_bin == 1))
    fp = np.sum((preds_bin == 1) & (targets_bin == 0))
    fn = np.sum((preds_bin == 0) & (targets_bin == 1))

    prec = tp / (tp + fp + 1e-7)
    rec = tp / (tp + fn + 1e-7)
    f1 = 2 * prec * rec / (prec + rec + 1e-7)
    iou = tp / (tp + fp + fn + 1e-7)
    return f1, prec, rec, iou

def evaluate_model(model, val_loader, criterion, device):
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

    # Standard metrics at fixed tau = 0.5
    f1_05, prec_05, rec_05, iou_05 = calculate_metrics_at_threshold(all_preds, all_targets, tau=0.5)

    # Optimal threshold metrics
    opt_f1, opt_tau, opt_prec, opt_rec, opt_iou = find_optimal_threshold(all_preds, all_targets)

    # AUC calculation
    flat_targets = all_targets.flatten()
    flat_preds = all_preds.flatten()
    if len(flat_targets) > 500000:
        subsample_idx = np.random.choice(len(flat_targets), size=500000, replace=False)
        flat_targets = flat_targets[subsample_idx]
        flat_preds = flat_preds[subsample_idx]

    try:
        auc = roc_auc_score(flat_targets, flat_preds)
    except Exception:
        auc = 0.5

    return {
        'val_loss': val_loss,
        'f1_05': f1_05,
        'iou_05': iou_05,
        'opt_f1': opt_f1,
        'opt_tau': opt_tau,
        'opt_iou': opt_iou,
        'auc': auc
    }

def main():
    parser = argparse.ArgumentParser(description="Train DF-CMNet v2 for Passive Image Forgery Detection")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to real dataset folder")
    parser.add_argument("--use_synthetic", action="store_true", default=True, help="Use synthetic dataset generator")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Initial learning rate")
    parser.add_argument("--img_size", type=int, default=128, help="Image dimension")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device}")

    img_size = (args.img_size, args.img_size)
    if args.data_dir and os.path.exists(args.data_dir):
        print(f"[*] Loading dataset from: {args.data_dir}")
        train_dataset = PassiveForgeryDataset(args.data_dir, img_size=img_size, is_train=True)
        val_dataset = PassiveForgeryDataset(args.data_dir, img_size=img_size, is_train=False)
    else:
        print("[*] Using SyntheticForgeryDataset with diverse tampering profiles...")
        train_dataset = SyntheticForgeryDataset(num_samples=120, img_size=img_size, is_train=True)
        val_dataset = SyntheticForgeryDataset(num_samples=30, img_size=img_size, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print("[*] Instantiating DF-CMNet v2 (with Dual-Attention CDBM & Deep Supervision)...")
    model = DF_CMNet().to(device)
    criterion = CompoundForgeryLoss(lambda_foc=1.0, lambda_iou=1.0, lambda_bound=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_opt_f1 = 0.0

    print(f"[*] Starting training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        start_time = time.time()

        for images, masks, _ in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss, _ = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)
        epoch_time = time.time() - start_time

        # Validation
        res = evaluate_model(model, val_loader, criterion, device)

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_time:.1f}s) - "
              f"Train Loss: {train_loss:.4f} | Val Loss: {res['val_loss']:.4f} | "
              f"F1@0.5: {res['f1_05']*100:.2f}% | Opt-F1 (t={res['opt_tau']:.2f}): {res['opt_f1']*100:.2f}% | "
              f"AUC: {res['auc']*100:.2f}% | Opt-IoU: {res['opt_iou']*100:.2f}%")

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_f1': best_opt_f1,
            'f1_05': res['f1_05'],
            'opt_f1': res['opt_f1'],
            'opt_tau': res['opt_tau'],
            'auc': res['auc']
        }
        torch.save(checkpoint, os.path.join(args.save_dir, "latest_model.pth"))

        if res['opt_f1'] > best_opt_f1:
            best_opt_f1 = res['opt_f1']
            torch.save(checkpoint, os.path.join(args.save_dir, "best_model.pth"))
            print(f"  [+] Saved new best model checkpoint! Best Opt-F1: {best_opt_f1*100:.2f}%")

    print("[*] Enhanced training pipeline finished successfully.")

if __name__ == "__main__":
    main()
