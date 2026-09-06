"""Train the ResNet-18 pneumonia classifier.

Usage:
    python src/train.py --data-dir data/chest_xray --epochs 15 --batch-size 32
"""
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import recall_score, f1_score
from tqdm import tqdm

from dataset import get_dataloaders, class_counts
from model import build_model
from utils import set_seed, get_device, compute_class_weights, save_json


def parse_args():
    p = argparse.ArgumentParser(description="Train pneumonia CNN classifier")
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--freeze-backbone", action="store_true",
                    help="train only the classification head")
    p.add_argument("--patience", type=int, default=4,
                    help="early-stopping patience, in epochs without val F1 improvement")
    p.add_argument("--out-dir", type=str, default="outputs")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, desc="train" if train else "eval", leave=False):
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    # recall on class index 1 (PNEUMONIA) is the metric we actually care about:
    # missing a true pneumonia case is far costlier than an extra false alarm.
    pneumonia_recall = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
    f1 = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)
    return avg_loss, pneumonia_recall, f1


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, _, classes = get_dataloaders(
        args.data_dir, batch_size=args.batch_size
    )
    print(f"Classes: {classes}")

    counts = class_counts(train_loader.dataset)
    weights = compute_class_weights(counts).to(device)
    print(f"Class counts: {counts} -> loss weights: {weights.tolist()}")

    model = build_model(num_classes=len(classes), freeze_backbone=args.freeze_backbone).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = out_dir / "best_model.pt"

    best_f1, epochs_no_improve = 0.0, 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_recall, train_f1 = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_recall, val_f1 = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )
        scheduler.step(val_f1)
        elapsed = time.time() - t0

        print(
            f"[{epoch:02d}/{args.epochs}] "
            f"train_loss={train_loss:.4f} train_recall(pneu)={train_recall:.3f} | "
            f"val_loss={val_loss:.4f} val_recall(pneu)={val_recall:.3f} val_f1={val_f1:.3f} "
            f"({elapsed:.1f}s)"
        )
        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_recall": train_recall,
            "val_loss": val_loss, "val_recall": val_recall, "val_f1": val_f1,
        })

        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": classes,
                "epoch": epoch,
                "val_f1": val_f1,
            }, best_ckpt)
            print(f"  -> new best model saved (val_f1={val_f1:.3f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping: no val F1 improvement in {args.patience} epochs.")
                break

    save_json({"history": history, "best_val_f1": best_f1}, out_dir / "train_history.json")
    print(f"Training complete. Best checkpoint: {best_ckpt} (val_f1={best_f1:.3f})")


if __name__ == "__main__":
    main()
