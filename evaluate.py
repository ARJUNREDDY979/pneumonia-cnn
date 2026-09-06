"""Evaluate a trained checkpoint on the held-out test split.

Usage:
    python src/evaluate.py --data-dir data/chest_xray --checkpoint outputs/best_model.pt
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report
)

from dataset import get_dataloaders
from model import build_model
from utils import get_device, save_confusion_matrix, save_roc_curve, save_json


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate pneumonia CNN classifier")
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out-dir", type=str, default="outputs")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()

    _, _, test_loader, classes = get_dataloaders(args.data_dir, batch_size=args.batch_size)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = build_model(num_classes=len(ckpt["classes"])).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_labels, all_preds, all_scores = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            all_labels.extend(labels.tolist())
            all_preds.extend(preds.cpu().tolist())
            all_scores.extend(probs[:, 1].cpu().tolist())  # P(PNEUMONIA)

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, pos_label=1)
    recall = recall_score(all_labels, all_preds, pos_label=1)
    f1 = f1_score(all_labels, all_preds, pos_label=1)

    print(f"Test accuracy:            {acc:.4f}")
    print(f"Test precision (PNEUMONIA): {precision:.4f}")
    print(f"Test recall (PNEUMONIA):    {recall:.4f}")
    print(f"Test F1 (PNEUMONIA):        {f1:.4f}")
    print()
    print(classification_report(all_labels, all_preds, target_names=classes))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_confusion_matrix(all_labels, all_preds, classes, out_dir / "confusion_matrix.png")
    roc_auc = save_roc_curve(all_labels, all_scores, out_dir / "roc_curve.png")

    save_json({
        "test_accuracy": acc,
        "test_precision_pneumonia": precision,
        "test_recall_pneumonia": recall,
        "test_f1_pneumonia": f1,
        "test_roc_auc": roc_auc,
        "checkpoint_epoch": ckpt.get("epoch"),
    }, out_dir / "metrics.json")

    print(f"\nSaved confusion_matrix.png, roc_curve.png, metrics.json -> {out_dir}/")


if __name__ == "__main__":
    main()
