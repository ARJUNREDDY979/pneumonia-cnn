"""Shared helpers: reproducibility, class weighting, and plotting."""
import random
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc


def set_seed(seed: int = 42) -> None:
    """Seed every source of randomness we use, for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_class_weights(class_counts: dict) -> torch.Tensor:
    """Inverse-frequency class weights, normalized to sum to num_classes.

    class_counts: {class_index: num_samples}, e.g. {0: 1341, 1: 3875}
    A class seen less often gets a proportionally larger weight in the loss,
    so misclassifying it costs more than the raw sample count would suggest.
    """
    total = sum(class_counts.values())
    n_classes = len(class_counts)
    weights = torch.zeros(n_classes)
    for idx, count in class_counts.items():
        weights[idx] = total / (n_classes * count)
    return weights


def save_confusion_matrix(y_true, y_pred, class_names, out_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=13,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_roc_curve(y_true, y_score, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, linewidth=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — PNEUMONIA vs NORMAL")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return roc_auc


def save_json(data: dict, out_path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
