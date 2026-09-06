"""Grad-CAM visualization: which regions of the X-ray drove each prediction.

Usage:
    python src/gradcam.py --data-dir data/chest_xray --checkpoint outputs/best_model.pt --n 6
"""
import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets

from dataset import build_transforms
from model import build_model
from utils import get_device, set_seed

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class GradCAM:
    """Minimal Grad-CAM hooked onto a ResNet's last conv block (layer4)."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer = model.layer4[-1]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[0, class_idx]
        score.backward()

        # global-average-pool the gradients -> per-channel importance weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, output.detach()


def unnormalize(tensor):
    img = tensor.cpu() * IMAGENET_STD + IMAGENET_MEAN
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def main():
    p = argparse.ArgumentParser(description="Grad-CAM on sample test images")
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--n", type=int, default=6, help="number of sample images")
    p.add_argument("--out-dir", type=str, default="outputs")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device()

    ckpt = torch.load(args.checkpoint, map_location=device)
    classes = ckpt["classes"]
    model = build_model(num_classes=len(classes)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _, eval_tf = build_transforms()
    test_ds = datasets.ImageFolder(Path(args.data_dir) / "test", transform=eval_tf)

    indices = random.sample(range(len(test_ds)), min(args.n, len(test_ds)))
    cam = GradCAM(model)

    fig, axes = plt.subplots(2, len(indices), figsize=(3 * len(indices), 6))
    if len(indices) == 1:
        axes = axes.reshape(2, 1)

    for col, idx in enumerate(indices):
        image, label = test_ds[idx]
        input_tensor = image.unsqueeze(0).to(device)

        with torch.no_grad():
            probs = F.softmax(model(input_tensor), dim=1)
        pred_class = probs.argmax(dim=1).item()

        heatmap, _ = cam(input_tensor, pred_class)
        img_np = unnormalize(image)

        axes[0, col].imshow(img_np)
        axes[0, col].set_title(f"true={classes[label]}", fontsize=9)
        axes[0, col].axis("off")

        axes[1, col].imshow(img_np)
        axes[1, col].imshow(heatmap, cmap="jet", alpha=0.45)
        conf = probs[0, pred_class].item()
        axes[1, col].set_title(f"pred={classes[pred_class]} ({conf:.2f})", fontsize=9)
        axes[1, col].axis("off")

    fig.suptitle("Grad-CAM: original (top) vs. model attention (bottom)")
    fig.tight_layout()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gradcam_examples.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
