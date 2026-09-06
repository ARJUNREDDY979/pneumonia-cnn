"""ResNet-18 backbone with a swapped classification head for binary X-ray classification."""
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def build_model(num_classes: int = 2, freeze_backbone: bool = False) -> nn.Module:
    """ImageNet-pretrained ResNet-18 with its final FC layer replaced.

    freeze_backbone=True only trains the new head (fast, good baseline / low-data
    regime). freeze_backbone=False fine-tunes every layer (used for the full run
    once the head has converged) -- with ~5k training images this still trains in
    minutes on a single GPU and consistently outperforms head-only training on this
    dataset, since low-level ImageNet filters aren't perfectly matched to X-ray
    texture statistics.
    """
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    # the new head is always trainable, even when the backbone is frozen
    for param in model.fc.parameters():
        param.requires_grad = True

    return model
