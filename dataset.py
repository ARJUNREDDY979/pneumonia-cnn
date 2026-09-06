"""Dataset loading for the chest X-ray ImageFolder layout.

Expected directory structure (standard torchvision ImageFolder):
    data_dir/
        train/{NORMAL,PNEUMONIA}/*.jpeg
        val/{NORMAL,PNEUMONIA}/*.jpeg
        test/{NORMAL,PNEUMONIA}/*.jpeg
"""
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms():
    """Separate train/eval transforms.

    Augmentation is deliberately conservative and anatomy-aware: chest X-rays have
    a fixed up/down and left/right anatomical orientation, so we allow small
    rotations and brightness/contrast jitter (simulating positioning and exposure
    variance) but explicitly avoid vertical flips and aggressive cropping, which
    would produce images no radiologist would ever see.
    """
    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # X-rays are single-channel;
                                                       # replicate to 3ch for ResNet
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomRotation(degrees=7),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_tf, eval_tf


def get_dataloaders(data_dir: str, batch_size: int = 32, num_workers: int = 2):
    data_dir = Path(data_dir)
    train_tf, eval_tf = build_transforms()

    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=eval_tf)
    test_ds = datasets.ImageFolder(data_dir / "test", transform=eval_tf)

    # ImageFolder sorts class names alphabetically -> {'NORMAL': 0, 'PNEUMONIA': 1}
    assert train_ds.classes == val_ds.classes == test_ds.classes, (
        "train/val/test must expose the same classes in the same order"
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader, train_ds.classes


def class_counts(dataset: datasets.ImageFolder) -> dict:
    """Count samples per class index, for computing loss weights."""
    counts = {i: 0 for i in range(len(dataset.classes))}
    for _, label in dataset.samples:
        counts[label] += 1
    return counts
