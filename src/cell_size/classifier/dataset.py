"""PyTorch Dataset wrappers and data augmentation transforms."""

from __future__ import annotations

from pathlib import Path

from torchvision import transforms
from torchvision.datasets import ImageFolder

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(crop_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        # transforms.RandomHorizontalFlip(), # TODO: not sure if for cell data makes sense
        # transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(90),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        # transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms(crop_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_dataset(
    root: str | Path,
    split: str = "train",
    crop_size: int = 224,
) -> ImageFolder:
    """Build an ImageFolder dataset for a given split.

    Expects the directory layout::

        root/train/good/*.png
        root/train/bad/*.png
        root/val/good/*.png
        ...

    The class-to-index mapping will be ``{"bad": 0, "good": 1}``.
    """
    split_dir = Path(root) / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    tfm = get_train_transforms(crop_size) if split == "train" else get_eval_transforms(crop_size)
    ds = ImageFolder(str(split_dir), transform=tfm)
    return ds
