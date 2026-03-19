"""Model factory for binary cell-quality classifiers using torchvision."""

from __future__ import annotations

import logging

import torch.nn as nn
from torchvision import models

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, tuple[type, str]] = {
    "resnet18": (models.ResNet18_Weights, "fc"),
    "resnet50": (models.ResNet50_Weights, "fc"),
    "vit_b_16": (models.ViT_B_16_Weights, "heads.head"),
    "efficientnet_b0": (models.EfficientNet_B0_Weights, "classifier.1"),
    "squeezenet1_1": (models.SqueezeNet1_1_Weights, "classifier"),
}


def _get_model_and_head(name: str, pretrained: bool) -> tuple[nn.Module, str, int]:
    """Instantiate a torchvision model and return (model, head_attr, in_features)."""
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        return model, "fc", model.fc.in_features

    if name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        return model, "fc", model.fc.in_features

    if name == "vit_b_16":
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        model = models.vit_b_16(weights=weights)
        in_features = model.heads.head.in_features
        return model, "heads.head", in_features

    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        return model, "classifier.1", in_features

    if name == "squeezenet1_1":
        # Use ImageNet1K V1 weights by default when pretrained=True.
        weights = (
            models.SqueezeNet1_1_Weights.IMAGENET1K_V1 if pretrained else None
        )
        model = models.squeezenet1_1(weights=weights)
        # The backbone outputs 512 channels; we'll attach our own classifier head.
        in_features = 512
        return model, "classifier", in_features

    raise ValueError(
        f"Unknown encoder '{name}'. Choose from: {list(_REGISTRY.keys())}"
    )


def _set_nested_attr(model: nn.Module, attr_path: str, value: nn.Module) -> None:
    """Set a nested attribute like ``heads.head`` on a module."""
    parts = attr_path.split(".")
    parent = model
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = value
    else:
        setattr(parent, last, value)


def build_model(
    encoder: str = "resnet18",
    pretrained: bool = True,
    freeze_encoder: bool = False,
) -> nn.Module:
    """Build a binary classifier from a pretrained backbone.

    The final classification head is replaced with ``Linear(in_features, 1)``
    for binary classification with ``BCEWithLogitsLoss``.
    """
    model, head_attr, in_features = _get_model_and_head(encoder, pretrained)

    if encoder == "squeezenet1_1":
        # Replace the classifier block entirely with a simple global pool + linear head.
        model.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_features, 1),
        )
    else:
        _set_nested_attr(model, head_attr, nn.Linear(in_features, 1))

    if freeze_encoder:
        head_parts = set(head_attr.split("."))
        for name, param in model.named_parameters():
            name_parts = name.split(".")
            if not head_parts.intersection(name_parts):
                param.requires_grad = False

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(
        "Built %s (pretrained=%s, freeze=%s): %d/%d trainable params",
        encoder,
        pretrained,
        freeze_encoder,
        n_trainable,
        n_total,
    )
    return model
