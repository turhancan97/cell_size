"""Model factory for binary cell-quality classifiers.

Supports torchvision encoders and generic timm encoders via ``timm/<name>``.
"""

from __future__ import annotations

import logging

import torch.nn as nn
from torchvision import models

try:
    import timm
except Exception:  # pragma: no cover - optional at import time
    timm = None

logger = logging.getLogger(__name__)

_TORCHVISION_ENCODERS = (
    "resnet18",
    "resnet50",
    "vit_b_16",
    "efficientnet_b0",
    "squeezenet1_1",
)
_MLP_HIDDEN_DIMS = (128, 32, 8)


def _is_timm_encoder(name: str) -> bool:
    return name.startswith("timm/") and len(name.split("/", 1)[1].strip()) > 0


def _supported_encoder_hint() -> str:
    tv = ", ".join(_TORCHVISION_ENCODERS)
    return f"{tv}, timm/<model_name>"


def _set_nested_attr(model: nn.Module, attr_path: str, value: nn.Module) -> None:
    """Set a nested attribute like ``heads.head`` on a module."""
    parts = attr_path.split(".")
    parent: nn.Module = model
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]  # type: ignore[index]
        else:
            parent = getattr(parent, part)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = value  # type: ignore[index]
    else:
        setattr(parent, last, value)


def _get_nested_module(model: nn.Module, attr_path: str) -> nn.Module:
    """Get a nested module like ``heads.head`` from a module."""
    module: nn.Module = model
    for part in attr_path.split("."):
        if part.isdigit():
            module = module[int(part)]  # type: ignore[index]
        else:
            module = getattr(module, part)
    return module


def _find_module_path(model: nn.Module, target: nn.Module) -> str | None:
    """Return dotted path for ``target`` module inside ``model``."""
    for name, module in model.named_modules():
        if module is target and name:
            return name
    return None


def _infer_classifier_in_features(head: nn.Module, encoder: str) -> int:
    if hasattr(head, "in_features"):
        return int(getattr(head, "in_features"))
    if isinstance(head, nn.Conv2d):
        return int(head.in_channels)
    raise ValueError(
        f"Could not infer classifier input features for encoder '{encoder}' "
        f"(head type: {type(head).__name__})."
    )


def _make_binary_head(in_features: int, use_mlp_head: bool) -> nn.Module:
    if not use_mlp_head:
        return nn.Linear(in_features, 1)

    layers: list[nn.Module] = []
    prev = int(in_features)
    for hidden in _MLP_HIDDEN_DIMS:
        layers.append(nn.Linear(prev, int(hidden)))
        layers.append(nn.ReLU())
        prev = int(hidden)
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


def get_classifier_module(model: nn.Module, encoder: str) -> nn.Module:
    """Resolve final classifier head module for a built model."""
    if encoder.startswith("resnet"):
        return model.fc
    if encoder == "efficientnet_b0":
        return model.classifier[1]
    if encoder == "vit_b_16":
        return model.heads.head
    if encoder == "squeezenet1_1":
        return model.classifier[2]

    if _is_timm_encoder(encoder):
        if hasattr(model, "get_classifier"):
            classifier = model.get_classifier()
            if isinstance(classifier, str) and classifier:
                module = _get_nested_module(model, classifier)
                if not isinstance(module, nn.Identity):
                    return module
            elif isinstance(classifier, nn.Module) and not isinstance(classifier, nn.Identity):
                return classifier

        for candidate in ("head", "fc", "classifier", "last_linear"):
            maybe = getattr(model, candidate, None)
            if isinstance(maybe, nn.Module) and not isinstance(maybe, nn.Identity):
                return maybe

        last_linear = None
        for module in model.modules():
            if isinstance(module, nn.Linear):
                last_linear = module
        if last_linear is not None:
            return last_linear

        raise ValueError(
            f"Could not resolve classifier head module for timm encoder '{encoder}'."
        )

    raise ValueError(
        f"Unknown encoder '{encoder}'. Choose from: {_supported_encoder_hint()}"
    )


def _get_model_and_head(name: str, pretrained: bool) -> tuple[nn.Module, str, int, bool]:
    """Instantiate a model and return (model, head_attr, in_features, replace_head)."""
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        return model, "fc", int(model.fc.in_features), True

    if name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        return model, "fc", int(model.fc.in_features), True

    if name == "vit_b_16":
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        model = models.vit_b_16(weights=weights)
        return model, "heads.head", int(model.heads.head.in_features), True

    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        return model, "classifier.1", int(model.classifier[1].in_features), True

    if name == "squeezenet1_1":
        weights = models.SqueezeNet1_1_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.squeezenet1_1(weights=weights)
        return model, "classifier.2", 512, True

    if _is_timm_encoder(name):
        if timm is None:
            raise ImportError(
                "Encoder starts with 'timm/' but timm is not installed. "
                "Install dependencies with: pip install -e ."
            )
        timm_name = name.split("/", 1)[1].strip()
        model = timm.create_model(timm_name, pretrained=pretrained, num_classes=1)
        head_module = get_classifier_module(model, name)
        head_attr = _find_module_path(model, head_module)
        if head_attr is None:
            raise ValueError(
                f"Failed to resolve classifier path for timm encoder '{name}'."
            )
        in_features = _infer_classifier_in_features(head_module, name)
        return model, head_attr, in_features, False

    raise ValueError(
        f"Unknown encoder '{name}'. Choose from: {_supported_encoder_hint()}"
    )


def build_model(
    encoder: str = "resnet18",
    pretrained: bool = True,
    freeze_encoder: bool = False,
    use_mlp_head: bool = False,
) -> nn.Module:
    """Build a binary classifier from a selected backbone.

    - torchvision models replace their final classification head with
      ``Linear(in_features, 1)``.
    - timm models are created with ``num_classes=1`` directly.
    - when ``use_mlp_head=True``, head becomes:
      ``Linear(in_features,128)->ReLU->Linear(128,32)->ReLU->Linear(32,8)->ReLU->Linear(8,1)``.
    """
    model, head_attr, in_features, replace_head = _get_model_and_head(encoder, pretrained)

    if encoder == "squeezenet1_1":
        model.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            _make_binary_head(in_features, use_mlp_head),
        )
    elif replace_head:
        _set_nested_attr(model, head_attr, _make_binary_head(in_features, use_mlp_head))
    elif use_mlp_head:
        _set_nested_attr(model, head_attr, _make_binary_head(in_features, use_mlp_head))

    head_module = get_classifier_module(model, encoder)
    if freeze_encoder:
        for param in model.parameters():
            param.requires_grad = False
        for param in head_module.parameters():
            param.requires_grad = True

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
