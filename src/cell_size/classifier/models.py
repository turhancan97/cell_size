"""Model factory for binary cell-quality classifiers.

Supports torchvision encoders and generic timm encoders via ``timm/<name>``.
Also supports efficient probing for timm ViT encoders.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
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
_EFFICIENT_PROBING_DEFAULTS = {
    "num_queries": 32,
    "num_heads": 1,
    "d_out": 1,
    "qkv_bias": False,
    "qk_scale": None,
}


def _is_timm_encoder(name: str) -> bool:
    return name.startswith("timm/") and len(name.split("/", 1)[1].strip()) > 0


def _supported_encoder_hint() -> str:
    tv = ", ".join(_TORCHVISION_ENCODERS)
    return f"{tv}, timm/<model_name>"


def _normalize_efficient_probing_cfg(cfg: dict[str, Any] | None) -> dict[str, Any]:
    if cfg is None:
        cfg = {}

    normalized = {
        "num_queries": int(cfg.get("num_queries", _EFFICIENT_PROBING_DEFAULTS["num_queries"])),
        "num_heads": int(cfg.get("num_heads", _EFFICIENT_PROBING_DEFAULTS["num_heads"])),
        "d_out": int(cfg.get("d_out", _EFFICIENT_PROBING_DEFAULTS["d_out"])),
        "qkv_bias": bool(cfg.get("qkv_bias", _EFFICIENT_PROBING_DEFAULTS["qkv_bias"])),
        "qk_scale": cfg.get("qk_scale", _EFFICIENT_PROBING_DEFAULTS["qk_scale"]),
    }
    if normalized["qk_scale"] is not None:
        normalized["qk_scale"] = float(normalized["qk_scale"])

    if normalized["num_queries"] <= 0:
        raise ValueError("efficient_probing.num_queries must be > 0.")
    if normalized["num_heads"] <= 0:
        raise ValueError("efficient_probing.num_heads must be > 0.")
    if normalized["d_out"] <= 0:
        raise ValueError("efficient_probing.d_out must be > 0.")

    return normalized


class EfficientProbing(nn.Module):
    """Efficient probing head operating on patch tokens."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        num_queries: int = 32,
        d_out: int = 1,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("EfficientProbing dim must be > 0.")
        if num_heads <= 0:
            raise ValueError("EfficientProbing num_heads must be > 0.")
        if dim % num_heads != 0:
            raise ValueError(
                f"EfficientProbing requires dim divisible by num_heads, got dim={dim}, num_heads={num_heads}."
            )
        if d_out <= 0:
            raise ValueError("EfficientProbing d_out must be > 0.")
        if dim % d_out != 0:
            raise ValueError(f"EfficientProbing requires dim divisible by d_out, got dim={dim}, d_out={d_out}.")
        if num_queries <= 0:
            raise ValueError("EfficientProbing num_queries must be > 0.")

        c_prime = dim // d_out
        if c_prime % num_queries != 0:
            raise ValueError(
                "EfficientProbing requires (dim // d_out) divisible by num_queries, "
                f"got dim={dim}, d_out={d_out}, num_queries={num_queries}."
            )

        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.d_out = int(d_out)
        self.num_queries = int(num_queries)
        self.c_prime = int(c_prime)
        self.scale = float(qk_scale) if qk_scale is not None else (dim // num_heads) ** -0.5

        self.v = nn.Linear(dim, self.c_prime, bias=qkv_bias)
        self.cls_token = nn.Parameter(torch.randn(1, self.num_queries, dim) * 0.02)

    def forward(self, x: torch.Tensor, cls: torch.Tensor | None = None, **_: Any) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"EfficientProbing expects x with shape [B, N, C], got {tuple(x.shape)}.")
        bsz, n_tokens, channels = x.shape
        if channels != self.dim:
            raise ValueError(f"EfficientProbing expected token dim {self.dim}, got {channels}.")

        if cls is not None:
            cls_token = cls
        else:
            cls_token = self.cls_token.expand(bsz, -1, -1)

        q = cls_token.reshape(
            bsz,
            self.num_queries,
            self.num_heads,
            channels // self.num_heads,
        ).permute(0, 2, 1, 3)
        k = x.reshape(
            bsz,
            n_tokens,
            self.num_heads,
            channels // self.num_heads,
        ).permute(0, 2, 1, 3)
        q = q * self.scale

        v = self.v(x).reshape(
            bsz,
            n_tokens,
            self.num_queries,
            self.c_prime // self.num_queries,
        ).permute(0, 2, 1, 3)

        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = attn.mean(dim=1)  # average across heads for multi-head support

        x_cls = torch.matmul(attn.unsqueeze(2), v).view(bsz, self.c_prime)
        return x_cls


class EfficientProbingViTClassifier(nn.Module):
    """Classifier wrapping a timm ViT backbone with efficient probing."""

    is_efficient_probing_model = True

    def __init__(
        self,
        backbone: nn.Module,
        token_dim: int,
        probing_cfg: dict[str, Any],
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.probing_cfg = dict(probing_cfg)
        self.probe = EfficientProbing(dim=token_dim, **self.probing_cfg)
        self.classifier = nn.Linear(token_dim // int(self.probing_cfg["d_out"]), 1)

    def _patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone.forward_features(x)  # type: ignore[attr-defined]
        if not torch.is_tensor(tokens):
            raise ValueError(
                "Efficient probing requires forward_features(x) to return a tensor [B, N, C], "
                f"got {type(tokens).__name__}."
            )
        if tokens.ndim != 3:
            raise ValueError(
                "Efficient probing requires forward_features(x) with shape [B, N, C], "
                f"got shape {tuple(tokens.shape)}."
            )
        if tokens.shape[1] <= 1:
            raise ValueError(
                "Efficient probing requires patch tokens beyond CLS token; "
                f"got token count N={tokens.shape[1]}."
            )
        return tokens[:, 1:, :]

    def extract_probe_features(self, x: torch.Tensor) -> torch.Tensor:
        patches = self._patch_tokens(x)
        return self.probe(patches)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_cls = self.extract_probe_features(x)
        return self.classifier(x_cls)


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
    if isinstance(model, EfficientProbingViTClassifier):
        return model.classifier

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
    use_efficient_probing: bool = False,
    efficient_probing_cfg: dict[str, Any] | None = None,
) -> nn.Module:
    """Build a binary classifier from a selected backbone.

    - torchvision models replace their final classification head with
      ``Linear(in_features, 1)``.
    - timm models are created with ``num_classes=1`` directly.
    - when ``use_mlp_head=True``, head becomes:
      ``Linear(in_features,128)->ReLU->Linear(128,32)->ReLU->Linear(32,8)->ReLU->Linear(8,1)``.
    - when ``use_efficient_probing=True``, use patch-token probing for timm ViT encoders.
    """
    if use_mlp_head and use_efficient_probing:
        raise ValueError("`use_mlp_head` and `use_efficient_probing` are mutually exclusive.")

    if use_efficient_probing:
        if not _is_timm_encoder(encoder):
            raise ValueError("Efficient probing is supported only for timm ViT encoders (encoder must be `timm/...`).")
        timm_name = encoder.split("/", 1)[1].strip()
        if "vit" not in timm_name.lower():
            raise ValueError(
                f"Efficient probing supports only ViT timm models; got '{timm_name}'."
            )
        if timm is None:
            raise ImportError(
                "Efficient probing requested for a timm encoder but timm is not installed. "
                "Install dependencies with: pip install -e ."
            )

        probing_cfg = _normalize_efficient_probing_cfg(efficient_probing_cfg)
        backbone = timm.create_model(timm_name, pretrained=pretrained, num_classes=1)
        head_module = get_classifier_module(backbone, encoder)
        token_dim = _infer_classifier_in_features(head_module, encoder)
        model = EfficientProbingViTClassifier(
            backbone=backbone,
            token_dim=token_dim,
            probing_cfg=probing_cfg,
        )

        if freeze_encoder:
            for param in model.backbone.parameters():
                param.requires_grad = False
            for param in model.probe.parameters():
                param.requires_grad = True
            for param in model.classifier.parameters():
                param.requires_grad = True

        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        logger.info(
            "Built %s with efficient probing (pretrained=%s, freeze=%s): %d/%d trainable params",
            encoder,
            pretrained,
            freeze_encoder,
            n_trainable,
            n_total,
        )
        return model

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
