from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import cell_size.classifier.models as classifier_models
from cell_size.classifier.models import (
    EfficientProbingViTClassifier,
    build_model,
    get_classifier_module,
)

_TIMM_VIT_ENCODER = "timm/vit_small_patch16_dinov3.lvd1689m"


def test_timm_vit_small_forward_smoke() -> None:
    pytest.importorskip("timm")

    model = build_model(
        encoder=_TIMM_VIT_ENCODER,
        pretrained=False,
        freeze_encoder=False,
    )
    model.eval()

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        y = model(x)

    assert y.shape == (2, 1)


def test_timm_vit_small_freeze_encoder_only_head_trainable() -> None:
    pytest.importorskip("timm")

    encoder = _TIMM_VIT_ENCODER
    model = build_model(
        encoder=encoder,
        pretrained=False,
        freeze_encoder=True,
    )
    head = get_classifier_module(model, encoder)
    head_param_ids = {id(p) for p in head.parameters()}

    assert len(head_param_ids) > 0

    saw_non_head_param = False
    for p in model.parameters():
        if id(p) in head_param_ids:
            assert p.requires_grad
        else:
            saw_non_head_param = True
            assert not p.requires_grad

    assert saw_non_head_param


def test_timm_vit_small_mlp_head_forward_smoke() -> None:
    pytest.importorskip("timm")

    model = build_model(
        encoder=_TIMM_VIT_ENCODER,
        pretrained=False,
        freeze_encoder=False,
        use_mlp_head=True,
    )
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 1)


def test_resnet18_mlp_head_forward_and_shape() -> None:
    model = build_model(
        encoder="resnet18",
        pretrained=False,
        freeze_encoder=False,
        use_mlp_head=True,
    )
    head = get_classifier_module(model, "resnet18")
    assert isinstance(head, nn.Sequential)

    x = torch.randn(3, 3, 224, 224)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (3, 1)


def test_resnet18_mlp_head_freeze_encoder_only_head_trainable() -> None:
    encoder = "resnet18"
    model = build_model(
        encoder=encoder,
        pretrained=False,
        freeze_encoder=True,
        use_mlp_head=True,
    )
    head = get_classifier_module(model, encoder)
    head_param_ids = {id(p) for p in head.parameters()}
    assert len(head_param_ids) > 0

    saw_non_head_param = False
    for p in model.parameters():
        if id(p) in head_param_ids:
            assert p.requires_grad
        else:
            saw_non_head_param = True
            assert not p.requires_grad

    assert saw_non_head_param


def test_efficient_probing_forward_shape_smoke() -> None:
    pytest.importorskip("timm")

    model = build_model(
        encoder=_TIMM_VIT_ENCODER,
        pretrained=False,
        freeze_encoder=False,
        use_efficient_probing=True,
    )
    assert isinstance(model, EfficientProbingViTClassifier)

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 1)


def test_efficient_probing_freeze_encoder_true_trainables() -> None:
    pytest.importorskip("timm")

    model = build_model(
        encoder=_TIMM_VIT_ENCODER,
        pretrained=False,
        freeze_encoder=True,
        use_efficient_probing=True,
    )
    assert isinstance(model, EfficientProbingViTClassifier)

    backbone_params = list(model.backbone.parameters())
    probe_params = list(model.probe.parameters())
    cls_params = list(model.classifier.parameters())

    assert len(backbone_params) > 0
    assert len(probe_params) > 0
    assert len(cls_params) > 0
    assert all(not p.requires_grad for p in backbone_params)
    assert all(p.requires_grad for p in probe_params)
    assert all(p.requires_grad for p in cls_params)


def test_efficient_probing_freeze_encoder_false_backbone_trainable() -> None:
    pytest.importorskip("timm")

    model = build_model(
        encoder=_TIMM_VIT_ENCODER,
        pretrained=False,
        freeze_encoder=False,
        use_efficient_probing=True,
    )
    assert isinstance(model, EfficientProbingViTClassifier)
    assert any(p.requires_grad for p in model.backbone.parameters())


def test_efficient_probing_mutually_exclusive_with_mlp_head() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        build_model(
            encoder="resnet18",
            pretrained=False,
            freeze_encoder=False,
            use_mlp_head=True,
            use_efficient_probing=True,
        )


def test_efficient_probing_requires_timm_encoder_prefix() -> None:
    with pytest.raises(ValueError, match="timm ViT encoders"):
        build_model(
            encoder="vit_b_16",
            pretrained=False,
            freeze_encoder=False,
            use_efficient_probing=True,
        )


def test_efficient_probing_requires_timm_vit_name() -> None:
    with pytest.raises(ValueError, match="ViT"):
        build_model(
            encoder="timm/resnet18",
            pretrained=False,
            freeze_encoder=False,
            use_efficient_probing=True,
        )


def test_efficient_probing_errors_if_forward_features_not_bnc(monkeypatch: pytest.MonkeyPatch) -> None:
    class _DummyBadForwardFeatures(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head = nn.Linear(16, 1)

        def get_classifier(self) -> nn.Module:
            return self.head

        def forward_features(self, x: torch.Tensor) -> torch.Tensor:
            return torch.randn(x.shape[0], 16, device=x.device)

    class _StubTimm:
        @staticmethod
        def create_model(_name: str, pretrained: bool, num_classes: int) -> nn.Module:
            _ = pretrained, num_classes
            return _DummyBadForwardFeatures()

    monkeypatch.setattr(classifier_models, "timm", _StubTimm())

    model = build_model(
        encoder="timm/vit_dummy_bad_forward_features",
        pretrained=False,
        freeze_encoder=False,
        use_efficient_probing=True,
        efficient_probing_cfg={"num_queries": 4},
    )
    with pytest.raises(ValueError, match=r"\[B, N, C\]"):
        _ = model(torch.randn(2, 3, 224, 224))


def test_efficient_probing_checkpoint_roundtrip(tmp_path) -> None:
    pytest.importorskip("timm")
    pytest.importorskip("sklearn")
    from cell_size.classifier.inference import load_checkpoint

    probing_cfg = {
        "num_queries": 16,
        "num_heads": 1,
        "d_out": 1,
        "qkv_bias": False,
        "qk_scale": None,
    }
    model = build_model(
        encoder=_TIMM_VIT_ENCODER,
        pretrained=False,
        freeze_encoder=False,
        use_efficient_probing=True,
        efficient_probing_cfg=probing_cfg,
    )
    ckpt_path = tmp_path / "best_model.pt"
    torch.save(
        {
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "encoder": _TIMM_VIT_ENCODER,
            "use_mlp_head": False,
            "use_efficient_probing": True,
            "efficient_probing": probing_cfg,
            "val_f1": 0.5,
            "class_to_idx": {"bad": 0, "good": 1},
            "crop_size": 224,
        },
        ckpt_path,
    )

    loaded_model, ckpt = load_checkpoint(ckpt_path, device=torch.device("cpu"))
    assert isinstance(loaded_model, EfficientProbingViTClassifier)
    assert bool(ckpt["use_efficient_probing"]) is True
    assert ckpt["efficient_probing"] == probing_cfg

    with torch.no_grad():
        y = loaded_model(torch.randn(2, 3, 224, 224))
    assert y.shape == (2, 1)
