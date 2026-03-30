from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from cell_size.classifier.models import build_model, get_classifier_module


def test_timm_vit_small_forward_smoke() -> None:
    pytest.importorskip("timm")

    model = build_model(
        encoder="timm/vit_small_patch16_dinov3.lvd1689m",
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

    encoder = "timm/vit_small_patch16_dinov3.lvd1689m"
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
        encoder="timm/vit_small_patch16_dinov3.lvd1689m",
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
