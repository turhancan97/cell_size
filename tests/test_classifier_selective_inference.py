from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

pytest.importorskip("sklearn")

import cell_size.classifier.inference as inference


class _LogitFromTensorModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use the first pixel/channel as pre-defined logit.
        return x[:, 0, 0, 0].unsqueeze(1)


def _crop_from_logit(logit: float) -> np.ndarray:
    return np.full((2, 2, 3), fill_value=float(logit), dtype=np.float32)


def _identity_transform(crop: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(crop).permute(2, 0, 1)


def test_classify_cells_single_threshold_backward_compatible() -> None:
    model = _LogitFromTensorModel()
    crops = [
        (1, _crop_from_logit(-2.0)),
        (2, _crop_from_logit(2.0)),
    ]

    out = inference.classify_cells(
        model=model,
        crops=crops,
        transform=_identity_transform,
        device=torch.device("cpu"),
        confidence_threshold=0.7,
        batch_size=8,
        selective_rejection_enabled=False,
    )

    assert [row["predicted_verdict"] for row in out] == ["bad", "good"]
    assert [row["accepted"] for row in out] == [True, True]


def test_classify_cells_selective_rejection_policy() -> None:
    model = _LogitFromTensorModel()
    crops = [
        (1, _crop_from_logit(-3.0)),  # p ~= 0.05 -> bad
        (2, _crop_from_logit(0.0)),   # p = 0.50 -> rejected
        (3, _crop_from_logit(3.0)),   # p ~= 0.95 -> good
    ]

    out = inference.classify_cells(
        model=model,
        crops=crops,
        transform=_identity_transform,
        device=torch.device("cpu"),
        batch_size=8,
        selective_rejection_enabled=True,
        selective_t_bad=0.09,
        selective_t_good=0.51,
    )

    assert [row["predicted_verdict"] for row in out] == ["bad", "rejected", "good"]
    assert [row["accepted"] for row in out] == [True, False, True]


def test_classify_cells_selective_rejection_invalid_thresholds() -> None:
    model = _LogitFromTensorModel()
    crops = [(1, _crop_from_logit(0.0))]

    try:
        inference.classify_cells(
            model=model,
            crops=crops,
            transform=_identity_transform,
            device=torch.device("cpu"),
            selective_rejection_enabled=True,
            selective_t_bad=0.8,
            selective_t_good=0.2,
        )
    except ValueError as exc:
        assert "0 <= t_bad < t_good <= 1" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid selective thresholds")


def test_run_inference_predictions_csv_columns_and_rejected(monkeypatch, tmp_path: Path) -> None:
    class _DummyModel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros((x.shape[0], 1), dtype=torch.float32, device=x.device)

    def _fake_load_checkpoint(_checkpoint_path: str | Path, _device: torch.device):
        return _DummyModel(), {"crop_size": 224}

    def _fake_find_processed_images(_data_dir: Path):
        image_dir = tmp_path / "imgA"
        image_dir.mkdir(parents=True, exist_ok=True)
        return [image_dir / "imgA.tif"]

    def _fake_load_mask(_image_folder: Path, _image_stem: str):
        return np.array([[0, 1]], dtype=np.int32)

    def _fake_read_rgb(_img_path: Path):
        return np.zeros((1, 2, 3), dtype=np.uint8)

    def _fake_extract_all_crops(_img_rgb, _mask, _crop_size, _padding_pct, _mask_bg):
        return [(1, np.zeros((2, 2, 3), dtype=np.uint8))]

    def _fake_classify_cells(*args, **kwargs):
        _ = args, kwargs
        return [{
            "mask_index": 1,
            "predicted_verdict": "rejected",
            "confidence": 0.5,
            "accepted": False,
        }]

    monkeypatch.setattr(inference, "load_checkpoint", _fake_load_checkpoint)
    monkeypatch.setattr(inference, "_find_processed_images", _fake_find_processed_images)
    monkeypatch.setattr(inference, "_load_mask", _fake_load_mask)
    monkeypatch.setattr(inference, "_read_image_rgb", _fake_read_rgb)
    monkeypatch.setattr(inference, "extract_all_crops", _fake_extract_all_crops)
    monkeypatch.setattr(inference, "classify_cells", _fake_classify_cells)

    cfg = SimpleNamespace(
        gpu=False,
        crop_size=224,
        confidence_threshold=0.7,
        crop_padding_pct=0.2,
        mask_background=False,
        batch_size=4,
        selective_rejection=SimpleNamespace(enabled=True, t_bad=0.09, t_good=0.51),
    )

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir = tmp_path / "out"
    df = inference.run_inference(
        data_dir=data_dir,
        checkpoint_path=tmp_path / "best_model.pt",
        output_dir=out_dir,
        cfg=cfg,
    )

    assert list(df.columns) == [
        "image_path",
        "mask_index",
        "predicted_verdict",
        "confidence",
        "accepted",
    ]
    assert df.loc[0, "predicted_verdict"] == "rejected"
    assert bool(df.loc[0, "accepted"]) is False
    assert (out_dir / "predictions.csv").is_file()
