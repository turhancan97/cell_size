from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

import cell_size.classifier.inference as inference


def test_generate_filtered_overlays_from_existing_predictions(monkeypatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    image_dir = data_dir / "TIFF_AH_160_01"
    image_dir.mkdir(parents=True, exist_ok=True)
    img_path = image_dir / "TIFF_AH_160_01.tif"
    img_path.write_bytes(b"fake")

    predictions_df = pd.DataFrame(
        [
            {
                "image_path": "TIFF_AH_160_01",
                "mask_index": 1,
                "predicted_verdict": "good",
                "confidence": 0.95,
                "accepted": True,
                "frog_id": 160,
            },
            {
                "image_path": "TIFF_AH_160_01",
                "mask_index": 2,
                "predicted_verdict": "bad",
                "confidence": 0.10,
                "accepted": True,
                "frog_id": 160,
            },
        ]
    )

    mask = np.array([[0, 1], [2, 2]], dtype=np.int32)
    nuc_mask = np.array([[0, 1], [0, 2]], dtype=np.int32)
    img_rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    seen: dict[str, object] = {}

    monkeypatch.setattr(inference, "_find_processed_images", lambda _data_dir: [img_path])
    monkeypatch.setattr(inference, "_load_mask", lambda _folder, _stem: mask)
    monkeypatch.setattr(inference, "_load_nucleus_mask", lambda _folder, _stem: nuc_mask)
    monkeypatch.setattr(inference, "_read_image_rgb", lambda _path: img_rgb)

    def _fake_generate_filtered_overlay(img_rgb_arg, mask_arg, preds_arg, overlay_path, nuc_masks=None):
        seen["img_rgb_shape"] = img_rgb_arg.shape
        seen["mask_shape"] = mask_arg.shape
        seen["pred_count"] = len(preds_arg)
        seen["overlay_path"] = overlay_path
        seen["nuc_masks"] = nuc_masks
        overlay_path.write_bytes(b"overlay")

    monkeypatch.setattr(inference, "generate_filtered_overlay", _fake_generate_filtered_overlay)

    out_dir = tmp_path / "classify_output"
    overlays_dir = inference.generate_filtered_overlays_from_predictions(data_dir, predictions_df, out_dir)

    assert overlays_dir == out_dir / "overlays"
    assert (overlays_dir / "TIFF_AH_160_01_filtered_overlay.jpg").is_file()
    assert seen["img_rgb_shape"] == (2, 2, 3)
    assert seen["mask_shape"] == (2, 2)
    assert seen["pred_count"] == 2
    assert seen["nuc_masks"] is nuc_mask
