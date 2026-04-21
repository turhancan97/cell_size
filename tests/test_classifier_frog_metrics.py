from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("sklearn")

import cell_size.classifier.inference as inference


def test_extract_frog_id_examples_and_invalid() -> None:
    assert inference._extract_frog_id("TIFF_AH_001_04") == 1
    assert inference._extract_frog_id("TIFF_AH_476_10") == 476
    assert inference._extract_frog_id("TIFF_AH_ABC_10") is None
    assert inference._extract_frog_id("imgA") is None


def test_safe_ratio_division_guard() -> None:
    assert inference._safe_ratio(10, 2) == 5.0
    assert pd.isna(inference._safe_ratio(10, 0))
    assert pd.isna(inference._safe_ratio(None, 2))
    assert pd.isna(inference._safe_ratio(10, None))


def test_build_frog_aggregated_metrics_mean_std_counts() -> None:
    df = pd.DataFrame(
        [
            {
                "image_path": "TIFF_AH_001_01",
                "mask_index": 1,
                "area_px": 10.0,
                "major_axis_px": 5.0,
                "minor_axis_px": 2.5,
                "cell_axis_ratio": 2.0,
                "nc_ratio": 0.40,
            },
            {
                "image_path": "TIFF_AH_001_02",
                "mask_index": 2,
                "area_px": 14.0,
                "major_axis_px": 7.0,
                "minor_axis_px": 3.5,
                "cell_axis_ratio": 2.0,
                "nc_ratio": 0.20,
            },
            {
                "image_path": "TIFF_AH_476_01",
                "mask_index": 1,
                "area_px": 20.0,
                "major_axis_px": 8.0,
                "minor_axis_px": 4.0,
                "cell_axis_ratio": 2.0,
                "nc_ratio": 0.50,
            },
        ]
    )

    frog_df, unparsed = inference._build_frog_aggregated_metrics(df)
    assert unparsed == []
    assert list(frog_df["frog_id"]) == [1, 476]

    frog_1 = frog_df.loc[frog_df["frog_id"] == 1].iloc[0]
    assert int(frog_1["n_images"]) == 2
    assert int(frog_1["n_cells"]) == 2
    assert float(frog_1["area_px_mean"]) == pytest.approx(12.0)
    assert float(frog_1["area_px_std"]) == pytest.approx(2.82842712)
    assert float(frog_1["nc_ratio_mean"]) == pytest.approx(0.30)
    assert float(frog_1["nc_ratio_std"]) == pytest.approx(0.14142135)

    frog_476 = frog_df.loc[frog_df["frog_id"] == 476].iloc[0]
    assert int(frog_476["n_images"]) == 1
    assert int(frog_476["n_cells"]) == 1
    assert pd.isna(frog_476["area_px_std"])


def test_build_frog_aggregated_metrics_excludes_unparseable_names() -> None:
    df = pd.DataFrame(
        [
            {"image_path": "TIFF_AH_001_01", "mask_index": 1, "area_px": 10.0},
            {"image_path": "unparseable_image", "mask_index": 2, "area_px": 40.0},
        ]
    )
    frog_df, unparsed = inference._build_frog_aggregated_metrics(df)

    assert unparsed == ["unparseable_image"]
    assert len(frog_df) == 1
    assert int(frog_df.loc[0, "frog_id"]) == 1
    assert int(frog_df.loc[0, "n_cells"]) == 1


def test_write_frog_aggregated_metrics_empty_writes_header_only(tmp_path: Path) -> None:
    empty = pd.DataFrame(
        columns=[
            "image_path",
            "frog_id",
            "mask_index",
            "area_px",
            "major_axis_px",
            "minor_axis_px",
            "cell_axis_ratio",
            "nc_ratio",
        ]
    )

    out_path = tmp_path / "frog_aggregated_metrics.csv"
    written = inference._write_frog_aggregated_metrics(empty, out_path)
    assert written == out_path
    assert out_path.is_file()

    out_df = pd.read_csv(out_path)
    assert out_df.empty
    assert "frog_id" in out_df.columns
    assert "n_images" in out_df.columns
    assert "n_cells" in out_df.columns
    assert "area_px_mean" in out_df.columns
    assert "area_px_std" in out_df.columns
