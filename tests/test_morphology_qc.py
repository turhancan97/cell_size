from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cell_size.classifier.morphology_qc import (
    MorphologyQCConfig,
    apply_morphology_qc,
    build_frog_aggregated_metrics,
    run_morphology_qc,
    run_morphology_qc_from_csv,
)


def _qc_input_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "image_path": "TIFF_AH_001_01",
                "frog_id": 1,
                "mask_index": 1,
                "area_px": 100.0,
                "area_um2": 10.0,
                "nucleus_area_px": 20.0,
                "nucleus_area_um2": 2.0,
                "nc_ratio": 0.20,
            },
            {
                "image_path": "TIFF_AH_001_01",
                "frog_id": 1,
                "mask_index": 2,
                "area_px": 100.0,
                "area_um2": 10.0,
                "nucleus_area_px": np.nan,
                "nucleus_area_um2": np.nan,
                "nc_ratio": np.nan,
            },
            {
                "image_path": "TIFF_AH_002_01",
                "frog_id": 2,
                "mask_index": 1,
                "area_px": 100.0,
                "area_um2": 10.0,
                "nucleus_area_px": 4.0,
                "nucleus_area_um2": 0.4,
                "nc_ratio": 0.04,
            },
            {
                "image_path": "TIFF_AH_002_01",
                "frog_id": 2,
                "mask_index": 2,
                "area_px": 100.0,
                "area_um2": 10.0,
                "nucleus_area_px": 80.0,
                "nucleus_area_um2": 8.0,
                "nc_ratio": 0.80,
            },
            {
                "image_path": "TIFF_AH_003_01",
                "frog_id": 3,
                "mask_index": 1,
                "area_px": 0.0,
                "area_um2": 0.0,
                "nucleus_area_px": 10.0,
                "nucleus_area_um2": 1.0,
                "nc_ratio": 0.10,
            },
        ]
    )


def test_apply_morphology_qc_reasons() -> None:
    annotated = apply_morphology_qc(_qc_input_df(), MorphologyQCConfig())

    assert annotated["qc_pass"].tolist() == [True, False, False, False, False]
    assert annotated.loc[0, "qc_reason"] == "pass"
    assert "missing_or_invalid_nucleus_area_px" in annotated.loc[1, "qc_reason"]
    assert "invalid_nc_ratio" in annotated.loc[1, "qc_reason"]
    assert "nc_ratio_below_min" in annotated.loc[2, "qc_reason"]
    assert "nc_ratio_above_max" in annotated.loc[3, "qc_reason"]
    assert "invalid_area_px" in annotated.loc[4, "qc_reason"]


def test_run_morphology_qc_writes_expected_outputs_and_preserves_raw(tmp_path: Path) -> None:
    raw_df = _qc_input_df()
    raw_before = raw_df.copy(deep=True)

    result = run_morphology_qc(raw_df, tmp_path, MorphologyQCConfig())

    pd.testing.assert_frame_equal(raw_df, raw_before)
    assert len(result.clean_df) == 1
    assert len(result.rejected_df) == 4
    assert result.clean_df["qc_pass"].all()
    assert set(result.paths) == {
        "filtered_areas_qc",
        "filtered_areas_qc_rejected",
        "frog_aggregated_metrics_qc",
        "morphology_qc_image_summary",
        "morphology_qc_frog_summary",
        "morphology_qc_threshold_sensitivity",
        "frog_aggregated_metrics_qc_comparison",
    }
    for path in result.paths.values():
        assert path.is_file()

    raw_schema = list(build_frog_aggregated_metrics(raw_df).columns)
    qc_schema = list(pd.read_csv(result.paths["frog_aggregated_metrics_qc"]).columns)
    assert qc_schema == raw_schema

    image_summary = pd.read_csv(result.paths["morphology_qc_image_summary"])
    row = image_summary.loc[image_summary["image_path"] == "TIFF_AH_001_01"].iloc[0]
    assert int(row["n_cells"]) == 2
    assert int(row["n_qc_pass"]) == 1
    assert int(row["n_qc_rejected"]) == 1

    comparison = pd.read_csv(result.paths["frog_aggregated_metrics_qc_comparison"])
    frog_1 = comparison.loc[comparison["frog_id"] == 1].iloc[0]
    assert int(frog_1["n_cells_raw"]) == 2
    assert int(frog_1["n_cells_qc"]) == 1
    assert float(frog_1["n_cells_delta"]) == pytest.approx(-1)

    frog_2 = comparison.loc[comparison["frog_id"] == 2].iloc[0]
    assert int(frog_2["n_cells_raw"]) == 2
    assert int(frog_2["n_cells_qc"]) == 0
    assert float(frog_2["n_cells_delta"]) == pytest.approx(-2)


def test_run_morphology_qc_from_csv(tmp_path: Path) -> None:
    input_csv = tmp_path / "filtered_areas.csv"
    _qc_input_df().to_csv(input_csv, index=False)

    result = run_morphology_qc_from_csv(input_csv, tmp_path, MorphologyQCConfig())

    assert result.paths["filtered_areas_qc"].is_file()
    assert result.paths["filtered_areas_qc_rejected"].is_file()
    assert len(pd.read_csv(result.paths["filtered_areas_qc"])) == 1


def test_threshold_sensitivity_counts(tmp_path: Path) -> None:
    result = run_morphology_qc(
        _qc_input_df(),
        tmp_path,
        MorphologyQCConfig(sensitivity_max_nc_ratios=(0.30, 0.50, 0.80)),
    )

    sensitivity = result.sensitivity_df.set_index("max_nc_ratio")
    assert int(sensitivity.loc[0.30, "n_rejected_by_nc_ratio_above_max"]) == 1
    assert int(sensitivity.loc[0.50, "n_rejected_by_nc_ratio_above_max"]) == 1
    assert int(sensitivity.loc[0.80, "n_rejected_by_nc_ratio_above_max"]) == 0
