"""Morphology quality control for classifier-good cell measurements."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FROG_ID_RE = re.compile(r"^TIFF_AH_(\d+)_\d+$")
DEFAULT_SENSITIVITY_MAX_NC_RATIOS = (0.30, 0.40, 0.50, 0.80)
QC_PASS_REASON = "pass"


@dataclass(frozen=True)
class MorphologyQCConfig:
    enabled: bool = True
    require_nucleus: bool = True
    min_nc_ratio: float = 0.05
    max_nc_ratio: float = 0.50
    sensitivity_max_nc_ratios: tuple[float, ...] = DEFAULT_SENSITIVITY_MAX_NC_RATIOS


@dataclass(frozen=True)
class MorphologyQCResult:
    clean_df: pd.DataFrame
    rejected_df: pd.DataFrame
    frog_aggregated_qc_df: pd.DataFrame
    image_summary_df: pd.DataFrame
    frog_summary_df: pd.DataFrame
    sensitivity_df: pd.DataFrame
    comparison_df: pd.DataFrame
    paths: dict[str, Path]


def resolve_morphology_qc_config(cfg: Any | None = None) -> MorphologyQCConfig:
    """Resolve a Hydra/namespace/dict-like QC config into a typed config."""
    if cfg is None:
        return MorphologyQCConfig()

    def _get(name: str, default: Any) -> Any:
        if isinstance(cfg, dict):
            return cfg.get(name, default)
        return getattr(cfg, name, default)

    ratios = _get("sensitivity_max_nc_ratios", DEFAULT_SENSITIVITY_MAX_NC_RATIOS)
    if ratios is None:
        ratios = DEFAULT_SENSITIVITY_MAX_NC_RATIOS

    return MorphologyQCConfig(
        enabled=bool(_get("enabled", True)),
        require_nucleus=bool(_get("require_nucleus", True)),
        min_nc_ratio=float(_get("min_nc_ratio", 0.05)),
        max_nc_ratio=float(_get("max_nc_ratio", 0.50)),
        sensitivity_max_nc_ratios=tuple(float(x) for x in ratios),
    )


def _extract_frog_id(image_name: str) -> int | None:
    m = FROG_ID_RE.match(str(image_name))
    if m is None:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _ensure_frog_id(df: pd.DataFrame) -> pd.DataFrame:
    if "frog_id" in df.columns:
        return df
    out = df.copy()
    out["frog_id"] = out["image_path"].map(_extract_frog_id) if "image_path" in out.columns else np.nan
    return out


def _numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def _append_reason(reasons: pd.Series, mask: pd.Series, reason: str) -> pd.Series:
    mask = mask.fillna(False).astype(bool)
    if not mask.any():
        return reasons
    out = reasons.copy()
    current = out.loc[mask]
    out.loc[mask] = np.where(current.eq(""), reason, current + "|" + reason)
    return out


def apply_morphology_qc(
    filtered_df: pd.DataFrame,
    cfg: MorphologyQCConfig | Any | None = None,
) -> pd.DataFrame:
    """Return a copy of ``filtered_df`` annotated with ``qc_pass`` and ``qc_reason``."""
    qc_cfg = cfg if isinstance(cfg, MorphologyQCConfig) else resolve_morphology_qc_config(cfg)
    out = _ensure_frog_id(filtered_df.copy())
    reasons = pd.Series("", index=out.index, dtype="object")

    area_px = _numeric(out, "area_px")
    nucleus_area_px = _numeric(out, "nucleus_area_px")
    nc_ratio = _numeric(out, "nc_ratio")

    area_bad = area_px.isna() | ~np.isfinite(area_px) | (area_px <= 0)
    nucleus_bad = nucleus_area_px.isna() | ~np.isfinite(nucleus_area_px) | (nucleus_area_px <= 0)
    nc_bad = nc_ratio.isna() | ~np.isfinite(nc_ratio) | (nc_ratio <= 0)

    reasons = _append_reason(reasons, area_bad, "invalid_area_px")
    if qc_cfg.require_nucleus:
        reasons = _append_reason(reasons, nucleus_bad, "missing_or_invalid_nucleus_area_px")
    else:
        reasons = _append_reason(
            reasons,
            nucleus_bad & (nucleus_area_px.notna() | nc_ratio.notna()),
            "invalid_nucleus_area_px",
        )
    reasons = _append_reason(reasons, nc_bad, "invalid_nc_ratio")

    valid_nc = ~nc_bad
    reasons = _append_reason(reasons, valid_nc & (nc_ratio < qc_cfg.min_nc_ratio), "nc_ratio_below_min")
    reasons = _append_reason(reasons, valid_nc & (nc_ratio > qc_cfg.max_nc_ratio), "nc_ratio_above_max")

    out["qc_pass"] = reasons.eq("")
    out["qc_reason"] = reasons.mask(out["qc_pass"], QC_PASS_REASON)
    return out


def split_morphology_qc(
    filtered_df: pd.DataFrame,
    cfg: MorphologyQCConfig | Any | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    annotated = apply_morphology_qc(filtered_df, cfg)
    return (
        annotated.loc[annotated["qc_pass"]].copy(),
        annotated.loc[~annotated["qc_pass"]].copy(),
    )


def _frog_aggregate_columns(filtered_df: pd.DataFrame) -> list[str]:
    preferred_metrics = [
        "area_px",
        "area_um2",
        "major_axis_px",
        "minor_axis_px",
        "cell_axis_ratio",
        "major_axis_um",
        "minor_axis_um",
        "nucleus_area_px",
        "nucleus_major_axis_px",
        "nucleus_minor_axis_px",
        "nucleus_axis_ratio",
        "nc_ratio",
        "nucleus_area_um2",
        "nucleus_major_axis_um",
        "nucleus_minor_axis_um",
    ]
    numeric_metrics = [c for c in preferred_metrics if c in filtered_df.columns]
    if not numeric_metrics:
        numeric_metrics = [
            c
            for c in filtered_df.columns
            if pd.api.types.is_numeric_dtype(filtered_df[c])
            and c not in {"frog_id", "mask_index", "qc_pass"}
        ]
    out_cols = ["frog_id", "n_images", "n_cells"]
    for metric in numeric_metrics:
        out_cols.extend([f"{metric}_mean", f"{metric}_std"])
    return out_cols


def build_frog_aggregated_metrics(filtered_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-cell morphology metrics into one row per frog."""
    work = _ensure_frog_id(filtered_df.copy())
    out_cols = _frog_aggregate_columns(work)
    if work.empty:
        return pd.DataFrame(columns=out_cols)

    work = work.loc[work["frog_id"].notna()].copy()
    if work.empty:
        return pd.DataFrame(columns=out_cols)

    work["frog_id"] = work["frog_id"].astype(int)
    grouped = work.groupby("frog_id", sort=True)
    if "mask_index" in work.columns:
        agg_df = grouped.agg(n_images=("image_path", "nunique"), n_cells=("mask_index", "count"))
    else:
        agg_df = grouped.agg(n_images=("image_path", "nunique"), n_cells=("image_path", "size"))

    expected_cols = _frog_aggregate_columns(work)
    numeric_metrics = [c[:-5] for c in expected_cols if c.endswith("_mean")]
    for metric in numeric_metrics:
        agg_df[f"{metric}_mean"] = grouped[metric].mean()
        agg_df[f"{metric}_std"] = grouped[metric].std(ddof=1)

    return agg_df.reset_index()[expected_cols]


def _summary_by_group(annotated_df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if annotated_df.empty or group_col not in annotated_df.columns:
        return pd.DataFrame(
            columns=[
                group_col,
                "n_cells",
                "n_qc_pass",
                "n_qc_rejected",
                "qc_pass_fraction",
            ]
        )

    rows: list[dict[str, Any]] = []
    for group_value, group in annotated_df.groupby(group_col, dropna=False, sort=True):
        row: dict[str, Any] = {
            group_col: group_value,
            "n_cells": int(len(group)),
            "n_qc_pass": int(group["qc_pass"].sum()),
            "n_qc_rejected": int((~group["qc_pass"]).sum()),
        }
        row["qc_pass_fraction"] = row["n_qc_pass"] / max(row["n_cells"], 1)
        reasons = group.loc[~group["qc_pass"], "qc_reason"].fillna("")
        for reason in [
            "invalid_area_px",
            "missing_or_invalid_nucleus_area_px",
            "invalid_nucleus_area_px",
            "invalid_nc_ratio",
            "nc_ratio_below_min",
            "nc_ratio_above_max",
        ]:
            row[f"n_{reason}"] = int(reasons.str.contains(reason, regex=False).sum())
        rows.append(row)

    return pd.DataFrame(rows)


def build_threshold_sensitivity(
    filtered_df: pd.DataFrame,
    cfg: MorphologyQCConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for max_nc_ratio in cfg.sensitivity_max_nc_ratios:
        sensitivity_cfg = MorphologyQCConfig(
            enabled=cfg.enabled,
            require_nucleus=cfg.require_nucleus,
            min_nc_ratio=cfg.min_nc_ratio,
            max_nc_ratio=max_nc_ratio,
            sensitivity_max_nc_ratios=cfg.sensitivity_max_nc_ratios,
        )
        annotated = apply_morphology_qc(filtered_df, sensitivity_cfg)
        rejected = annotated.loc[~annotated["qc_pass"]]
        rejected_high = rejected["qc_reason"].fillna("").str.contains(
            "nc_ratio_above_max", regex=False
        )
        frog_rejected = (
            int(rejected.loc[rejected["frog_id"].notna(), "frog_id"].nunique())
            if "frog_id" in rejected.columns
            else 0
        )
        rows.append(
            {
                "max_nc_ratio": max_nc_ratio,
                "min_nc_ratio": cfg.min_nc_ratio,
                "n_cells_total": int(len(annotated)),
                "n_qc_pass": int(annotated["qc_pass"].sum()),
                "n_qc_rejected": int((~annotated["qc_pass"]).sum()),
                "n_rejected_by_nc_ratio_above_max": int(rejected_high.sum()),
                "n_frogs_with_any_rejection": frog_rejected,
                "n_frogs_retained": int(
                    annotated.loc[annotated["qc_pass"] & annotated["frog_id"].notna(), "frog_id"].nunique()
                )
                if "frog_id" in annotated.columns
                else 0,
            }
        )
    return pd.DataFrame(rows)


def build_frog_qc_comparison(
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
) -> pd.DataFrame:
    raw_agg = build_frog_aggregated_metrics(raw_df)
    qc_agg = build_frog_aggregated_metrics(clean_df)
    if raw_agg.empty and qc_agg.empty:
        return pd.DataFrame(columns=["frog_id"])

    comparison = raw_agg.merge(qc_agg, on="frog_id", how="outer", suffixes=("_raw", "_qc"))
    for count_col in ("n_images", "n_cells"):
        for suffix in ("raw", "qc"):
            col = f"{count_col}_{suffix}"
            if col in comparison.columns:
                comparison[col] = comparison[col].fillna(0).astype(int)

    shared_cols = [c for c in raw_agg.columns if c != "frog_id" and c in qc_agg.columns]
    for col in shared_cols:
        raw_col = f"{col}_raw"
        qc_col = f"{col}_qc"
        if raw_col in comparison.columns and qc_col in comparison.columns:
            raw_values = pd.to_numeric(comparison[raw_col], errors="coerce")
            qc_values = pd.to_numeric(comparison[qc_col], errors="coerce")
            comparison[f"{col}_delta"] = qc_values - raw_values
    return comparison.sort_values("frog_id").reset_index(drop=True)


def run_morphology_qc(
    filtered_df: pd.DataFrame,
    output_dir: Path,
    cfg: MorphologyQCConfig | Any | None = None,
) -> MorphologyQCResult:
    """Apply morphology QC and write all configured QC CSV outputs."""
    qc_cfg = cfg if isinstance(cfg, MorphologyQCConfig) else resolve_morphology_qc_config(cfg)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    annotated = apply_morphology_qc(filtered_df, qc_cfg)
    clean_df = annotated.loc[annotated["qc_pass"]].copy()
    rejected_df = annotated.loc[~annotated["qc_pass"]].copy()
    frog_aggregated_qc_df = build_frog_aggregated_metrics(clean_df)
    image_summary_df = _summary_by_group(annotated, "image_path")
    frog_summary_df = _summary_by_group(annotated, "frog_id")
    sensitivity_df = build_threshold_sensitivity(filtered_df, qc_cfg)
    comparison_df = build_frog_qc_comparison(filtered_df, clean_df)

    paths = {
        "filtered_areas_qc": output_dir / "filtered_areas_qc.csv",
        "filtered_areas_qc_rejected": output_dir / "filtered_areas_qc_rejected.csv",
        "frog_aggregated_metrics_qc": output_dir / "frog_aggregated_metrics_qc.csv",
        "morphology_qc_image_summary": output_dir / "morphology_qc_image_summary.csv",
        "morphology_qc_frog_summary": output_dir / "morphology_qc_frog_summary.csv",
        "morphology_qc_threshold_sensitivity": output_dir
        / "morphology_qc_threshold_sensitivity.csv",
        "frog_aggregated_metrics_qc_comparison": output_dir
        / "frog_aggregated_metrics_qc_comparison.csv",
    }

    clean_df.to_csv(paths["filtered_areas_qc"], index=False)
    rejected_df.to_csv(paths["filtered_areas_qc_rejected"], index=False)
    frog_aggregated_qc_df.to_csv(paths["frog_aggregated_metrics_qc"], index=False)
    image_summary_df.to_csv(paths["morphology_qc_image_summary"], index=False)
    frog_summary_df.to_csv(paths["morphology_qc_frog_summary"], index=False)
    sensitivity_df.to_csv(paths["morphology_qc_threshold_sensitivity"], index=False)
    comparison_df.to_csv(paths["frog_aggregated_metrics_qc_comparison"], index=False)

    logger.info(
        "Morphology QC complete: %d passed, %d rejected -> %s",
        len(clean_df),
        len(rejected_df),
        output_dir,
    )

    return MorphologyQCResult(
        clean_df=clean_df,
        rejected_df=rejected_df,
        frog_aggregated_qc_df=frog_aggregated_qc_df,
        image_summary_df=image_summary_df,
        frog_summary_df=frog_summary_df,
        sensitivity_df=sensitivity_df,
        comparison_df=comparison_df,
        paths=paths,
    )


def run_morphology_qc_from_csv(
    input_csv: Path,
    output_dir: Path | None = None,
    cfg: MorphologyQCConfig | Any | None = None,
) -> MorphologyQCResult:
    input_csv = Path(input_csv)
    out_dir = Path(output_dir) if output_dir is not None else input_csv.parent
    filtered_df = pd.read_csv(input_csv)
    return run_morphology_qc(filtered_df, out_dir, cfg)
