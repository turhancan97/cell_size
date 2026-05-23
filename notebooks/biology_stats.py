"""Load biology metrics for the LaTeX report from classify_output CSVs."""

from __future__ import annotations

from datetime import date

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_CSV = REPO_ROOT / "classify_output" / "predictions.csv"
FILTERED_AREAS_CSV = REPO_ROOT / "classify_output" / "filtered_areas.csv"
FROG_METRICS_CSV = REPO_ROOT / "classify_output" / "frog_aggregated_metrics.csv"
ANALYSIS_DIR = REPO_ROOT / "classify_output" / "analysis"
REGRESSION_DIR = REPO_ROOT / "classify_output" / "regression"
LOW_N_CELLS = 60


def icc_oneway(df: pd.DataFrame, group_col: str, value_col: str) -> dict[str, Any]:
    """ICC(1,1) — single measures, groups = frogs."""
    work = df[[group_col, value_col]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna()
    if work[group_col].nunique() < 2:
        return {"icc": float("nan"), "n": len(work), "n_groups": int(work[group_col].nunique())}

    groups = work.groupby(group_col)[value_col]
    counts = groups.count().to_numpy(dtype=float)
    means = groups.mean()
    grand = work[value_col].mean()
    k = len(counts)
    n = counts.sum()

    ss_between = float(((means - grand) ** 2 * groups.count()).sum())
    ss_within = float(groups.apply(lambda s: ((s - s.mean()) ** 2).sum()).sum())
    df_b, df_w = k - 1, n - k
    ms_b = ss_between / df_b if df_b > 0 else 0.0
    ms_w = ss_within / df_w if df_w > 0 else 0.0
    n_bar = float(counts.mean())
    denom = ms_b + (n_bar - 1) * ms_w
    icc = (ms_b - ms_w) / denom if denom > 0 else float("nan")
    within_cv = groups.apply(
        lambda s: s.std(ddof=1) / s.mean() * 100 if s.mean() else np.nan
    )
    return {
        "icc": float(icc),
        "n": int(n),
        "n_groups": int(k),
        "median_within_frog_cv_pct": float(within_cv.median()),
    }


def _fmt(x: float | int | None, ndigits: int = 2) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    if isinstance(x, int):
        return f"{x:,}"
    return f"{float(x):.{ndigits}f}"


def _series_summary(s: pd.Series) -> dict[str, float]:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return {}
    return {
        "n": int(len(s)),
        "median": float(s.median()),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)),
        "q25": float(s.quantile(0.25)),
        "q75": float(s.quantile(0.75)),
        "min": float(s.min()),
        "max": float(s.max()),
    }


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


_REGRESSION_LABELS = {
    "log(nucleus_area_um2) ~ log(area_um2)": "Nucleus area vs cell area",
    "log(nucleus_major_axis_um) ~ log(major_axis_um)": "Nucleus long axis vs cell long axis",
    "log(nucleus_minor_axis_um) ~ log(minor_axis_um)": "Nucleus short axis vs cell short axis",
}

_NC_MODEL_LABELS = {
    "random intercept": "Random intercept per frog",
    "random intercept + slope": "Random intercept + slope per frog",
}


def _short_regression_label(name: str) -> str:
    return _REGRESSION_LABELS.get(str(name), str(name))


def _short_nc_model_label(name: str) -> str:
    return _NC_MODEL_LABELS.get(str(name), str(name))


def load_biology_stats(
    *,
    predictions_path: Path | None = None,
    areas_path: Path | None = None,
    frog_path: Path | None = None,
) -> dict[str, Any]:
    """Return flat string placeholders and nested data for report_biology.md."""
    pred_path = predictions_path or PREDICTIONS_CSV
    area_path = areas_path or FILTERED_AREAS_CSV
    frog_metrics_path = frog_path or FROG_METRICS_CSV

    pred_df = pd.read_csv(pred_path)
    area_df = pd.read_csv(area_path)
    frog_df = pd.read_csv(frog_metrics_path) if frog_metrics_path.is_file() else pd.DataFrame()

    stats: dict[str, Any] = {}

    # Cohort counts
    n_scored = len(pred_df)
    n_good = int((pred_df["predicted_verdict"] == "good").sum()) if "predicted_verdict" in pred_df.columns else len(area_df)
    n_bad = int((pred_df["predicted_verdict"] == "bad").sum()) if "predicted_verdict" in pred_df.columns else 0
    n_rejected = int((pred_df["predicted_verdict"] == "rejected").sum()) if "predicted_verdict" in pred_df.columns else 0
    n_frogs = int(area_df["frog_id"].nunique()) if "frog_id" in area_df.columns else 0
    n_images = int(area_df["image_path"].nunique()) if "image_path" in area_df.columns else 0
    n_nucleus = int(pd.to_numeric(area_df.get("nucleus_area_um2", pd.Series(dtype=float)), errors="coerce").notna().sum())

    stats["n_scored"] = n_scored
    stats["n_good_cells"] = n_good
    stats["n_bad_cells"] = n_bad
    stats["n_rejected_cells"] = n_rejected
    stats["n_frogs"] = n_frogs
    stats["n_images"] = n_images
    stats["n_nucleus_matched"] = n_nucleus
    stats["good_fraction_global"] = n_good / max(n_scored, 1)

    # Yield per frog
    if "predicted_verdict" in pred_df.columns and "frog_id" in pred_df.columns:
        yield_frog = (
            pred_df.groupby("frog_id", as_index=False)
            .agg(n_total=("mask_index", "count"), n_good=("predicted_verdict", lambda s: (s == "good").sum()))
            .assign(good_fraction=lambda d: d["n_good"] / d["n_total"].clip(lower=1))
        )
        stats["yield_median_pct"] = float(yield_frog["good_fraction"].median() * 100)
        stats["yield_min_pct"] = float(yield_frog["good_fraction"].min() * 100)
        stats["yield_max_pct"] = float(yield_frog["good_fraction"].max() * 100)
    else:
        stats["yield_median_pct"] = stats["yield_min_pct"] = stats["yield_max_pct"] = float("nan")

    if "image_path" in pred_df.columns and "predicted_verdict" in pred_df.columns:
        yield_img = (
            pred_df.groupby("image_path", as_index=False)
            .agg(n_total=("mask_index", "count"), n_good=("predicted_verdict", lambda s: (s == "good").sum()))
            .assign(good_fraction=lambda d: d["n_good"] / d["n_total"].clip(lower=1))
        )
        stats["yield_image_median_pct"] = float(yield_img["good_fraction"].median() * 100)
        stats["n_images_zero_good"] = int((yield_img["n_good"] == 0).sum())
    else:
        stats["yield_image_median_pct"] = float("nan")
        stats["n_images_zero_good"] = 0

    # Cell area / N/C summaries
    area_sum = _series_summary(area_df["area_um2"]) if "area_um2" in area_df.columns else {}
    nc_sum = _series_summary(area_df["nc_ratio"]) if "nc_ratio" in area_df.columns else {}
    equiv_med = 2 * np.sqrt(area_sum.get("median", np.nan) / np.pi) if area_sum else float("nan")

    stats["area_median_um2"] = area_sum.get("median", float("nan"))
    stats["area_mean_um2"] = area_sum.get("mean", float("nan"))
    stats["area_std_um2"] = area_sum.get("std", float("nan"))
    stats["area_iqr_lo"] = area_sum.get("q25", float("nan"))
    stats["area_iqr_hi"] = area_sum.get("q75", float("nan"))
    stats["equiv_diameter_median_um"] = float(equiv_med)
    stats["nc_median"] = nc_sum.get("median", float("nan"))
    stats["nc_mean"] = nc_sum.get("mean", float("nan"))
    stats["nc_std"] = nc_sum.get("std", float("nan"))

    major_sum = _series_summary(area_df["major_axis_um"]) if "major_axis_um" in area_df.columns else {}
    minor_sum = _series_summary(area_df["minor_axis_um"]) if "minor_axis_um" in area_df.columns else {}
    axis_ratio_sum = _series_summary(area_df["cell_axis_ratio"]) if "cell_axis_ratio" in area_df.columns else {}
    stats["major_median_um"] = major_sum.get("median", float("nan"))
    stats["major_iqr_lo"] = major_sum.get("q25", float("nan"))
    stats["major_iqr_hi"] = major_sum.get("q75", float("nan"))
    stats["minor_median_um"] = minor_sum.get("median", float("nan"))
    stats["minor_iqr_lo"] = minor_sum.get("q25", float("nan"))
    stats["minor_iqr_hi"] = minor_sum.get("q75", float("nan"))
    stats["cell_axis_ratio_median"] = axis_ratio_sum.get("median", float("nan"))

    nuc_area_sum = _series_summary(area_df["nucleus_area_um2"]) if "nucleus_area_um2" in area_df.columns else {}
    nuc_major_sum = _series_summary(area_df["nucleus_major_axis_um"]) if "nucleus_major_axis_um" in area_df.columns else {}
    nuc_minor_sum = _series_summary(area_df["nucleus_minor_axis_um"]) if "nucleus_minor_axis_um" in area_df.columns else {}
    stats["nucleus_area_median_um2"] = nuc_area_sum.get("median", float("nan"))
    stats["nucleus_area_iqr_lo"] = nuc_area_sum.get("q25", float("nan"))
    stats["nucleus_area_iqr_hi"] = nuc_area_sum.get("q75", float("nan"))
    stats["nucleus_major_median_um"] = nuc_major_sum.get("median", float("nan"))
    stats["nucleus_minor_median_um"] = nuc_minor_sum.get("median", float("nan"))

    # Per-frog mean area range
    if not frog_df.empty and "area_um2_mean" in frog_df.columns:
        fm = pd.to_numeric(frog_df["area_um2_mean"], errors="coerce").dropna()
        stats["frog_mean_area_min"] = float(fm.min())
        stats["frog_mean_area_max"] = float(fm.max())
        stats["frog_mean_area_median"] = float(fm.median())
    else:
        stats["frog_mean_area_min"] = stats["frog_mean_area_max"] = stats["frog_mean_area_median"] = float("nan")

    # ICC
    icc = icc_oneway(area_df, "frog_id", "area_um2") if "frog_id" in area_df.columns else {}
    stats["icc_area"] = icc.get("icc", float("nan"))
    stats["median_within_frog_cv_pct"] = icc.get("median_within_frog_cv_pct", float("nan"))

    # Reference intervals table
    ref_path = ANALYSIS_DIR / "reference_intervals.csv"
    if ref_path.is_file():
        ref_df = pd.read_csv(ref_path)
        ref_rows = []
        for _, r in ref_df.iterrows():
            ref_rows.append([
                r.get("label", r.get("metric", "")),
                _fmt(r.get("p2.5"), 2),
                _fmt(r.get("p50"), 2),
                _fmt(r.get("p97.5"), 2),
            ])
        stats["reference_intervals_table"] = _md_table(
            ["Metric", "2.5th %ile", "Median", "97.5th %ile"], ref_rows
        )
        ref_by_metric = ref_df.set_index("metric") if "metric" in ref_df.columns else pd.DataFrame()

        def _ref_val(metric: str, col: str) -> float:
            if ref_by_metric.empty or metric not in ref_by_metric.index:
                return float("nan")
            return float(ref_by_metric.loc[metric, col])

        stats["ref_area_p2_5"] = _ref_val("area_um2", "p2.5")
        stats["ref_area_p50"] = _ref_val("area_um2", "p50")
        stats["ref_area_p97_5"] = _ref_val("area_um2", "p97.5")
        stats["ref_nucleus_p2_5"] = _ref_val("nucleus_area_um2", "p2.5")
        stats["ref_nucleus_p50"] = _ref_val("nucleus_area_um2", "p50")
        stats["ref_nucleus_p97_5"] = _ref_val("nucleus_area_um2", "p97.5")
        stats["ref_nc_p2_5"] = _ref_val("nc_ratio", "p2.5")
        stats["ref_nc_p50"] = _ref_val("nc_ratio", "p50")
        stats["ref_nc_p97_5"] = _ref_val("nc_ratio", "p97.5")
    else:
        stats["reference_intervals_table"] = "_Reference intervals CSV not found._"
        for key in (
            "ref_area_p2_5", "ref_area_p50", "ref_area_p97_5",
            "ref_nucleus_p2_5", "ref_nucleus_p50", "ref_nucleus_p97_5",
            "ref_nc_p2_5", "ref_nc_p50", "ref_nc_p97_5",
        ):
            stats[key] = float("nan")

    # Regression summary
    reg_path = REGRESSION_DIR / "regression_summary.csv"
    if reg_path.is_file():
        reg_df = pd.read_csv(reg_path)
    else:
        reg_df = _compute_regression_summary(area_df)

    reg_rows = []
    for _, r in reg_df.iterrows():
        reg_rows.append([
            _short_regression_label(r.get("regression", "")),
            _fmt(r.get("n_cells"), 0),
            _fmt(r.get("ols_slope"), 3),
            f"[{_fmt(r.get('ols_slope_ci_lo'), 3)}, {_fmt(r.get('ols_slope_ci_hi'), 3)}]",
            _fmt(r.get("mixedlm_slope"), 3),
            _fmt(r.get("ols_r2"), 3),
        ])
    stats["regression_table"] = _md_table(
        ["Comparison", "n", "OLS slope", "OLS 95% CI", "Mixed slope", "R$^2$"],
        reg_rows,
    )
    if not reg_df.empty:
        slope = float(reg_df.iloc[0].get("ols_slope", np.nan))
        stats["scaling_interpretation"] = _interpret_slope(slope)

    # N/C mixed models
    nc_path = ANALYSIS_DIR / "nc_ratio_mixed_models.csv"
    if nc_path.is_file():
        nc_df = pd.read_csv(nc_path)
    else:
        nc_df = _compute_nc_mixed_models(area_df)

    nc_rows = []
    for _, r in nc_df.iterrows():
        nc_rows.append([
            _short_nc_model_label(r.get("model", "")),
            _fmt(r.get("slope_log_area"), 4),
            f"[{_fmt(r.get('slope_ci_lo'), 4)}, {_fmt(r.get('slope_ci_hi'), 4)}]",
            _fmt(r.get("n_cells"), 0),
            _fmt(r.get("n_frogs"), 0),
        ])
    stats["nc_mixed_table"] = _md_table(
        ["Model", "Slope (log cell area)", "95% CI", "n", "Frogs"],
        nc_rows,
    )

    # Frog snapshot tables
    frog_report_path = ANALYSIS_DIR / "frog_summary_report.csv"
    if frog_report_path.is_file():
        fr = pd.read_csv(frog_report_path)
    elif not frog_df.empty:
        fr = frog_df.copy()
    else:
        fr = pd.DataFrame()

    if not fr.empty and "area_um2_mean" in fr.columns:
        fr = fr.sort_values("area_um2_mean")
        snap_cols = ["frog_id", "n_cells", "area_um2_mean", "nc_ratio_mean", "good_fraction"]
        snap_cols = [c for c in snap_cols if c in fr.columns]
        small = fr.head(10)[snap_cols]
        large = fr.tail(10)[snap_cols]
        stats["frog_smallest_table"] = _df_to_md(small.round(2))
        stats["frog_largest_table"] = _df_to_md(large.round(2))
    else:
        stats["frog_smallest_table"] = stats["frog_largest_table"] = "_Frog summary not available._"

    # Flat string placeholders for {{key}} substitution
    skip_keys = {"reference_intervals_table", "regression_table", "nc_mixed_table",
                 "frog_smallest_table", "frog_largest_table", "scaling_interpretation", "placeholders"}
    placeholders: dict[str, str] = {}
    int_keys = {"n_scored", "n_good_cells", "n_bad_cells", "n_rejected_cells",
                "n_frogs", "n_images", "n_nucleus_matched", "n_images_zero_good"}
    for k, v in stats.items():
        if k in skip_keys:
            continue
        if k in int_keys:
            placeholders[k] = f"{int(v):,}"
        elif isinstance(v, float):
            placeholders[k] = _fmt(v)
        else:
            placeholders[k] = str(v)
    placeholders["good_fraction_pct"] = _fmt(stats["good_fraction_global"] * 100, 1)
    placeholders["yield_median_pct"] = _fmt(stats.get("yield_median_pct"), 1)
    placeholders["yield_min_pct"] = _fmt(stats.get("yield_min_pct"), 1)
    placeholders["yield_max_pct"] = _fmt(stats.get("yield_max_pct"), 1)
    placeholders["yield_image_median_pct"] = _fmt(stats.get("yield_image_median_pct"), 1)
    placeholders["n_images_zero_good"] = str(stats.get("n_images_zero_good", 0))
    placeholders["icc_area"] = _fmt(stats.get("icc_area"), 3)
    placeholders["median_within_frog_cv_pct"] = _fmt(stats.get("median_within_frog_cv_pct"), 1)
    placeholders["scaling_interpretation"] = stats.get("scaling_interpretation", "")
    placeholders["reference_intervals_table"] = stats["reference_intervals_table"]
    placeholders["regression_table"] = stats["regression_table"]
    placeholders["nc_mixed_table"] = stats["nc_mixed_table"]
    placeholders["frog_smallest_table"] = stats["frog_smallest_table"]
    placeholders["frog_largest_table"] = stats["frog_largest_table"]
    placeholders["report_date"] = date.today().strftime("%d %B %Y")
    nc_med = stats.get("nc_median")
    placeholders["nc_median_pct"] = (
        _fmt(float(nc_med) * 100, 0) if nc_med is not None and np.isfinite(float(nc_med)) else "—"
    )

    stats["placeholders"] = placeholders
    return stats


def _df_to_md(df: pd.DataFrame) -> str:
    headers = [str(c) for c in df.columns]
    rows = [[str(v) for v in row] for row in df.to_numpy()]
    return _md_table(headers, rows)


def _interpret_slope(slope: float) -> str:
    if not np.isfinite(slope):
        return ""
    if slope > 1.0:
        return "positively allometric (nucleus grows faster than cell area)"
    if slope < 1.0:
        return "negatively allometric (nucleus grows slower than cell area; larger cells are relatively less nucleus-dense)"
    return "consistent with isometric scaling (slope $\\approx$ 1)"


def _compute_regression_summary(area_df: pd.DataFrame) -> pd.DataFrame:
    try:
        import statsmodels.api as sm
    except ImportError:
        return pd.DataFrame()

    targets = [
        ("area_um2", "nucleus_area_um2", "log(nucleus_area_um2) ~ log(area_um2)"),
        ("major_axis_um", "nucleus_major_axis_um", "log(nucleus_major_axis_um) ~ log(major_axis_um)"),
        ("minor_axis_um", "nucleus_minor_axis_um", "log(nucleus_minor_axis_um) ~ log(minor_axis_um)"),
    ]
    rows = []
    for x_col, y_col, label in targets:
        if x_col not in area_df.columns or y_col not in area_df.columns:
            continue
        work = area_df[[x_col, y_col, "frog_id"]].dropna()
        work = work[(work[x_col] > 0) & (work[y_col] > 0)]
        if len(work) < 10:
            continue
        x = np.log(work[x_col].astype(float))
        y = np.log(work[y_col].astype(float))
        ols = sm.OLS(y, sm.add_constant(x)).fit()
        ci = ols.conf_int()
        mixed_slope = np.nan
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                work = work.assign(log_x=np.log(work[x_col].astype(float)), log_y=np.log(work[y_col].astype(float)))
                mres = sm.MixedLM.from_formula("log_y ~ log_x", groups="frog_id", data=work).fit(
                    method="lbfgs", reml=True
                )
                mixed_slope = float(mres.fe_params["log_x"])
        except Exception:
            pass
        rows.append({
            "regression": label,
            "n_cells": len(work),
            "n_frogs": int(work["frog_id"].nunique()),
            "ols_slope": float(ols.params[1]),
            "ols_slope_ci_lo": float(ci.iloc[1, 0]),
            "ols_slope_ci_hi": float(ci.iloc[1, 1]),
            "mixedlm_slope": mixed_slope,
            "ols_r2": float(ols.rsquared),
        })
    return pd.DataFrame(rows)


def _compute_nc_mixed_models(area_df: pd.DataFrame) -> pd.DataFrame:
    try:
        import statsmodels.api as sm
    except ImportError:
        return pd.DataFrame()

    if not {"area_um2", "nc_ratio", "frog_id"}.issubset(area_df.columns):
        return pd.DataFrame()

    work = area_df[["area_um2", "nc_ratio", "frog_id"]].dropna()
    work = work[(work["area_um2"] > 0) & (work["nc_ratio"] > 0)].copy()
    work["log_area"] = np.log(work["area_um2"].astype(float))
    work["log_nc"] = np.log(work["nc_ratio"].astype(float))
    rows = []
    for label, re_formula in [("random intercept", "1"), ("random intercept + slope", "~log_area")]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mres = sm.MixedLM.from_formula(
                    "log_nc ~ log_area", data=work, groups="frog_id", re_formula=re_formula
                ).fit(method="lbfgs", reml=True, maxiter=200)
            ci = mres.conf_int().loc["log_area"]
            rows.append({
                "model": label,
                "slope_log_area": float(mres.fe_params["log_area"]),
                "slope_ci_lo": float(ci[0]),
                "slope_ci_hi": float(ci[1]),
                "n_cells": len(work),
                "n_frogs": int(work["frog_id"].nunique()),
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


def fill_template(template_text: str, placeholders: dict[str, str]) -> str:
    out = template_text
    for key, val in placeholders.items():
        out = out.replace("{{" + key + "}}", str(val))
    return out


def write_stats_cache(path: Path | None = None) -> Path:
    path = path or (ANALYSIS_DIR / "biology_stats.json")
    stats = load_biology_stats()
    cache = {k: v for k, v in stats.items() if k != "placeholders"}
    cache["placeholders"] = stats["placeholders"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, default=str), encoding="utf-8")
    return path


if __name__ == "__main__":
    s = load_biology_stats()
    p = s["placeholders"]
    print("n_good_cells:", p["n_good_cells"])
    print("area_median_um2:", p.get("area_median_um2"))
    print("icc_area:", p.get("icc_area"))
