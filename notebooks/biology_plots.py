"""Biology report figures — shared by make_report_figures.py and build_report_latex.py."""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = REPO_ROOT / "notebooks" / "figures"
FILTERED_AREAS_CSV = REPO_ROOT / "classify_output" / "filtered_areas.csv"
FROG_METRICS_CSV = REPO_ROOT / "classify_output" / "frog_aggregated_metrics.csv"
PREDICTIONS_CSV = REPO_ROOT / "classify_output" / "predictions.csv"

BIO_COLOR_CELL = "#2a9d8f"
BIO_COLOR_NUCLEUS = "#264653"
BIO_COLOR_HIGHLIGHT = "#e76f51"
BIO_COLOR_HIGHLIGHT_LARGE = "#1d4ed8"

RANDOM_SEED = 42
TOP_N_FROGS_FOR_COLOR = 40
FIG_DPI = 140


def hist_with_summary(
    ax: plt.Axes,
    values,
    color: str,
    title: str,
    xlabel: str,
    bins: int = 40,
    xlim_percentiles: tuple[float, float] | None = None,
) -> None:
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy()
    plot_values = values
    hist_range = None

    if len(values) and xlim_percentiles is not None:
        lo_p, hi_p = xlim_percentiles
        lo, hi = np.quantile(values, [lo_p / 100.0, hi_p / 100.0])
        span = hi - lo
        pad = span * 0.05 if span > 0 else 0.5
        lo, hi = lo - pad, hi + pad
        hist_range = (lo, hi)
        plot_values = values[(values >= lo) & (values <= hi)]

    if len(plot_values):
        ax.hist(
            plot_values, bins=bins, range=hist_range, color=color,
            alpha=0.85, edgecolor="white", linewidth=0.5,
        )
        if hist_range is not None:
            ax.set_xlim(hist_range)

    if len(values):
        med = float(np.median(values))
        q1, q3 = np.quantile(values, [0.25, 0.75])
        ax.axvline(med, color="#222", linestyle="--", linewidth=1.3, label=f"median={med:.2f}")
        ax.axvspan(q1, q3, color="#222", alpha=0.08, label=f"IQR {q1:.2f}–{q3:.2f}")
        handles, labels = ax.get_legend_handles_labels()
        if xlim_percentiles is not None:
            n_out = len(values) - len(plot_values)
            if n_out:
                handles.append(Line2D([0], [0], linestyle="none", marker="none", color="none"))
                labels.append(f"{n_out} outside {xlim_percentiles[0]:.0f}–{xlim_percentiles[1]:.0f}%")
        ax.legend(handles, labels, loc="upper right", fontsize=9, framealpha=0.92)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of cells")


def save_area_distribution(area_df: pd.DataFrame, figures_dir: Path | None = None) -> Path:
    out_dir = figures_dir or FIGURES_DIR
    fig, ax = plt.subplots(figsize=(7, 4.6))
    hist_with_summary(ax, area_df["area_um2"], BIO_COLOR_CELL, "Cell area (µm²)", "area_um2")
    fig.suptitle("Cell area distribution (good cells only)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = out_dir / "AreaDistribution.png"
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {path.relative_to(REPO_ROOT)}")
    return path


def save_diameter_distribution(area_df: pd.DataFrame, figures_dir: Path | None = None) -> Path:
    out_dir = figures_dir or FIGURES_DIR
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    hist_with_summary(axes[0], area_df["major_axis_um"], BIO_COLOR_CELL, "Cell long diameter (µm)", "major_axis_um")
    hist_with_summary(axes[1], area_df["minor_axis_um"], BIO_COLOR_CELL, "Cell short diameter (µm)", "minor_axis_um")
    fig.suptitle("Cell diameter distribution (good cells only)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = out_dir / "DiameterDistribution.png"
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {path.relative_to(REPO_ROOT)}")
    return path


def save_nucleus_distribution(area_df: pd.DataFrame, figures_dir: Path | None = None) -> Path:
    out_dir = figures_dir or FIGURES_DIR
    zoom = (1, 99)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, col, title in zip(
        axes,
        ["nucleus_area_um2", "nucleus_major_axis_um", "nucleus_minor_axis_um"],
        ["Nucleus area (µm²)", "Nucleus long diameter (µm)", "Nucleus short diameter (µm)"],
    ):
        hist_with_summary(ax, area_df[col], BIO_COLOR_NUCLEUS, title, col, bins=50, xlim_percentiles=zoom)
    fig.suptitle("Nucleus size (x-axis zoomed to 1st–99th percentile)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = out_dir / "NucleusDistribution.png"
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {path.relative_to(REPO_ROOT)}")
    return path


def save_nc_ratio_distribution(area_df: pd.DataFrame, figures_dir: Path | None = None) -> Path:
    out_dir = figures_dir or FIGURES_DIR
    fig, ax = plt.subplots(figsize=(9, 4.8))
    hist_with_summary(
        ax, area_df["nc_ratio"], BIO_COLOR_NUCLEUS,
        "Nucleus-to-cell area ratio (N/C)", "nc_ratio", bins=50, xlim_percentiles=(1, 99),
    )
    fig.suptitle("N/C ratio (x-axis zoomed to 1st–99th percentile)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = out_dir / "NCRatioDistribution.png"
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {path.relative_to(REPO_ROOT)}")
    return path


def _top_frogs_by_count(area_df: pd.DataFrame, k: int = 40) -> list:
    return area_df.groupby("frog_id").size().sort_values(ascending=False).head(k).index.tolist()


def _per_frog_boxplot(
    area_df: pd.DataFrame, value_col: str, frogs_sorted: list, color: str,
    ylabel: str, title: str, output_name: str, figures_dir: Path,
) -> None:
    data = [area_df.loc[area_df["frog_id"] == fid, value_col].dropna().to_numpy() for fid in frogs_sorted]
    sample_counts = [len(d) for d in data]
    fig, ax = plt.subplots(figsize=(14, 5.5))
    bp = ax.boxplot(
        data, positions=np.arange(len(frogs_sorted)), widths=0.6,
        patch_artist=True, showfliers=False,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        patch.set_edgecolor("#0f172a")
    for med in bp["medians"]:
        med.set_color("#0f172a")
        med.set_linewidth(1.5)
    ax.set_xticks(np.arange(len(frogs_sorted)))
    ax.set_xticklabels(
        [f"{fid}\n(n={n})" for fid, n in zip(frogs_sorted, sample_counts)],
        rotation=60, ha="right", fontsize=7,
    )
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = figures_dir / output_name
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {path.relative_to(REPO_ROOT)}")


def save_per_frog_boxplot(area_df: pd.DataFrame, figures_dir: Path | None = None) -> None:
    out_dir = figures_dir or FIGURES_DIR
    if "frog_id" not in area_df.columns:
        print("[warn] no frog_id; skipping per-frog boxplots")
        return
    top_frogs = _top_frogs_by_count(area_df, k=40)
    sub = area_df[area_df["frog_id"].isin(top_frogs)]
    frogs_sorted = sub.groupby("frog_id")["area_um2"].median().sort_values().index.tolist()
    _per_frog_boxplot(
        area_df, "area_um2", frogs_sorted, BIO_COLOR_CELL,
        "Cell area (µm²)",
        f"Per-frog cell area (top-{len(frogs_sorted)} frogs by count)",
        "PerFrogBoxplot.png", out_dir,
    )
    if "nucleus_area_um2" in area_df.columns:
        _per_frog_boxplot(
            area_df, "nucleus_area_um2", frogs_sorted, BIO_COLOR_NUCLEUS,
            "Nucleus area (µm²)",
            f"Per-frog nucleus area (same {len(frogs_sorted)} frogs)",
            "PerFrogNucleusBoxplot.png", out_dir,
        )


def save_yield_by_frog(pred_df: pd.DataFrame, frog_df: pd.DataFrame, figures_dir: Path | None = None) -> Path:
    out_dir = figures_dir or FIGURES_DIR
    yield_frog = (
        pred_df.groupby("frog_id", as_index=False)
        .agg(
            n_total=("mask_index", "count"),
            n_good=("predicted_verdict", lambda s: (s == "good").sum()),
            n_bad=("predicted_verdict", lambda s: (s == "bad").sum()),
            n_rejected=("predicted_verdict", lambda s: (s == "rejected").sum()),
        )
        .assign(good_fraction=lambda d: d["n_good"] / d["n_total"].clip(lower=1))
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    yf = yield_frog.sort_values("frog_id").head(80)
    bottom = np.zeros(len(yf))
    x = np.arange(len(yf))
    for verdict, col, color in [
        ("good", "n_good", "#009E73"), ("bad", "n_bad", "#D55E00"), ("rejected", "n_rejected", "#CC79A7"),
    ]:
        ax = axes[0]
        vals = yf[col].to_numpy(dtype=float)
        ax.bar(x, vals, bottom=bottom, color=color, label=verdict, width=1.0)
        bottom += vals
    axes[0].set_xlim(-0.5, len(yf) - 0.5)
    axes[0].set_xticks([])
    axes[0].set_ylabel("Cell count")
    axes[0].set_title(f"Stacked verdicts (first {len(yf)} frogs)")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].scatter(yield_frog["n_good"], yield_frog["good_fraction"], s=22, alpha=0.55, c="#4C78A8", edgecolors="none")
    if not frog_df.empty and "area_um2_mean" in frog_df.columns:
        y2 = yield_frog.merge(frog_df[["frog_id", "area_um2_mean"]], on="frog_id", how="left")
        r = y2["good_fraction"].corr(y2["area_um2_mean"])
        axes[1].text(0.03, 0.97, f"corr(good_fraction, mean area) = {r:.3f}", transform=axes[1].transAxes, va="top", fontsize=9)
    axes[1].set_xlabel("Good cells per frog")
    axes[1].set_ylabel("Good fraction")
    axes[1].set_title("Classifier yield vs. good-cell count")
    axes[1].grid(alpha=0.2)
    fig.suptitle("Classification yield per frog", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = out_dir / "YieldByFrog.png"
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {path.relative_to(REPO_ROOT)}")
    return path


def save_icc_variation(area_df: pd.DataFrame, figures_dir: Path | None = None) -> Path:
    out_dir = figures_dir or FIGURES_DIR
    within_cv = (
        area_df.groupby("frog_id")["area_um2"]
        .apply(lambda s: pd.to_numeric(s, errors="coerce").std(ddof=1) / pd.to_numeric(s, errors="coerce").mean() * 100)
        .dropna()
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].hist(within_cv, bins=35, color=BIO_COLOR_CELL, alpha=0.85, edgecolor="white")
    axes[0].axvline(within_cv.median(), color="#222", ls="--", label=f"median CV = {within_cv.median():.1f}%")
    axes[0].set_xlabel("Within-frog CV% (area_um2)")
    axes[0].set_ylabel("Number of frogs")
    axes[0].set_title("Cell-to-cell spread within each frog")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.2)

    if "image_path" in area_df.columns:
        img_means = area_df.groupby(["frog_id", "image_path"])["area_um2"].mean().reset_index()
        frogs_multi = img_means.groupby("frog_id").filter(lambda g: len(g) >= 3)["frog_id"].unique()[:12]
        data = [img_means.loc[img_means["frog_id"] == fid, "area_um2"].to_numpy() for fid in frogs_multi]
        axes[1].boxplot(data, tick_labels=[str(int(f)) for f in frogs_multi], vert=True)
        axes[1].set_xlabel("frog_id (≥3 images)")
        axes[1].set_ylabel("Mean cell area per image (µm²)")
        axes[1].set_title("Image-level replicate variation")
        axes[1].tick_params(axis="x", rotation=45, labelsize=8)
        axes[1].grid(axis="y", alpha=0.2)

    fig.suptitle("Within- vs between-frog variation", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = out_dir / "ICCVariation.png"
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {path.relative_to(REPO_ROOT)}")
    return path


def _annotate_outlier_frog_ids(
    ax: plt.Axes,
    rows: pd.DataFrame,
    *,
    side: str,
) -> None:
    """Label outlier frog IDs with staggered offsets so labels do not overlap."""
    if side == "small":
        # Left cluster: alternate labels to the left/right with vertical stagger.
        offsets = [(-24, 18), (12, 16), (-24, 0), (12, -14), (-24, -18)]
        ha_cycle = ["right", "left", "right", "left", "right"]
    else:
        # Right cluster: mirror pattern so labels fan outward from the group.
        offsets = [(24, 16), (-12, 14), (24, 0), (-12, -12), (24, -16)]
        ha_cycle = ["left", "right", "left", "right", "left"]

    for i, (_, row) in enumerate(rows.iterrows()):
        ox, oy = offsets[i % len(offsets)]
        ax.annotate(
            str(int(row["frog_id"])),
            (row["rank"], row["area_um2_mean"]),
            textcoords="offset points",
            xytext=(ox, oy),
            fontsize=8,
            fontweight="bold",
            color="#1f2937",
            ha=ha_cycle[i % len(ha_cycle)],
            va="center",
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#94a3b8", "alpha": 0.92, "lw": 0.6},
            arrowprops={"arrowstyle": "-", "color": "#64748b", "lw": 0.6, "shrinkA": 0, "shrinkB": 2},
            zorder=4,
        )


def save_outlier_frogs(frog_df: pd.DataFrame, area_df: pd.DataFrame, figures_dir: Path | None = None) -> Path:
    out_dir = figures_dir or FIGURES_DIR
    df_ranked = frog_df.dropna(subset=["area_um2_mean"]).copy().sort_values("area_um2_mean")
    df_ranked = df_ranked.reset_index(drop=True)
    df_ranked["rank"] = df_ranked.index + 1
    small = df_ranked.head(5)["frog_id"].tolist()
    large = df_ranked.tail(5)["frog_id"].tolist()
    overall_median = df_ranked["area_um2_mean"].median()
    n_frogs = len(df_ranked)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
    ax1.scatter(df_ranked["rank"], df_ranked["area_um2_mean"], s=16, c="#cbd5e1", alpha=0.7, edgecolors="none",
                label=f"all frogs (n={n_frogs})")
    for mask, color, label in [
        (df_ranked["frog_id"].isin(small), BIO_COLOR_HIGHLIGHT, "smallest 5"),
        (df_ranked["frog_id"].isin(large), BIO_COLOR_HIGHLIGHT_LARGE, "largest 5"),
    ]:
        ax1.scatter(df_ranked.loc[mask, "rank"], df_ranked.loc[mask, "area_um2_mean"],
                    s=48, c=color, edgecolors="white", linewidths=0.6, zorder=3, label=label)
    outliers = df_ranked[df_ranked["frog_id"].isin(small + large)]
    _annotate_outlier_frog_ids(ax1, outliers[outliers["frog_id"].isin(small)], side="small")
    _annotate_outlier_frog_ids(ax1, outliers[outliers["frog_id"].isin(large)], side="large")
    ax1.axhline(overall_median, color="#1a4f49", linestyle="--", linewidth=1.2,
                label=f"median = {overall_median:.1f} µm²")
    ax1.set_xlabel("Rank by mean cell area (1 = smallest)")
    ax1.set_ylabel("Mean cell area per frog (µm²)")
    ax1.set_title("Per-frog ranking")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(alpha=0.2)

    bins = np.linspace(float(area_df["area_um2"].quantile(0.01)), float(area_df["area_um2"].quantile(0.99)), 60)
    ax2.hist(area_df["area_um2"].dropna(), bins=bins, color="#cbd5e1", alpha=0.6, label="all good cells")
    for fid in small + large:
        color = BIO_COLOR_HIGHLIGHT if fid in small else BIO_COLOR_HIGHLIGHT_LARGE
        sub = area_df.loc[area_df["frog_id"] == fid, "area_um2"].dropna()
        if len(sub):
            ax2.hist(sub, bins=bins, alpha=0.55, color=color, label=f"frog {fid} (n={len(sub)})")
    ax2.set_xlabel("Cell area (µm²)")
    ax2.set_ylabel("Number of cells")
    ax2.set_title("Outlier frogs vs. cohort")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(axis="y", alpha=0.2)
    fig.suptitle("Outlier frogs by mean cell area", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = out_dir / "OutlierFrogs.png"
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {path.relative_to(REPO_ROOT)}")
    return path


def _ols_fit_log(area_df: pd.DataFrame, x_col: str, y_col: str) -> dict:
    import statsmodels.api as sm

    work = area_df[[x_col, y_col, "frog_id"]].dropna()
    work = work[(work[x_col] > 0) & (work[y_col] > 0)]
    if len(work) < 10:
        return {"ok": False}
    x = np.log(work[x_col].astype(float))
    y = np.log(work[y_col].astype(float))
    model = sm.OLS(y, sm.add_constant(x)).fit()
    work = work.assign(log_x=x.to_numpy(), log_y=y.to_numpy())
    mixed = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mres = sm.MixedLM.from_formula("log_y ~ log_x", groups="frog_id", data=work).fit(
                method="lbfgs", reml=True
            )
        mixed = {"slope": float(mres.fe_params["log_x"]), "intercept": float(mres.fe_params["Intercept"])}
    except Exception:
        pass
    return {
        "ok": True, "data": work, "x_col": x_col, "y_col": y_col,
        "slope": float(model.params.iloc[1]), "intercept": float(model.params.iloc[0]),
        "r2": float(model.rsquared), "n": len(work), "n_frogs": int(work["frog_id"].nunique()),
        "mixed": mixed,
    }


def _plot_regression_scatter(fit: dict, mixed: dict | None, label: str, ax) -> None:
    df = fit["data"]
    counts = df.groupby("frog_id").size().sort_values(ascending=False)
    top_frogs = counts.head(TOP_N_FROGS_FOR_COLOR).index.tolist()
    cmap = plt.colormaps.get_cmap("tab20")
    other = df.loc[~df["frog_id"].isin(top_frogs)]
    ax.scatter(other["log_x"], other["log_y"], s=10, c="#cbd5e1", alpha=0.55, edgecolors="none")
    for i, fid in enumerate(top_frogs):
        sub = df.loc[df["frog_id"] == fid]
        ax.scatter(sub["log_x"], sub["log_y"], s=14, color=cmap(i % cmap.N), alpha=0.85, edgecolors="none")
    xx = np.linspace(df["log_x"].min(), df["log_x"].max(), 100)
    ax.plot(xx, fit["intercept"] + fit["slope"] * xx, color="#0f172a", lw=2.2,
            label=f"OLS slope={fit['slope']:.3f} R²={fit['r2']:.3f}")
    if mixed:
        ax.plot(xx, mixed["intercept"] + mixed["slope"] * xx, color="#e76f51", lw=2.0, ls="--",
                label=f"mixed slope={mixed['slope']:.3f}")
    ax.set_xlabel(f"log({fit['x_col']})")
    ax.set_ylabel(f"log({fit['y_col']})")
    ax.set_title(f"{label}\n(n={fit['n']} cells, {fit['n_frogs']} frogs)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.25)


def save_regression_figures(area_df: pd.DataFrame, figures_dir: Path | None = None) -> None:
    out_dir = figures_dir or FIGURES_DIR
    targets = [
        ("area_um2", "nucleus_area_um2", "Nucleus area vs cell area", "NucleusVsCellRegression.png"),
        ("major_axis_um", "nucleus_major_axis_um", "Nucleus vs cell long axis", "NucleusMajorVsCellMajor.png"),
        ("minor_axis_um", "nucleus_minor_axis_um", "Nucleus vs cell short axis", "NucleusMinorVsCellMinor.png"),
    ]
    for x_col, y_col, label, fname in targets:
        if x_col not in area_df.columns or y_col not in area_df.columns:
            continue
        fit = _ols_fit_log(area_df, x_col, y_col)
        if not fit.get("ok"):
            continue
        fig, ax = plt.subplots(figsize=(11, 6))
        _plot_regression_scatter(fit, fit.get("mixed"), label, ax)
        fig.tight_layout()
        path = out_dir / fname
        fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"[ok] wrote {path.relative_to(REPO_ROOT)}")


def save_nc_ratio_vs_area(area_df: pd.DataFrame, figures_dir: Path | None = None) -> Path:
    import statsmodels.api as sm

    out_dir = figures_dir or FIGURES_DIR
    work = area_df[["area_um2", "nc_ratio", "frog_id"]].dropna()
    work = work[(work["area_um2"] > 0) & (work["nc_ratio"] > 0)].copy()
    work["log_area"] = np.log(work["area_um2"].astype(float))
    work["log_nc"] = np.log(work["nc_ratio"].astype(float))
    ols = sm.OLS.from_formula("log_nc ~ log_area", data=work).fit()
    sample = work.sample(min(8000, len(work)), random_state=RANDOM_SEED)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(sample["log_area"], sample["log_nc"], s=10, alpha=0.2, c="#cbd5e1", edgecolors="none")
    xx = np.linspace(work["log_area"].min(), work["log_area"].max(), 100)
    ax.plot(xx, ols.params["Intercept"] + ols.params["log_area"] * xx, color="#0f172a", lw=2, label="OLS fit")
    ax.set_xlabel("log(cell area µm²)")
    ax.set_ylabel("log(N/C ratio)")
    ax.set_title("N/C ratio decreases with cell size (log scale)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path = out_dir / "NCRatioVsArea.png"
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {path.relative_to(REPO_ROOT)}")
    return path


def write_all(
    area_df: pd.DataFrame | None = None,
    frog_df: pd.DataFrame | None = None,
    pred_df: pd.DataFrame | None = None,
    figures_dir: Path | None = None,
) -> None:
    """Generate all biology report figures."""
    out_dir = figures_dir or FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if area_df is None:
        area_df = pd.read_csv(FILTERED_AREAS_CSV)
    if frog_df is None:
        frog_df = pd.read_csv(FROG_METRICS_CSV) if FROG_METRICS_CSV.is_file() else pd.DataFrame()
    if pred_df is None and PREDICTIONS_CSV.is_file():
        pred_df = pd.read_csv(PREDICTIONS_CSV)

    save_area_distribution(area_df, out_dir)
    save_diameter_distribution(area_df, out_dir)
    save_nucleus_distribution(area_df, out_dir)
    save_nc_ratio_distribution(area_df, out_dir)
    save_per_frog_boxplot(area_df, out_dir)
    if pred_df is not None and not pred_df.empty:
        save_yield_by_frog(pred_df, frog_df, out_dir)
    save_icc_variation(area_df, out_dir)
    save_regression_figures(area_df, out_dir)
    save_nc_ratio_vs_area(area_df, out_dir)
    if not frog_df.empty:
        save_outlier_frogs(frog_df, area_df, out_dir)


if __name__ == "__main__":
    write_all()
