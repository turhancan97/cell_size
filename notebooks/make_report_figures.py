"""Regenerate every figure referenced by notebooks/report.md / report.pdf.

This script is the single source of truth for the biologist-facing combined report. It:

    1. Loads the best classifier checkpoint and runs inference on the held-out test split.
    2. Saves qualitative example panels for Part A and Part B.
    3. Produces Part C biology figures via biology_plots.write_report_c_figures.

Run from the repo root:

    conda activate cell-size
    python notebooks/make_report_figures.py
    python notebooks/make_report_figures.py --biology-only

All figures are written to notebooks/figures/.

For the full PDF pipeline, prefer: python notebooks/build_report.py
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from biology_plots import write_all as write_biology_figures
from biology_plots import write_report_c_figures
from classifier_stats import (
    CHECKPOINT_PATH,
    T_BAD,
    T_GOOD,
    TEMPERATURE,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
FIGURES_DIR = REPO_ROOT / "notebooks" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Best model checkpoint (ResNet18, run_2).
CHECKPOINT_PATH = CHECKPOINT_PATH  # re-export for backwards compatibility
CROPS_ROOT = REPO_ROOT / "classifier_output" / "crops" / "mask_bg_false"

# Selective-rejection thresholds (see classifier_selective_rejection_eda.ipynb).
BASELINE_THRESHOLD = 0.5

# Biology inputs.
FILTERED_AREAS_CSV = REPO_ROOT / "classify_output" / "filtered_areas.csv"
FROG_METRICS_CSV = REPO_ROOT / "classify_output" / "frog_aggregated_metrics.csv"

# Misc.
RANDOM_SEED = 42
BATCH_SIZE = 128
NUM_WORKERS = 4
N_EXAMPLES_PER_GROUP = 20


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img.convert("RGB"))


def plot_image_grid(
    paths: list[str],
    titles: list[str],
    suptitle: str,
    output_path: Path,
    ncols: int = 4,
    figsize_scale: float = 3.0,
) -> None:
    if len(paths) == 0:
        print(f"[warn] no examples to plot for: {suptitle}")
        return

    n = len(paths)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(figsize_scale * ncols, figsize_scale * nrows)
    )
    axes = np.atleast_1d(axes).reshape(nrows, ncols)
    for i in range(nrows * ncols):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        ax.axis("off")
        if i < n:
            ax.imshow(load_rgb(paths[i]))
            ax.set_title(titles[i], fontsize=9)
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {output_path.relative_to(REPO_ROOT)}")


def sample_rows(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) == 0:
        return df
    return df.sample(n=min(n, len(df)), random_state=seed)


# ---------------------------------------------------------------------------
# Classifier figures
# ---------------------------------------------------------------------------


def save_classifier_panels(df: pd.DataFrame) -> None:
    """Part A + Part B classifier example panels and threshold plot."""
    save_baseline_panels(df)
    save_selective_panels(df)
    save_threshold_plot(df)
    print_classifier_summary(df)


def generate_all_report_figures() -> None:
    """Regenerate all figures for the combined LaTeX report."""
    from classifier_stats import run_classifier_inference

    set_seed(RANDOM_SEED)

    if CHECKPOINT_PATH.exists():
        df = run_classifier_inference()
        save_classifier_panels(df)
    else:
        print(f"[warn] checkpoint not found at {CHECKPOINT_PATH} — skipping classifier figures")

    if not FILTERED_AREAS_CSV.exists():
        print(f"[warn] biology CSV missing — looked for\n  {FILTERED_AREAS_CSV}")
        return

    area_df = pd.read_csv(FILTERED_AREAS_CSV)
    frog_df = pd.read_csv(FROG_METRICS_CSV) if FROG_METRICS_CSV.exists() else pd.DataFrame()
    print("\n" + "=" * 70)
    print("Part C biology figures")
    print("=" * 70)
    write_report_c_figures(area_df=area_df, frog_df=frog_df, figures_dir=FIGURES_DIR)
    print_biology_summary(area_df, frog_df)
    print("\nAll figures written to", FIGURES_DIR.relative_to(REPO_ROOT))


# ---------------------------------------------------------------------------
# Panel helpers
# ---------------------------------------------------------------------------


def save_baseline_panels(df: pd.DataFrame) -> None:
    """TP / TN / FP / FN panels for the baseline decision policy."""
    mapping = {
        "TP": ("TruePositive.png", "True Positives — cells the model correctly marked as good"),
        "TN": ("TrueNegative.png", "True Negatives — cells the model correctly marked as bad"),
        "FP": ("FalsePositive.png", "False Positives — bad cells mistakenly accepted as good"),
        "FN": ("FalseNegative.png", "False Negatives — good cells mistakenly rejected as bad"),
    }
    for case, (fname, suptitle) in mapping.items():
        sub = sample_rows(df[df["case"] == case], N_EXAMPLES_PER_GROUP, RANDOM_SEED)
        if len(sub) == 0:
            print(f"[warn] no {case} samples to plot")
            continue
        titles = [f"p(good)={row.p_good:.2f}" for row in sub.itertuples(index=False)]
        plot_image_grid(
            sub["path"].tolist(), titles, suptitle, FIGURES_DIR / fname
        )


def save_selective_panels(df: pd.DataFrame) -> None:
    """Accepted / Rejected panels for the selective-rejection policy."""
    acc = df[df["accepted"]]
    rej = df[~df["accepted"]]

    acc_good = acc[(acc["y_pred_sel"] == 1) & (acc["y_true"] == 1)]
    acc_bad = acc[(acc["y_pred_sel"] == 0) & (acc["y_true"] == 0)]
    acc_fp = acc[(acc["y_pred_sel"] == 1) & (acc["y_true"] == 0)]
    acc_fn = acc[(acc["y_pred_sel"] == 0) & (acc["y_true"] == 1)]

    panels = [
        (acc_good, "AcceptedGood.png",
         "Accepted as good (and truly good) — confident correct accepts"),
        (acc_bad, "AcceptedBad.png",
         "Accepted as bad (and truly bad) — confident correct rejects"),
        (acc_fp, "AcceptedFalsePositive.png",
         "Accepted-as-good but actually bad — remaining mistakes after filtering"),
        (acc_fn, "AcceptedFalseNegative.png",
         "Accepted-as-bad but actually good — good cells still missed after filtering"),
        (rej, "RejectedUncertain.png",
         "Rejected / unsure — deferred by the model because confidence was low"),
    ]
    for sub, fname, suptitle in panels:
        sub = sample_rows(sub, N_EXAMPLES_PER_GROUP, RANDOM_SEED + hash(fname) % 10_000)
        if len(sub) == 0:
            print(f"[warn] no samples to plot for {fname}")
            continue
        titles = [f"p(good)={row.p_good_cal:.2f}" for row in sub.itertuples(index=False)]
        plot_image_grid(sub["path"].tolist(), titles, suptitle, FIGURES_DIR / fname)


def save_threshold_plot(df: pd.DataFrame) -> None:
    """Histogram of calibrated p(good) split by true label + threshold lines."""
    fig, ax = plt.subplots(figsize=(10, 5))
    p_bad = df.loc[df["y_true"] == 0, "p_good_cal"].to_numpy()
    p_good = df.loc[df["y_true"] == 1, "p_good_cal"].to_numpy()
    bins = np.linspace(0, 1, 41)
    ax.hist(p_bad, bins=bins, alpha=0.7, color="#f4a261", label=f"truly bad (n={len(p_bad)})")
    ax.hist(p_good, bins=bins, alpha=0.7, color="#2a9d8f", label=f"truly good (n={len(p_good)})")
    ax.axvline(T_BAD, color="#b91c1c", linestyle="--", lw=1.4, label=f"t_bad = {T_BAD:.2f}")
    ax.axvline(T_GOOD, color="#1d4ed8", linestyle="--", lw=1.4, label=f"t_good = {T_GOOD:.2f}")
    ax.axvspan(T_BAD, T_GOOD, color="gray", alpha=0.15, label="reject zone")
    ax.set_yscale("log")
    ax.set_xlabel("Calibrated probability the cell is good")
    ax.set_ylabel("Number of test cells (log scale)")
    ax.set_title("How the model spreads its confidence between good and bad cells")
    ax.legend(loc="upper center")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "Threshold.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {(FIGURES_DIR / 'Threshold.png').relative_to(REPO_ROOT)}")


def print_classifier_summary(df: pd.DataFrame) -> None:
    """Log all headline numbers so report.md can be verified."""
    from sklearn.metrics import confusion_matrix

    n = len(df)
    n_good = int((df["y_true"] == 1).sum())
    n_bad = int((df["y_true"] == 0).sum())
    print(f"\n[summary] test split: n={n}  good={n_good}  bad={n_bad}")

    cm = confusion_matrix(df["y_true"], df["y_pred"], labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    acc = (tp + tn) / n
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    print("[summary] baseline @ 0.50:")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(
        f"  acc={acc:.4f}  recall={rec:.4f}  precision={prec:.4f}  "
        f"f1={f1:.4f}  specificity={spec:.4f}"
    )

    acc_mask = df["accepted"]
    n_acc = int(acc_mask.sum())
    n_rej = int((~acc_mask).sum())
    rej = df[~acc_mask]
    rej_good = int((rej["y_true"] == 1).sum())
    rej_bad = int((rej["y_true"] == 0).sum())
    sub = df[acc_mask]
    cm_s = confusion_matrix(sub["y_true"], sub["y_pred_sel"], labels=[0, 1])
    tn_s, fp_s, fn_s, tp_s = cm_s.ravel()
    n_s = n_acc
    acc_s = (tp_s + tn_s) / n_s if n_s else 0.0
    rec_s = tp_s / (tp_s + fn_s) if (tp_s + fn_s) else 0.0
    prec_s = tp_s / (tp_s + fp_s) if (tp_s + fp_s) else 0.0
    f1_s = 2 * prec_s * rec_s / (prec_s + rec_s) if (prec_s + rec_s) else 0.0
    spec_s = tn_s / (tn_s + fp_s) if (tn_s + fp_s) else 0.0
    cov = n_acc / n
    print(
        f"[summary] selective @ t_bad={T_BAD} / t_good={T_GOOD} "
        f"(temperature={TEMPERATURE:.4f}):"
    )
    print(
        f"  coverage={cov:.4f}  accepted={n_acc}  rejected={n_rej} "
        f"(good={rej_good}, bad={rej_bad})"
    )
    print(
        f"  accepted-only: TP={tp_s}  FP={fp_s}  TN={tn_s}  FN={fn_s}"
    )
    print(
        f"  acc={acc_s:.4f}  recall={rec_s:.4f}  precision={prec_s:.4f}  "
        f"f1={f1_s:.4f}  specificity={spec_s:.4f}"
    )


def print_biology_summary(area_df: pd.DataFrame, frog_df: pd.DataFrame) -> None:
    print("\n[summary] biology")
    print(f"  good cells in filtered_areas.csv : {len(area_df)}")
    if "frog_id" in area_df.columns:
        print(f"  frogs with good cells            : {area_df['frog_id'].nunique()}")
    if "area_um2" in area_df.columns:
        s = pd.to_numeric(area_df["area_um2"], errors="coerce").dropna()
        if len(s):
            print(
                f"  cell area µm² : median={s.median():.2f}  "
                f"mean={s.mean():.2f}  sd={s.std():.2f}  "
                f"IQR=[{s.quantile(0.25):.2f}, {s.quantile(0.75):.2f}]"
            )
    if "nc_ratio" in area_df.columns:
        s = pd.to_numeric(area_df["nc_ratio"], errors="coerce").dropna()
        if len(s):
            print(f"  N/C ratio     : median={s.median():.3f}  mean={s.mean():.3f}  sd={s.std():.3f}")
    if "area_um2_mean" in frog_df.columns:
        s = pd.to_numeric(frog_df["area_um2_mean"], errors="coerce").dropna()
        if len(s):
            print(
                f"  per-frog mean cell area µm² : median={s.median():.2f}  "
                f"mean={s.mean():.2f}  sd={s.std():.2f} "
                f"(min={s.min():.2f}, max={s.max():.2f})"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Regenerate report figures.")
    parser.add_argument(
        "--biology-only",
        action="store_true",
        help="Skip classifier panels; write all biology figures (biology_plots.write_all).",
    )
    parser.add_argument(
        "--combined-only",
        action="store_true",
        help="Classifier panels + Part C figures only (default full run).",
    )
    args = parser.parse_args()

    set_seed(RANDOM_SEED)

    if args.biology_only:
        print("=" * 70)
        print("Biology figures only")
        print("=" * 70)
        if not FILTERED_AREAS_CSV.exists():
            print(f"[warn] biology CSV missing: {FILTERED_AREAS_CSV}")
            return
        write_biology_figures()
        return

    generate_all_report_figures()


if __name__ == "__main__":
    main()
