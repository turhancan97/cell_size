"""Regenerate every figure referenced by notebooks/report.md / report.html.

This script is authored to be a single source of truth for the biologist-
facing report. It:

    1. Loads the best classifier checkpoint (SqueezeNet1.1, full fine-tune)
       and runs inference on the held-out test split of cell crops.
    2. Saves qualitative example panels used in Part A of the report:
       TruePositive.png, TrueNegative.png, FalsePositive.png, FalseNegative.png
    3. Applies temperature scaling + selective-rejection thresholds
       (t_bad, t_good) chosen on the validation split (see
       classifier_selective_rejection_eda.ipynb) and saves Part B panels:
       AcceptedGood.png, AcceptedBad.png, AcceptedFalsePositive.png,
       AcceptedFalseNegative.png, RejectedUncertain.png, Threshold.png
    4. Reads classify_output/filtered_areas.csv and
       classify_output/frog_aggregated_metrics.csv and produces the
       biology figures (via biology_plots.py) used in Part C / LaTeX report.

Run from the repo root:

    conda activate cell-size
    python notebooks/make_report_figures.py
    python notebooks/make_report_figures.py --biology-only

All figures are written to notebooks/figures/.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib.patches import Patch
from PIL import Image
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from biology_plots import write_all as write_biology_figures


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
FIGURES_DIR = REPO_ROOT / "notebooks" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Best model (from classifier_model_eda.ipynb and selective_rejection notebook).
CHECKPOINT_PATH = (
    REPO_ROOT
    / "classifier_output"
    / "run_2"
    / "best_model.pt"
)
CROPS_ROOT = REPO_ROOT / "classifier_output" / "crops" / "mask_bg_false"

# Selective-rejection thresholds selected on VAL (notebook cell output).
# (See classifier_selective_rejection_eda.ipynb cells 12 & 14.)
BASELINE_THRESHOLD = 0.5
T_BAD = 0.10
T_GOOD = 0.76
TEMPERATURE = 1.8838  # fitted on val logits in the notebook

# Biology inputs.
FILTERED_AREAS_CSV = REPO_ROOT / "classify_output" / "filtered_areas.csv"
FROG_METRICS_CSV = REPO_ROOT / "classify_output" / "frog_aggregated_metrics.csv"

# Misc.
RANDOM_SEED = 42
BATCH_SIZE = 128
NUM_WORKERS = 4
N_EXAMPLES_PER_GROUP = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ImageFolderWithPaths(ImageFolder):
    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        path, _ = self.samples[index]
        return image, target, path


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


def run_classifier_inference() -> pd.DataFrame:
    """Load checkpoint, run inference on test split, return per-sample df."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from cell_size.classifier.models import build_model  # noqa: E402

    print(f"[info] loading checkpoint: {CHECKPOINT_PATH}")
    ckpt = torch.load(str(CHECKPOINT_PATH), map_location=DEVICE, weights_only=False)
    encoder = ckpt["encoder"]
    crop_size = int(ckpt.get("crop_size", 224))
    print(f"[info] encoder={encoder}  crop_size={crop_size}  device={DEVICE}")

    model = build_model(encoder=encoder, pretrained=False, freeze_encoder=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from cell_size.classifier.dataset import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402

    eval_transform = transforms.Compose(
        [
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    test_dir = CROPS_ROOT / "test"
    ds = ImageFolderWithPaths(str(test_dir), transform=eval_transform)
    print(f"[info] loaded {len(ds)} test samples from {test_dir}")
    print(f"[info] class_to_idx={ds.class_to_idx}")
    good_idx = ds.class_to_idx["good"]

    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    rows = []
    with torch.no_grad():
        for images, targets, paths in loader:
            logits = model(images.to(DEVICE)).squeeze(1).cpu().numpy()
            probs_good = 1.0 / (1.0 + np.exp(-logits))
            probs_good_cal = 1.0 / (1.0 + np.exp(-logits / TEMPERATURE))
            y_true = (targets.numpy() == good_idx).astype(int)
            for path, yt, pg, pg_cal in zip(paths, y_true, probs_good, probs_good_cal):
                rows.append(
                    {
                        "path": path,
                        "y_true": int(yt),
                        "p_good": float(pg),
                        "p_good_cal": float(pg_cal),
                    }
                )

    df = pd.DataFrame(rows)
    df["y_pred"] = (df["p_good"] >= BASELINE_THRESHOLD).astype(int)
    df["true_label"] = np.where(df["y_true"] == 1, "good", "bad")
    df["pred_label"] = np.where(df["y_pred"] == 1, "good", "bad")
    df["case"] = [
        "TP" if yt == 1 and yp == 1
        else "TN" if yt == 0 and yp == 0
        else "FP" if yt == 0 and yp == 1
        else "FN"
        for yt, yp in zip(df["y_true"], df["y_pred"])
    ]

    accepted = (df["p_good_cal"] <= T_BAD) | (df["p_good_cal"] >= T_GOOD)
    y_pred_sel = np.full(len(df), -1, dtype=int)
    y_pred_sel[df["p_good_cal"] <= T_BAD] = 0
    y_pred_sel[df["p_good_cal"] >= T_GOOD] = 1
    df["accepted"] = accepted
    df["y_pred_sel"] = y_pred_sel
    return df


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
    """Log all headline numbers so report.md / report.html can be verified."""
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


# ---------------------------------------------------------------------------
# Biology figures
# ---------------------------------------------------------------------------


BIO_COLOR_CELL = "#2a9d8f"
BIO_COLOR_NUCLEUS = "#264653"
BIO_COLOR_HIGHLIGHT = "#e76f51"


def _hist_with_summary(
    ax: plt.Axes,
    values: np.ndarray,
    color: str,
    title: str,
    xlabel: str,
    bins: int = 40,
) -> None:
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy()
    ax.hist(values, bins=bins, color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
    if len(values):
        med = float(np.median(values))
        q1, q3 = np.quantile(values, [0.25, 0.75])
        ax.axvline(med, color="#222", linestyle="--", linewidth=1.3, label=f"median={med:.2f}")
        ax.axvspan(q1, q3, color="#222", alpha=0.08, label=f"IQR {q1:.2f}–{q3:.2f}")
        ax.legend(loc="upper right", fontsize=9)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of cells")


def save_area_distribution(area_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    _hist_with_summary(
        axes[0], area_df["area_px"], BIO_COLOR_CELL,
        "Cell area (pixels)", "area_px",
    )
    _hist_with_summary(
        axes[1], area_df["area_um2"], BIO_COLOR_CELL,
        "Cell area (µm²)", "area_um2",
    )
    fig.suptitle("Cell area distribution (good cells only)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIGURES_DIR / "AreaDistribution.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("[ok] wrote figures/AreaDistribution.png")


def save_diameter_distribution(area_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    _hist_with_summary(
        axes[0], area_df["major_axis_um"], BIO_COLOR_CELL,
        "Cell long diameter (µm)", "major_axis_um",
    )
    _hist_with_summary(
        axes[1], area_df["minor_axis_um"], BIO_COLOR_CELL,
        "Cell short diameter (µm)", "minor_axis_um",
    )
    fig.suptitle("Cell diameter distribution (good cells only)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIGURES_DIR / "DiameterDistribution.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("[ok] wrote figures/DiameterDistribution.png")


def save_nucleus_distribution(area_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    _hist_with_summary(
        axes[0], area_df["nucleus_area_um2"], BIO_COLOR_NUCLEUS,
        "Nucleus area (µm²)", "nucleus_area_um2",
    )
    _hist_with_summary(
        axes[1], area_df["nucleus_major_axis_um"], BIO_COLOR_NUCLEUS,
        "Nucleus long diameter (µm)", "nucleus_major_axis_um",
    )
    _hist_with_summary(
        axes[2], area_df["nucleus_minor_axis_um"], BIO_COLOR_NUCLEUS,
        "Nucleus short diameter (µm)", "nucleus_minor_axis_um",
    )
    fig.suptitle("Nucleus size distribution (only cells with a matched nucleus)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIGURES_DIR / "NucleusDistribution.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("[ok] wrote figures/NucleusDistribution.png")


def save_nc_ratio_distribution(area_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    _hist_with_summary(
        ax, area_df["nc_ratio"], BIO_COLOR_NUCLEUS,
        "Nucleus-to-cell area ratio (N/C)", "nc_ratio",
    )
    fig.suptitle("N/C ratio — how much of the cell is taken up by the nucleus", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIGURES_DIR / "NCRatioDistribution.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("[ok] wrote figures/NCRatioDistribution.png")


def _top_frogs_by_count(area_df: pd.DataFrame, k: int = 40) -> list:
    """Return the k frog_ids with the most good cells, in that order."""
    counts = area_df.groupby("frog_id").size()
    return counts.sort_values(ascending=False).head(k).index.tolist()


def _per_frog_boxplot(
    area_df: pd.DataFrame,
    value_col: str,
    frogs_sorted: list,
    color: str,
    ylabel: str,
    title: str,
    output_name: str,
) -> None:
    """Generic per-frog boxplot helper used for both cell area and nucleus area."""
    sub = area_df[area_df["frog_id"].isin(frogs_sorted)]
    data = [
        sub.loc[sub["frog_id"] == fid, value_col].dropna().to_numpy()
        for fid in frogs_sorted
    ]
    sample_counts = [len(d) for d in data]

    fig, ax = plt.subplots(figsize=(14, 5.5))
    bp = ax.boxplot(
        data,
        positions=np.arange(len(frogs_sorted)),
        widths=0.6,
        patch_artist=True,
        showfliers=False,
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
        rotation=60,
        ha="right",
        fontsize=7,
    )
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / output_name, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote figures/{output_name}")


def save_per_frog_boxplot(area_df: pd.DataFrame) -> None:
    """Per-frog cell-area boxplot + a matching nucleus-area boxplot.

    The two figures share the same frog ordering (sorted by per-frog median
    cell area) so they can be read side by side.
    """
    if "frog_id" not in area_df.columns:
        print("[warn] no frog_id column; skipping per-frog boxplots")
        return

    top_frogs = _top_frogs_by_count(area_df, k=40)
    sub = area_df[area_df["frog_id"].isin(top_frogs)]
    medians = sub.groupby("frog_id")["area_um2"].median().sort_values()
    frogs_sorted = medians.index.tolist()

    _per_frog_boxplot(
        area_df=area_df,
        value_col="area_um2",
        frogs_sorted=frogs_sorted,
        color=BIO_COLOR_CELL,
        ylabel="Cell area (µm²)",
        title=(
            f"Per-frog cell-area distribution (top-{len(frogs_sorted)} frogs "
            f"by cell count, sorted by median cell area)"
        ),
        output_name="PerFrogBoxplot.png",
    )

    if "nucleus_area_um2" not in area_df.columns:
        print("[warn] no nucleus_area_um2 column; skipping PerFrogNucleusBoxplot")
        return

    _per_frog_boxplot(
        area_df=area_df,
        value_col="nucleus_area_um2",
        frogs_sorted=frogs_sorted,
        color=BIO_COLOR_NUCLEUS,
        ylabel="Nucleus area (µm²)",
        title=(
            f"Per-frog nucleus-area distribution (same {len(frogs_sorted)} frogs, "
            f"same ordering as the cell-area boxplot)"
        ),
        output_name="PerFrogNucleusBoxplot.png",
    )


def save_per_frog_scatter(frog_df: pd.DataFrame) -> None:
    """Scatter of per-frog mean area vs. per-frog n_cells, with std as error bars."""
    if "area_um2_mean" not in frog_df.columns:
        print("[warn] frog metrics missing area_um2_mean; skipping PerFrogScatter")
        return

    df = frog_df.dropna(subset=["area_um2_mean"]).copy()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.errorbar(
        df["n_cells"],
        df["area_um2_mean"],
        yerr=df.get("area_um2_std"),
        fmt="o",
        color=BIO_COLOR_CELL,
        ecolor="#8ab8b3",
        elinewidth=0.8,
        capsize=2,
        markersize=5,
        alpha=0.85,
    )

    overall_median = df["area_um2_mean"].median()
    overall_mean = df["area_um2_mean"].mean()
    overall_std = df["area_um2_mean"].std()
    ax.axhline(overall_median, color="#1a4f49", linestyle="--", linewidth=1.2,
               label=f"across-frog median = {overall_median:.1f} µm²")
    ax.axhspan(
        overall_mean - overall_std, overall_mean + overall_std,
        color="#1a4f49", alpha=0.08, label=f"across-frog mean ± 1 SD"
    )

    ax.set_xlabel("Number of good cells measured per frog")
    ax.set_ylabel("Mean cell area per frog (µm²)")
    ax.set_title("Cross-frog variability of mean cell area (error bars = within-frog SD)")
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "PerFrogScatter.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("[ok] wrote figures/PerFrogScatter.png")


def save_outlier_frogs(
    frog_df: pd.DataFrame,
    area_df: pd.DataFrame,
) -> None:
    """Highlight the 5 frogs with the smallest and the 5 with the largest mean cell area."""
    if "area_um2_mean" not in frog_df.columns:
        print("[warn] frog metrics missing area_um2_mean; skipping OutlierFrogs")
        return

    df = frog_df.dropna(subset=["area_um2_mean"]).copy()
    df = df.sort_values("area_um2_mean")
    small = df.head(5)["frog_id"].tolist()
    large = df.tail(5)["frog_id"].tolist()

    overall_median = df["area_um2_mean"].median()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))

    ax1.barh(
        df["frog_id"].astype(str),
        df["area_um2_mean"],
        color="#cbd5e1",
        edgecolor="#94a3b8",
    )
    for fid in small:
        y = df.index[df["frog_id"] == fid][0]
        # Positional bar index (barh uses categorical, so use enumerate below).
    # Re-draw the highlighted bars on top using the positional index.
    order = df["frog_id"].tolist()
    for fid, color in list(zip(small, [BIO_COLOR_HIGHLIGHT] * len(small))) + list(
        zip(large, ["#1d4ed8"] * len(large))
    ):
        if fid in order:
            pos = order.index(fid)
            ax1.barh([str(fid)], [df.loc[df["frog_id"] == fid, "area_um2_mean"].iloc[0]],
                     color=color, edgecolor=color)
    ax1.axvline(overall_median, color="#1a4f49", linestyle="--", linewidth=1.2,
                label=f"median = {overall_median:.1f} µm²")
    ax1.set_xlabel("Mean cell area per frog (µm²)")
    ax1.set_title("Per-frog ranking (orange = smallest 5, blue = largest 5)")
    ax1.tick_params(axis="y", labelsize=5)
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(axis="x", alpha=0.2)

    bins = np.linspace(
        float(area_df["area_um2"].quantile(0.01)),
        float(area_df["area_um2"].quantile(0.99)),
        60,
    )
    ax2.hist(
        area_df["area_um2"].dropna(),
        bins=bins,
        color="#cbd5e1",
        alpha=0.6,
        label="all good cells",
    )
    for fid in small + large:
        color = BIO_COLOR_HIGHLIGHT if fid in small else "#1d4ed8"
        sub = area_df.loc[area_df["frog_id"] == fid, "area_um2"].dropna()
        if not len(sub):
            continue
        ax2.hist(
            sub, bins=bins, alpha=0.55, color=color,
            label=f"frog {fid} (n={len(sub)})",
        )
    ax2.set_xlabel("Cell area (µm²)")
    ax2.set_ylabel("Number of cells")
    ax2.set_title("Area distribution: outlier frogs vs. the whole dataset")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(axis="y", alpha=0.2)

    fig.suptitle("Outlier frogs — worth a manual check", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIGURES_DIR / "OutlierFrogs.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("[ok] wrote figures/OutlierFrogs.png")


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
        help="Skip classifier panels; write biology figures only.",
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

    print("=" * 70)
    print("Regenerating report figures")
    print("=" * 70)

    if CHECKPOINT_PATH.exists():
        df = run_classifier_inference()
        save_baseline_panels(df)
        save_selective_panels(df)
        save_threshold_plot(df)
        print_classifier_summary(df)
    else:
        print(f"[warn] checkpoint not found at {CHECKPOINT_PATH} — skipping classifier figures")

    print("\n" + "=" * 70)
    print("Biology figures")
    print("=" * 70)
    if not FILTERED_AREAS_CSV.exists():
        print(f"[warn] biology CSV missing — looked for\n  {FILTERED_AREAS_CSV}")
        return

    area_df = pd.read_csv(FILTERED_AREAS_CSV)
    frog_df = pd.read_csv(FROG_METRICS_CSV) if FROG_METRICS_CSV.exists() else pd.DataFrame()
    write_biology_figures(area_df=area_df, frog_df=frog_df)
    print_biology_summary(area_df, frog_df)

    print("\nAll figures written to", FIGURES_DIR.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
