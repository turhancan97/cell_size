"""Classifier metrics for the combined report (test split, Parts A & B)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from biology_stats import _md_table

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = REPO_ROOT / "classify_output" / "analysis"
CHECKPOINT_PATH = REPO_ROOT / "classifier_output" / "run_2" / "best_model.pt"
CROPS_ROOT = REPO_ROOT / "classifier_output" / "crops" / "mask_bg_false"

BASELINE_THRESHOLD = 0.5
T_BAD = 0.10
T_GOOD = 0.76
TEMPERATURE = 1.8838

BATCH_SIZE = 128
NUM_WORKERS = 4


def _pct(x: float, ndigits: int = 2) -> str:
    return f"{x * 100:.{ndigits}f}%"


def _metrics_from_cm(tn: int, fp: int, fn: int, tp: int) -> dict[str, float]:
    n = tn + fp + fn + tp
    acc = (tp + tn) / n if n else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return {"accuracy": acc, "recall": rec, "precision": prec, "f1": f1, "specificity": spec}


def run_classifier_inference() -> pd.DataFrame:
    """Load checkpoint, run inference on test split, return per-sample df."""
    import torch
    from sklearn.metrics import confusion_matrix  # noqa: F401 — used indirectly via compute
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from torchvision.datasets import ImageFolder

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from cell_size.classifier.dataset import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
    from cell_size.classifier.models import build_model  # noqa: E402

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class ImageFolderWithPaths(ImageFolder):
        def __getitem__(self, index):
            image, target = super().__getitem__(index)
            path, _ = self.samples[index]
            return image, target, path

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Classifier checkpoint not found: {CHECKPOINT_PATH}")

    ckpt = torch.load(str(CHECKPOINT_PATH), map_location=device, weights_only=False)
    encoder = ckpt["encoder"]
    crop_size = int(ckpt.get("crop_size", 224))

    model = build_model(encoder=encoder, pretrained=False, freeze_encoder=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    eval_transform = transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    test_dir = CROPS_ROOT / "test"
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test crop folder not found: {test_dir}")

    ds = ImageFolderWithPaths(str(test_dir), transform=eval_transform)
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
            logits = model(images.to(device)).squeeze(1).cpu().numpy()
            probs_good = 1.0 / (1.0 + np.exp(-logits))
            probs_good_cal = 1.0 / (1.0 + np.exp(-logits / TEMPERATURE))
            y_true = (targets.numpy() == good_idx).astype(int)
            for path, yt, pg, pg_cal in zip(paths, y_true, probs_good, probs_good_cal):
                rows.append({
                    "path": path,
                    "y_true": int(yt),
                    "p_good": float(pg),
                    "p_good_cal": float(pg_cal),
                })

    df = pd.DataFrame(rows)
    df["y_pred"] = (df["p_good"] >= BASELINE_THRESHOLD).astype(int)
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


def _cm_tables(tn: int, fp: int, fn: int, tp: int, *, accepted_label: str = "") -> tuple[str, str]:
    suffix = f" ({accepted_label})" if accepted_label else ""
    counts = _md_table(
        [f"True \\ Pred{suffix}", "Pred bad", "Pred good", "Row total"],
        [
            ["True bad", f"**{tn}** (TN)", f"**{fp}** (FP)", str(tn + fp)],
            ["True good", f"**{fn}** (FN)", f"**{tp}** (TP)", str(fn + tp)],
        ],
    )
    bad_total = tn + fp
    good_total = fn + tp
    pct = _md_table(
        [f"True \\ Pred{suffix}", "Pred bad", "Pred good"],
        [
            [
                "True bad",
                _pct(tn / bad_total if bad_total else 0),
                _pct(fp / bad_total if bad_total else 0),
            ],
            [
                "True good",
                _pct(fn / good_total if good_total else 0),
                _pct(tp / good_total if good_total else 0),
            ],
        ],
    )
    return counts, pct


def compute_classifier_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Return nested stats and string placeholders from inference dataframe."""
    from sklearn.metrics import confusion_matrix

    n = len(df)
    n_good = int((df["y_true"] == 1).sum())
    n_bad = int((df["y_true"] == 0).sum())

    cm = confusion_matrix(df["y_true"], df["y_pred"], labels=[0, 1])
    tn, fp, fn, tp = (int(x) for x in cm.ravel())
    base = _metrics_from_cm(tn, fp, fn, tp)

    acc_mask = df["accepted"]
    n_acc = int(acc_mask.sum())
    n_rej = int((~acc_mask).sum())
    rej = df[~acc_mask]
    rej_good = int((rej["y_true"] == 1).sum())
    rej_bad = int((rej["y_true"] == 0).sum())
    sub = df[acc_mask]
    cm_s = confusion_matrix(sub["y_true"], sub["y_pred_sel"], labels=[0, 1])
    tn_s, fp_s, fn_s, tp_s = (int(x) for x in cm_s.ravel())
    sel = _metrics_from_cm(tn_s, fp_s, fn_s, tp_s)
    cov = n_acc / n if n else float("nan")

    baseline_metrics_table = _md_table(
        ["Metric", "Value"],
        [
            ["Accuracy", f"**{_pct(base['accuracy'])}**"],
            ["Recall (sensitivity to good cells)", f"**{_pct(base['recall'])}**"],
            ["Precision", f"**{_pct(base['precision'])}**"],
            ["F1", f"**{_pct(base['f1'])}**"],
            ["Specificity (correctly discarded bad cells)", f"**{_pct(base['specificity'])}**"],
        ],
    )
    baseline_cm_counts, baseline_cm_pct = _cm_tables(tn, fp, fn, tp)
    selective_cm_counts, selective_cm_pct = _cm_tables(
        tn_s, fp_s, fn_s, tp_s, accepted_label="accepted",
    )
    selective_comparison_table = _md_table(
        ["Metric", "Baseline", "Selective rejection (accepted only)"],
        [
            ["Coverage (cells that got a decision)", "100.00%", f"**{_pct(cov)}**"],
            ["Accuracy", _pct(base["accuracy"]), f"**{_pct(sel['accuracy'])}**"],
            ["Recall", _pct(base["recall"]), f"**{_pct(sel['recall'])}**"],
            ["Precision", _pct(base["precision"]), f"**{_pct(sel['precision'])}**"],
            ["F1", _pct(base["f1"]), f"**{_pct(sel['f1'])}**"],
            ["Specificity", _pct(base["specificity"]), f"**{_pct(sel['specificity'])}**"],
            ["False positives (kept by mistake)", str(fp), f"**{fp_s}**"],
            ["False negatives (missed good cells)", str(fn), f"**{fn_s}**"],
        ],
    )

    placeholders = {
        "encoder_name": "ResNet18",
        "n_test_cells": f"{n:,}",
        "n_test_good": str(n_good),
        "n_test_bad": str(n_bad),
        "baseline_accuracy_pct": _pct(base["accuracy"], 2),
        "baseline_recall_pct": _pct(base["recall"], 2),
        "baseline_precision_pct": _pct(base["precision"], 2),
        "baseline_specificity_pct": _pct(base["specificity"], 2),
        "baseline_f1_pct": _pct(base["f1"], 2),
        "selective_accuracy_pct": _pct(sel["accuracy"], 2),
        "selective_recall_pct": _pct(sel["recall"], 2),
        "selective_precision_pct": _pct(sel["precision"], 2),
        "selective_f1_pct": _pct(sel["f1"], 2),
        "selective_specificity_pct": _pct(sel["specificity"], 2),
        "coverage_pct": _pct(cov, 2),
        "n_accepted": f"{n_acc:,}",
        "n_rejected": str(n_rej),
        "n_rejected_good": str(rej_good),
        "n_rejected_bad": str(rej_bad),
        "baseline_fp": str(fp),
        "baseline_fn": str(fn),
        "selective_fp": str(fp_s),
        "selective_fn": str(fn_s),
        "t_bad": str(T_BAD),
        "t_good": str(T_GOOD),
        "temperature": f"{TEMPERATURE:.4f}",
        "baseline_metrics_table": baseline_metrics_table,
        "baseline_cm_counts_table": baseline_cm_counts,
        "baseline_cm_pct_table": baseline_cm_pct,
        "selective_comparison_table": selective_comparison_table,
        "selective_cm_counts_table": selective_cm_counts,
        "selective_cm_pct_table": selective_cm_pct,
    }
    return {
        "n_test": n,
        "baseline": base,
        "selective": sel,
        "coverage": cov,
        "placeholders": placeholders,
    }


def load_classifier_stats(
    *,
    run_inference: bool = True,
    df: pd.DataFrame | None = None,
    cache_path: Path | None = None,
) -> dict[str, Any]:
    cache_path = cache_path or (ANALYSIS_DIR / "classifier_stats.json")
    if df is None and not run_inference and cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if "placeholders" in cached:
            return cached
    if df is None and run_inference:
        df = run_classifier_inference()
    if df is None:
        raise ValueError("No classifier dataframe available")
    stats = compute_classifier_stats(df)
    write_stats_cache(cache_path, stats=stats)
    return stats


def write_stats_cache(
    path: Path | None = None,
    *,
    df: pd.DataFrame | None = None,
    stats: dict[str, Any] | None = None,
) -> Path:
    path = path or (ANALYSIS_DIR / "classifier_stats.json")
    if stats is None:
        if df is not None:
            stats = compute_classifier_stats(df)
        elif path.is_file():
            return path
        else:
            stats = load_classifier_stats(run_inference=True)
    cache = {k: v for k, v in stats.items() if k != "placeholders"}
    cache["placeholders"] = stats["placeholders"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, default=str), encoding="utf-8")
    return path


if __name__ == "__main__":
    s = load_classifier_stats()
    p = s["placeholders"]
    print("n_test_cells:", p["n_test_cells"])
    print("baseline_accuracy:", p["baseline_accuracy_pct"])
