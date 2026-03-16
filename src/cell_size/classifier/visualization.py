"""Classifier visualisations: confusion matrix and filtered overlays."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import center_of_mass, find_objects
from skimage.segmentation import find_boundaries
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

logger = logging.getLogger(__name__)


def plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    save_path: str | Path,
    class_names: tuple[str, str] = ("bad", "good"),
) -> Path:
    """Generate and save a confusion matrix heatmap."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Cell Quality Classifier - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150)
    plt.close(fig)

    logger.info("Confusion matrix saved -> %s", save_path)
    return save_path


def _to_display_rgb(img: np.ndarray) -> np.ndarray:
    """Normalise any image to uint8 RGB [0..255]."""
    if img.ndim == 2:
        out = np.stack([img] * 3, axis=-1)
    else:
        out = img.copy()

    out = out.astype(np.float64)
    lo, hi = out.min(), out.max()
    if hi - lo > 0:
        out = (out - lo) / (hi - lo)
    else:
        out = np.zeros_like(out)

    return (out * 255).astype(np.uint8)


def generate_filtered_overlay(
    img: np.ndarray,
    masks: np.ndarray,
    predictions_df: pd.DataFrame,
    output_path: str | Path,
    fontsize: int = 8,
) -> Path:
    """Draw overlay where good cells are coloured and bad cells are greyed out.

    ``predictions_df`` must have columns ``mask_index`` and ``predicted_verdict``.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    img_rgb = _to_display_rgb(img)
    overlay = img_rgb.astype(np.float64) / 255.0

    n_cells = int(masks.max())
    if n_cells == 0:
        fig, ax = plt.subplots(figsize=(20, 15))
        ax.imshow(img_rgb)
        ax.set_title("No cells detected")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        return output_path

    good_labels = set(
        predictions_df.loc[
            predictions_df["predicted_verdict"] == "good", "mask_index"
        ].astype(int)
    )

    rng = np.random.RandomState(42)
    colors = rng.rand(n_cells + 1, 3)
    colors[0] = 0

    alpha_good = 0.4
    alpha_bad = 0.15
    grey = np.array([0.5, 0.5, 0.5])

    for label in range(1, n_cells + 1):
        cell_mask = masks == label
        if not cell_mask.any():
            continue
        if label in good_labels:
            colour = colors[label]
            alpha = alpha_good
        else:
            colour = grey
            alpha = alpha_bad
        overlay[cell_mask] = (1 - alpha) * overlay[cell_mask] + alpha * colour

    result = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)

    outlines = find_boundaries(masks, mode="inner")
    out_y, out_x = np.nonzero(outlines)
    for y, x in zip(out_y, out_x):
        label = masks[y, x]
        if label in good_labels:
            result[y, x] = (0, 200, 0)
        else:
            result[y, x] = (128, 128, 128)

    slices = find_objects(masks)
    centroids: list[tuple[int, int, int]] = []
    for i, slc in enumerate(slices):
        if slc is None:
            continue
        mask_label = i + 1
        mask_slice = masks[slc] == mask_label
        if mask_slice.sum() == 0:
            continue
        com = center_of_mass(mask_slice)
        y_cent = int(com[0] + slc[0].start)
        x_cent = int(com[1] + slc[1].start)
        centroids.append((y_cent, x_cent, mask_label))

    n_good = len(good_labels)
    n_bad = n_cells - n_good
    fig, ax = plt.subplots(figsize=(20, 15))
    ax.imshow(result)
    ax.set_title(f"Filtered overlay: {n_good} good (colour), {n_bad} bad (grey)", fontsize=16)
    ax.axis("off")

    for y, x, label in centroids:
        is_good = label in good_labels
        ax.text(
            x,
            y,
            str(label),
            fontsize=fontsize,
            fontweight="bold",
            color="white" if is_good else "gray",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="darkgreen" if is_good else "dimgray",
                alpha=0.7 if is_good else 0.4,
                edgecolor="white" if is_good else "gray",
                linewidth=1.5 if is_good else 0.5,
            ),
            ha="center",
            va="center",
        )

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    logger.info("Filtered overlay saved -> %s (%d good, %d bad)", output_path, n_good, n_bad)
    return output_path
