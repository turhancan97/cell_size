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
    fontsize: int = 12,
    nuc_masks: np.ndarray | None = None,
) -> Path:
    """Draw overlay where good cells are coloured, bad cells greyed, rejected highlighted.

    ``predictions_df`` must have columns ``mask_index`` and ``predicted_verdict``.

    When *nuc_masks* is provided, nucleus boundaries are drawn as cyan
    outlines inside good cells.
    """
    output_path = Path(output_path).with_suffix(".jpg")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    img_rgb = _to_display_rgb(img)
    # Dim the base image so class overlays are visually dominant.
    overlay = (img_rgb.astype(np.float64) / 255.0) * 0.75

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
    rejected_labels = set(
        predictions_df.loc[
            predictions_df["predicted_verdict"] == "rejected", "mask_index"
        ].astype(int)
    )

    # Fixed high-contrast, colorblind-friendly class palette.
    good_fill = np.array([0, 158, 115], dtype=np.float64) / 255.0
    bad_fill = np.array([213, 94, 0], dtype=np.float64) / 255.0
    rejected_fill = np.array([204, 121, 167], dtype=np.float64) / 255.0

    good_outline = np.array([0, 109, 85], dtype=np.uint8)
    bad_outline = np.array([140, 45, 4], dtype=np.uint8)
    rejected_outline = np.array([122, 46, 94], dtype=np.uint8)

    alpha_good = 0.45
    alpha_bad = 0.45
    alpha_rejected = 0.55

    for label in range(1, n_cells + 1):
        cell_mask = masks == label
        if not cell_mask.any():
            continue
        if label in good_labels:
            colour = good_fill
            alpha = alpha_good
        elif label in rejected_labels:
            colour = rejected_fill
            alpha = alpha_rejected
        else:
            colour = bad_fill
            alpha = alpha_bad
        overlay[cell_mask] = (1 - alpha) * overlay[cell_mask] + alpha * colour

    result = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)

    outlines = find_boundaries(masks, mode="inner")
    out_y, out_x = np.nonzero(outlines)
    for y, x in zip(out_y, out_x):
        label = masks[y, x]
        if label in good_labels:
            result[y, x] = good_outline
        elif label in rejected_labels:
            result[y, x] = rejected_outline
        else:
            result[y, x] = bad_outline

    # Draw nucleus boundaries inside good cells
    if nuc_masks is not None:
        nuc_outlines = find_boundaries(nuc_masks, mode="inner")
        nuc_y, nuc_x = np.nonzero(nuc_outlines)
        cyan = np.array([0, 220, 255], dtype=np.uint8)
        for y, x in zip(nuc_y, nuc_x):
            cell_label = masks[y, x]
            if cell_label in good_labels:
                result[y, x] = cyan

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
    n_rejected = len(rejected_labels)
    n_bad = max(0, n_cells - n_good - n_rejected)
    title = (
        f"Filtered overlay: {n_good} good (green), {n_bad} bad (orange), "
        f"{n_rejected} rejected (magenta)"
    )
    if nuc_masks is not None:
        title += " | nucleus boundaries (cyan)"
    fig, ax = plt.subplots(figsize=(20, 15))
    ax.imshow(result)
    ax.set_title(title, fontsize=16)
    ax.axis("off")

    for y, x, label in centroids:
        is_good = label in good_labels
        is_rejected = label in rejected_labels
        ax.text(
            x,
            y,
            str(label),
            fontsize=fontsize,
            fontweight="bold",
            color="white",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="#CC79A7" if is_rejected else ("#009E73" if is_good else "#D55E00"),
                alpha=0.9,
                edgecolor="#7A2E5E" if is_rejected else ("#006D55" if is_good else "#8C2D04"),
                linewidth=1.8,
            ),
            ha="center",
            va="center",
        )

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    logger.info(
        "Filtered overlay saved -> %s (%d good, %d bad, %d rejected)",
        output_path,
        n_good,
        n_bad,
        n_rejected,
    )
    return output_path
