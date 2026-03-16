"""Optional visualisation: overlay images and cell-area histograms."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import center_of_mass, find_objects
from skimage.segmentation import find_boundaries

logger = logging.getLogger(__name__)


def _to_display_rgb(img: np.ndarray) -> np.ndarray:
    """Normalise any image to uint8 RGB [0..255] suitable for matplotlib."""
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


def _mask_overlay(img_rgb: np.ndarray, masks: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blend random per-cell colours onto *img_rgb*.

    Each cell label gets a distinct colour; background is left unchanged.
    *img_rgb* is expected to be uint8 [0..255] (from ``_to_display_rgb``).
    """
    overlay = img_rgb.astype(np.float64) / 255.0
    n_cells = int(masks.max())
    if n_cells == 0:
        return img_rgb

    rng = np.random.RandomState(42)
    colors = rng.rand(n_cells + 1, 3)
    colors[0] = 0  # background gets no colour

    colour_map = colors[masks]
    blended = (1 - alpha) * overlay + alpha * colour_map
    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)


def generate_overlay(
    img: np.ndarray,
    masks: np.ndarray,
    output_path: str | Path,
    fontsize: int = 8,
) -> Path:
    """Draw coloured mask overlay with cell-number labels and save as PNG.

    Each cell gets a semi-transparent colour fill, boundary outlines, and a
    numbered label at its centroid.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    img_rgb = _to_display_rgb(img)
    overlay = _mask_overlay(img_rgb, masks)

    outlines = find_boundaries(masks, mode="inner")
    out_y, out_x = np.nonzero(outlines)
    overlay[out_y, out_x] = (255, 0, 0)

    slices = find_objects(masks)
    centroids: list[tuple[int, int]] = []
    labels: list[int] = []
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
        centroids.append((y_cent, x_cent))
        labels.append(mask_label)

    num_cells = int(masks.max())
    fig, ax = plt.subplots(figsize=(20, 15))
    ax.imshow(overlay)
    ax.set_title(f"Masks with labels (total cells: {num_cells})", fontsize=16)
    ax.axis("off")

    for (y, x), label in zip(centroids, labels):
        ax.text(
            x,
            y,
            str(label),
            fontsize=fontsize,
            fontweight="bold",
            color="white",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="black",
                alpha=0.7,
                edgecolor="white",
                linewidth=1.5,
            ),
            ha="center",
            va="center",
        )

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    logger.info("Overlay saved -> %s  (%d cells labelled)", output_path, len(labels))
    return output_path


def generate_area_histogram(
    masks: np.ndarray,
    output_path: str | Path,
    pixel_scale: Optional[float] = None,
) -> Path:
    """Plot a histogram of cell areas and save as PNG.

    If *pixel_scale* is provided, x-axis shows area in um^2; otherwise in pixels.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    unique_labels, counts = np.unique(masks, return_counts=True)
    cell_mask = unique_labels > 0
    areas = counts[cell_mask].astype(float)

    if pixel_scale is not None:
        areas *= pixel_scale**2
        unit = "µm²"
    else:
        unit = "pixels"

    if len(areas) == 0:
        logger.warning("No cells detected, skipping histogram")
        return output_path

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(areas, bins=min(50, max(10, len(areas) // 3)), edgecolor="black", alpha=0.75)
    ax.set_xlabel(f"Cell Area ({unit})")
    ax.set_ylabel("Count")
    ax.set_title(f"Cell Area Distribution  (n={len(areas)})")

    stats_text = (
        f"Mean: {np.mean(areas):.1f} {unit}\n"
        f"Median: {np.median(areas):.1f} {unit}\n"
        f"Std: {np.std(areas):.1f} {unit}"
    )
    ax.text(
        0.97,
        0.95,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close(fig)

    logger.info("Histogram saved -> %s", output_path)
    return output_path
