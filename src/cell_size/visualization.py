"""Optional visualisation: overlay images and cell-area histograms."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import find_boundaries

logger = logging.getLogger(__name__)


def generate_overlay(
    img: np.ndarray,
    masks: np.ndarray,
    output_path: str | Path,
    color: tuple[int, int, int] = (255, 0, 0),
) -> Path:
    """Draw segmentation outlines on the original image and save as PNG.

    Returns the output path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if img.ndim == 2:
        overlay = np.stack([img] * 3, axis=-1)
    else:
        overlay = img.copy()

    if overlay.dtype != np.uint8:
        if overlay.max() <= 1.0:
            overlay = (overlay * 255).astype(np.uint8)
        else:
            overlay = overlay.astype(np.uint8)

    outlines = find_boundaries(masks, mode="inner")
    out_y, out_x = np.nonzero(outlines)
    overlay[out_y, out_x] = color

    plt.figure(figsize=(12, 8))
    plt.imshow(overlay)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()

    logger.info("Overlay saved -> %s", output_path)
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
