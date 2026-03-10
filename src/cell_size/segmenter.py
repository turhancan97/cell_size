"""Cellpose-SAM segmentation wrapper."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_CELLPOSE_INJECTED = False


def _ensure_cellpose_importable() -> None:
    """Add the vendored cellpose submodule to ``sys.path`` if needed."""
    global _CELLPOSE_INJECTED
    if _CELLPOSE_INJECTED:
        return

    repo_root = Path(__file__).resolve().parents[2]
    submodule = repo_root / "cellpose"
    if submodule.is_dir() and str(submodule) not in sys.path:
        sys.path.insert(0, str(submodule))
        logger.debug("Injected cellpose submodule into sys.path: %s", submodule)
    _CELLPOSE_INJECTED = True


def _resize_image(img: np.ndarray, resize: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Resize image so the longest side equals *resize*. Returns (resized, original_shape)."""
    original_hw = (img.shape[0], img.shape[1])
    h, w = original_hw

    if resize <= 0 or max(h, w) <= resize:
        return img, original_hw

    if h >= w:
        new_h = resize
        new_w = int(w / h * resize)
    else:
        new_w = resize
        new_h = int(h / w * resize)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    logger.debug("Resized %dx%d -> %dx%d", w, h, new_w, new_h)
    return resized, original_hw


def _rescale_masks(masks: np.ndarray, original_hw: tuple[int, int]) -> np.ndarray:
    """Scale masks back to the original image resolution using nearest-neighbour."""
    oh, ow = original_hw
    if masks.shape[0] == oh and masks.shape[1] == ow:
        return masks
    return cv2.resize(
        masks.astype(np.uint16), (ow, oh), interpolation=cv2.INTER_NEAREST
    ).astype(np.uint16)


class Segmenter:
    """Wraps ``cellpose.models.CellposeModel`` with project-specific defaults."""

    def __init__(self, model_cfg: Any) -> None:
        _ensure_cellpose_importable()
        from cellpose import models, core

        gpu_requested = bool(model_cfg.gpu)
        gpu_available = core.use_gpu()

        if gpu_requested and not gpu_available:
            logger.warning("GPU requested but not available -- falling back to CPU")
        use_gpu = gpu_requested and gpu_available

        pretrained = model_cfg.custom_model_path or model_cfg.model_type
        logger.info(
            "Loading Cellpose model=%s  gpu=%s  batch_size=%d",
            pretrained,
            use_gpu,
            model_cfg.batch_size,
        )
        self.model = models.CellposeModel(gpu=use_gpu, pretrained_model=pretrained)
        self.batch_size: int = int(model_cfg.batch_size)

    def segment(
        self,
        img: np.ndarray,
        seg_cfg: Any,
    ) -> np.ndarray:
        """Run segmentation on a single image.

        Handles optional resizing, runs ``model.eval``, rescales masks back
        to original resolution, and filters by ``min_cell_size``.
        """
        resize_val = int(seg_cfg.resize)
        resized_img, original_hw = _resize_image(img, resize_val)

        tile_norm = int(seg_cfg.tile_norm_blocksize)
        normalize_param: dict | bool = True
        if tile_norm > 0:
            normalize_param = {"tile_norm_blocksize": tile_norm}

        masks, flows, styles = self.model.eval(
            resized_img,
            batch_size=self.batch_size,
            channels=[int(seg_cfg.chan), int(seg_cfg.chan2)],
            flow_threshold=float(seg_cfg.flow_threshold),
            cellprob_threshold=float(seg_cfg.cellprob_threshold),
            min_size=int(seg_cfg.min_cell_size),
            normalize=normalize_param,
        )

        masks = np.asarray(masks, dtype=np.uint16)

        if original_hw != (resized_img.shape[0], resized_img.shape[1]):
            masks = _rescale_masks(masks, original_hw)
            logger.debug("Rescaled masks back to %dx%d", original_hw[1], original_hw[0])

        n_cells = int(masks.max())
        logger.info(
            "Segmented %d cells  (resize=%d, target=%s)",
            n_cells,
            resize_val,
            seg_cfg.target,
        )
        return masks
