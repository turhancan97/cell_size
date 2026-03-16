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


def _normalize99(img: np.ndarray) -> np.ndarray:
    """Percentile 1-99 normalization, clipped to [0, 1]."""
    x = img.astype(np.float64)
    p1 = np.percentile(x, 1)
    p99 = np.percentile(x, 99)
    x = (x - p1) / (p99 - p1 + 1e-10)
    return np.clip(x, 0, 1)


def _remap_channels(img: np.ndarray, channel_map: list) -> np.ndarray:
    """Build a 3-channel image from *img* according to *channel_map*.

    ``channel_map`` is a list of up to 3 source-channel indices (or None).
    For example ``[None, 1, 2]`` places source channels 1 and 2 into
    positions 0 and 1 of the output, leaving the rest as zeros.
    ``[0, 0, 0]`` fills all three output channels with source channel 0.
    """
    if img.ndim == 2:
        logger.debug("Grayscale image -- skipping channel remap")
        return img

    src_channels = [c for c in channel_map if c is not None]
    if not src_channels:
        return img

    out = np.zeros((*img.shape[:2], 3), dtype=img.dtype)
    out_idx = 0
    for c in channel_map:
        if c is None:
            continue
        if out_idx >= 3:
            break
        if c >= img.shape[-1]:
            logger.warning(
                "channel_map references channel %d but image only has %d channels; skipping",
                c,
                img.shape[-1],
            )
            continue
        out[:, :, out_idx] = img[:, :, c]
        out_idx += 1

    logger.debug("Channel remap %s -> output has %d filled channels", channel_map, out_idx)
    return out


def _apply_channel_threshold(
    img: np.ndarray, channel_idx: int, threshold: float
) -> np.ndarray:
    """Clip values in *channel_idx* that exceed *threshold* to 1.0.

    Operates on a float [0,1] image. Used to suppress bright nuclei
    in the DAPI/Red channel before segmentation.
    """
    img = img.copy()
    img[:, :, channel_idx] = np.where(
        img[:, :, channel_idx] > threshold, 1.0, img[:, :, channel_idx]
    )
    logger.debug("Applied threshold %.2f on channel %d", threshold, channel_idx)
    return img


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

        Pipeline:
        1. Resize (optional)
        2. normalize99 (percentile 1-99 to [0,1])
        3. Channel threshold (optional, e.g. for nucleus)
        4. Channel remap via ``channel_map``
        5. ``model.eval``
        6. Rescale masks back to original resolution
        """
        resize_val = int(seg_cfg.resize)
        resized_img, original_hw = _resize_image(img, resize_val)

        normalized = _normalize99(resized_img)

        if (
            seg_cfg.threshold_channel is not None
            and normalized.ndim == 3
        ):
            normalized = _apply_channel_threshold(
                normalized,
                int(seg_cfg.threshold_channel),
                float(seg_cfg.threshold_value),
            )

        channel_map = list(seg_cfg.channel_map)
        if normalized.ndim >= 3:
            prepared = _remap_channels(normalized, channel_map)
        else:
            prepared = normalized

        tile_norm = int(seg_cfg.tile_norm_blocksize)
        normalize_param: dict | bool = True
        if tile_norm > 0:
            normalize_param = {"tile_norm_blocksize": tile_norm}

        niter = int(seg_cfg.niter) if seg_cfg.niter else None

        masks, flows, styles = self.model.eval(
            prepared,
            batch_size=self.batch_size,
            flow_threshold=float(seg_cfg.flow_threshold),
            cellprob_threshold=float(seg_cfg.cellprob_threshold),
            min_size=int(seg_cfg.min_cell_size),
            normalize=normalize_param,
            niter=niter,
        )

        masks = np.asarray(masks, dtype=np.uint16)

        if original_hw != (resized_img.shape[0], resized_img.shape[1]):
            masks = _rescale_masks(masks, original_hw)
            logger.debug("Rescaled masks back to %dx%d", original_hw[1], original_hw[0])

        n_cells = int(masks.max())
        logger.info(
            "Segmented %d cells  (resize=%d, target=%s, channel_map=%s)",
            n_cells,
            resize_val,
            seg_cfg.target,
            channel_map,
        )
        return masks
