"""Image I/O utilities: scanning, reading, mask saving, and folder organization."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import tifffile
from natsort import natsorted

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def scan_images(
    data_dir: str | Path,
    file_types: Sequence[str] = (".tif",),
    recursive: bool = True,
) -> list[Path]:
    """Find all image files matching *file_types* under *data_dir*.

    Returns naturally sorted list of absolute paths. Skips files whose names
    contain ``_mask`` to avoid picking up previously generated masks.
    """
    data_dir = Path(data_dir).resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    extensions = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in file_types}
    unknown = extensions - SUPPORTED_EXTENSIONS
    if unknown:
        logger.warning("Unsupported extensions requested (will still try): %s", unknown)

    found: list[Path] = []
    pattern_root = data_dir.rglob("*") if recursive else data_dir.glob("*")
    for p in pattern_root:
        if not p.is_file():
            continue
        if p.suffix.lower() not in extensions:
            continue
        if "_mask" in p.stem:
            continue
        found.append(p)

    found = natsorted(found, key=lambda p: str(p))
    logger.info("Found %d images in %s (recursive=%s)", len(found), data_dir, recursive)
    return found


def read_image(path: str | Path, channels: list[int] | None = None) -> np.ndarray:
    """Read an image from disk.

    For TIFF files uses ``tifffile``; for other formats uses OpenCV.
    If *channels* is provided, selects those channel indices from the last axis.
    Returns an array with shape (H, W) or (H, W, C).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    if path.suffix.lower() in {".tif", ".tiff"}:
        img = tifffile.imread(str(path))
    else:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"OpenCV failed to read image: {path}")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if channels is not None and img.ndim >= 3:
        img = img[..., channels]
        if img.shape[-1] == 1:
            img = img.squeeze(-1)

    logger.debug("Read image %s  shape=%s  dtype=%s", path.name, img.shape, img.dtype)
    return img


def save_mask(
    mask: np.ndarray,
    path: str | Path,
    mask_format: str = "tif",
) -> Path:
    """Save a segmentation mask as uint16 TIF or NumPy ``.npy`` file.

    Returns the path of the written file.
    """
    path = Path(path)
    mask_u16 = mask.astype(np.uint16)

    if mask_format == "npy":
        out = path.with_suffix(".npy")
        np.save(str(out), mask_u16)
    else:
        out = path.with_suffix(".tif")
        tifffile.imwrite(str(out), mask_u16, compression="zlib")

    logger.info("Saved mask -> %s  (cells=%d)", out, mask_u16.max())
    return out


def _is_in_own_folder(image_path: Path) -> bool:
    """True when the image already lives inside a folder named after itself."""
    return image_path.parent.name == image_path.stem


def organize_image_folder(image_path: Path, data_dir: Path) -> Path:
    """Create a per-image folder and move the source image into it.

    Given ``data_dir/projectA/image000.jpg``, creates
    ``data_dir/projectA/image000/`` and moves the image inside.
    Returns the new folder path.

    If the image is already in its named folder the function is a no-op.
    """
    if _is_in_own_folder(image_path):
        logger.debug("Image already in its folder: %s", image_path)
        return image_path.parent

    folder = image_path.parent / image_path.stem
    folder.mkdir(parents=True, exist_ok=True)

    dest = folder / image_path.name
    if not dest.exists():
        shutil.move(str(image_path), str(dest))
        logger.info("Moved %s -> %s", image_path.name, dest)
    else:
        logger.debug("Destination already exists, skipping move: %s", dest)

    return folder


def is_already_processed(image_path: Path, mask_format: str = "tif") -> bool:
    """Check whether a mask already exists for this image (for resume support).

    Handles both cases: image still in its original location, or already
    moved into a per-image folder.
    """
    mask_ext = ".npy" if mask_format == "npy" else ".tif"

    if _is_in_own_folder(image_path):
        mask_path = image_path.parent / (image_path.stem + "_mask" + mask_ext)
        return mask_path.is_file()

    folder = image_path.parent / image_path.stem
    mask_path = folder / (image_path.stem + "_mask" + mask_ext)
    return (folder / image_path.name).is_file() and mask_path.is_file()


def get_relative_path(folder: Path, data_dir: Path) -> str:
    """Return the POSIX-style relative path from *data_dir* to *folder*."""
    return folder.relative_to(data_dir).as_posix()
