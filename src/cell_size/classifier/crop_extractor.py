"""Extract individual cell crops from segmented images using feedback labels."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import pandas as pd
import tifffile
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

MASK_SUFFIXES = ("_mask.tif", "_mask.tiff", "_mask.npy")
IMAGE_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")


def _find_image_and_mask(
    data_dir: Path,
    image_name: str,
    dataset: str | None = None,
) -> tuple[Path, np.ndarray]:
    """Locate source image and load its mask from the cell-size output tree.

    The cell-size pipeline organises output as::

        data_dir / [dataset] / image_name / image_name.<ext>
        data_dir / [dataset] / image_name / image_name_mask.<ext>

    ``dataset`` may map to a subdirectory or be absent.
    """
    candidates: list[Path] = []
    if dataset:
        candidates.append(data_dir / dataset / image_name)
    candidates.append(data_dir / image_name)
    candidates.extend(data_dir.rglob(image_name))

    img_path: Path | None = None
    mask_arr: np.ndarray | None = None

    for folder in candidates:
        if not folder.is_dir():
            continue
        for ext in IMAGE_EXTENSIONS:
            p = folder / f"{image_name}{ext}"
            if p.is_file():
                img_path = p
                break
        if img_path is None:
            continue

        for suffix in MASK_SUFFIXES:
            mp = folder / f"{image_name}{suffix}"
            if mp.is_file():
                if mp.suffix == ".npy":
                    mask_arr = np.load(str(mp))
                else:
                    mask_arr = tifffile.imread(str(mp))
                break
        if mask_arr is not None:
            break

    if img_path is None or mask_arr is None:
        raise FileNotFoundError(
            f"Could not find image+mask for '{image_name}' "
            f"(dataset={dataset}) under {data_dir}"
        )

    return img_path, mask_arr


def _read_image_rgb(path: Path) -> np.ndarray:
    """Read an image and return as uint8 RGB."""
    if path.suffix.lower() in {".tif", ".tiff"}:
        img = tifffile.imread(str(path))
    else:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"Failed to read image: {path}")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.ndim == 3 and img.shape[2] > 3:
        img = img[:, :, :3]

    img = img.astype(np.float64)
    lo, hi = img.min(), img.max()
    if hi - lo > 0:
        img = (img - lo) / (hi - lo)
    return (img * 255).astype(np.uint8)


def _crop_cell(
    img: np.ndarray,
    mask: np.ndarray,
    label: int,
    padding_pct: float = 0.2,
    mask_background: bool = False,
) -> np.ndarray:
    """Extract a padded bounding-box crop for a single cell.

    Returns the cropped RGB image (H, W, 3) as uint8.
    """
    ys, xs = np.where(mask == label)
    if len(ys) == 0:
        raise ValueError(f"Label {label} not found in mask")

    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())

    h = y_max - y_min + 1
    w = x_max - x_min + 1
    pad_y = int(h * padding_pct)
    pad_x = int(w * padding_pct)

    y0 = max(0, y_min - pad_y)
    y1 = min(mask.shape[0], y_max + pad_y + 1)
    x0 = max(0, x_min - pad_x)
    x1 = min(mask.shape[1], x_max + pad_x + 1)

    crop = img[y0:y1, x0:x1].copy()

    if mask_background:
        cell_mask = mask[y0:y1, x0:x1] == label
        crop[~cell_mask] = 0

    return crop


def extract_crops(
    feedback_df: pd.DataFrame,
    data_dir: Path,
    crops_dir: Path,
    cfg: Any,
) -> int:
    """Extract cell crops from images+masks and save to ``crops_dir``.

    Creates the folder structure::

        crops_dir/good/<image_path>_cell_<mask_index>.<fmt>
        crops_dir/bad/<image_path>_cell_<mask_index>.<fmt>

    Returns the number of crops saved.
    """
    crop_size = int(cfg.crop_size)
    padding_pct = float(cfg.crop_padding_pct)
    mask_bg = bool(cfg.mask_background)
    fmt = cfg.crop_format.lower()
    if fmt not in ("png", "jpg", "jpeg"):
        fmt = "png"

    for cls in ("good", "bad"):
        (crops_dir / cls).mkdir(parents=True, exist_ok=True)

    grouped = feedback_df.groupby("image_path")
    saved = 0
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for image_name, group in grouped:
        dataset = group["dataset"].iloc[0] if "dataset" in group.columns else None
        if image_name not in cache:
            try:
                img_path, mask_arr = _find_image_and_mask(data_dir, image_name, dataset)
                img_rgb = _read_image_rgb(img_path)
                cache[image_name] = (img_rgb, mask_arr)
            except FileNotFoundError:
                logger.warning("Skipping image '%s': not found in %s", image_name, data_dir)
                continue

        img_rgb, mask_arr = cache[image_name]

        for _, row in group.iterrows():
            label = int(row["mask_index"])
            verdict = str(row["verdict"]).strip().lower()
            if verdict not in ("good", "bad"):
                logger.warning("Unknown verdict '%s' for %s cell %d, skipping", verdict, image_name, label)
                continue

            try:
                crop = _crop_cell(img_rgb, mask_arr, label, padding_pct, mask_bg)
            except ValueError:
                logger.warning("Label %d not in mask for '%s', skipping", label, image_name)
                continue

            crop_resized = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
            fname = f"{image_name}_cell_{label}.{fmt}"
            out_path = crops_dir / verdict / fname

            if fmt in ("jpg", "jpeg"):
                cv2.imwrite(str(out_path), cv2.cvtColor(crop_resized, cv2.COLOR_RGB2BGR))
            else:
                cv2.imwrite(str(out_path), cv2.cvtColor(crop_resized, cv2.COLOR_RGB2BGR))

            saved += 1

    logger.info("Extracted %d cell crops -> %s", saved, crops_dir)
    return saved


def extract_all_crops(
    img: np.ndarray,
    mask: np.ndarray,
    crop_size: int = 224,
    padding_pct: float = 0.2,
    mask_background: bool = False,
) -> list[tuple[int, np.ndarray]]:
    """Extract crops for ALL cells in a mask (used during inference).

    Returns a list of ``(label, crop_resized)`` tuples.
    """
    labels = np.unique(mask)
    labels = labels[labels > 0]
    crops: list[tuple[int, np.ndarray]] = []

    for label in labels:
        try:
            crop = _crop_cell(img, mask, int(label), padding_pct, mask_background)
            crop_resized = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
            crops.append((int(label), crop_resized))
        except ValueError:
            continue

    return crops


def split_dataset(
    crops_dir: Path,
    split_ratio: Sequence[float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> dict[str, int]:
    """Split crops into train/val/test using stratified sampling.

    Reads from ``crops_dir/good/`` and ``crops_dir/bad/``, then creates
    ``crops_dir/train/{good,bad}/``, ``crops_dir/val/{good,bad}/``, and
    ``crops_dir/test/{good,bad}/`` directories and moves the files.

    Returns counts per split.
    """
    all_files: list[Path] = []
    all_labels: list[int] = []

    for cls_idx, cls_name in enumerate(("good", "bad")):
        cls_dir = crops_dir / cls_name
        if not cls_dir.is_dir():
            continue
        for f in cls_dir.iterdir():
            if f.is_file() and f.suffix.lower() in (".png", ".jpg", ".jpeg"):
                all_files.append(f)
                all_labels.append(cls_idx)

    if len(all_files) == 0:
        raise RuntimeError(f"No crop files found in {crops_dir}")

    train_ratio, val_ratio, test_ratio = split_ratio
    files_arr = np.array(all_files, dtype=object)
    labels_arr = np.array(all_labels)

    train_files, rest_files, train_labels, rest_labels = train_test_split(
        files_arr,
        labels_arr,
        train_size=train_ratio,
        stratify=labels_arr,
        random_state=seed,
    )

    val_relative = val_ratio / (val_ratio + test_ratio)
    val_files, test_files, _, _ = train_test_split(
        rest_files,
        rest_labels,
        train_size=val_relative,
        stratify=rest_labels,
        random_state=seed,
    )

    counts: dict[str, int] = {}
    class_names = {0: "good", 1: "bad"}

    for split_name, file_list in [("train", train_files), ("val", val_files), ("test", test_files)]:
        count = 0
        for f in file_list:
            f = Path(f)
            cls_name = class_names[all_labels[all_files.index(f)]]
            dest_dir = crops_dir / split_name / cls_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), str(dest_dir / f.name))
            count += 1
        counts[split_name] = count

    for cls_name in ("good", "bad"):
        cls_dir = crops_dir / cls_name
        if cls_dir.is_dir() and not any(cls_dir.iterdir()):
            cls_dir.rmdir()

    logger.info(
        "Dataset split: train=%d, val=%d, test=%d",
        counts.get("train", 0),
        counts.get("val", 0),
        counts.get("test", 0),
    )
    return counts
