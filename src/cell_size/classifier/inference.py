"""Batch inference: classify every cell in a segmented dataset."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms

from cell_size.classifier.crop_extractor import (
    _read_image_rgb,
    extract_all_crops,
)
from cell_size.classifier.dataset import IMAGENET_MEAN, IMAGENET_STD
from cell_size.classifier.models import build_model

logger = logging.getLogger(__name__)

MASK_SUFFIXES = ("_mask.tif", "_mask.tiff", "_mask.npy")
IMAGE_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")


def _load_mask(image_folder: Path, image_stem: str) -> np.ndarray | None:
    """Try to load a mask from the per-image folder."""
    for suffix in MASK_SUFFIXES:
        mp = image_folder / f"{image_stem}{suffix}"
        if mp.is_file():
            if mp.suffix == ".npy":
                return np.load(str(mp))
            else:
                import tifffile
                return tifffile.imread(str(mp))
    return None


def _build_inference_transform(crop_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_checkpoint(checkpoint_path: str | Path, device: torch.device) -> tuple[nn.Module, dict]:
    """Load a saved model checkpoint and return (model, metadata)."""
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    model = build_model(
        encoder=ckpt["encoder"],
        pretrained=False,
        freeze_encoder=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info(
        "Loaded checkpoint: encoder=%s, val_f1=%.4f, epoch=%d",
        ckpt["encoder"],
        ckpt.get("val_f1", 0),
        ckpt.get("epoch", 0),
    )
    return model, ckpt


def classify_cells(
    model: nn.Module,
    crops: list[tuple[int, np.ndarray]],
    transform: transforms.Compose,
    device: torch.device,
    confidence_threshold: float = 0.7,
    batch_size: int = 32,
) -> list[dict]:
    """Classify a list of (label, crop_image) tuples.

    Returns a list of dicts with keys: mask_index, predicted_verdict, confidence.
    """
    if not crops:
        return []

    results: list[dict] = []
    model.eval()

    for i in range(0, len(crops), batch_size):
        batch_crops = crops[i : i + batch_size]
        tensors = []
        for _, crop in batch_crops:
            t = transform(crop.copy())
            tensors.append(t)

        batch_tensor = torch.stack(tensors).to(device)
        with torch.no_grad():
            logits = model(batch_tensor)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()

        for j, (label, _) in enumerate(batch_crops):
            prob = float(probs[j])
            if prob >= confidence_threshold:
                verdict = "good"
            else:
                verdict = "bad"

            results.append({
                "mask_index": label,
                "predicted_verdict": verdict,
                "confidence": round(prob, 4),
            })

    return results


def run_inference(
    data_dir: Path,
    checkpoint_path: Path,
    output_dir: Path,
    cfg: Any,
) -> pd.DataFrame:
    """Run batch inference on all images in ``data_dir``.

    Returns a DataFrame of all predictions.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if cfg.gpu and torch.cuda.is_available() else "cpu")

    model, ckpt = load_checkpoint(checkpoint_path, device)
    crop_size = ckpt.get("crop_size", int(cfg.crop_size))
    transform = _build_inference_transform(crop_size)

    confidence_threshold = float(cfg.confidence_threshold)
    padding_pct = float(cfg.crop_padding_pct)
    mask_bg = bool(cfg.mask_background)
    batch_size = int(cfg.batch_size)

    all_images = _find_processed_images(data_dir)
    logger.info("Found %d processed images for inference", len(all_images))

    all_rows: list[dict] = []

    for img_path in all_images:
        image_stem = img_path.stem
        image_folder = img_path.parent

        mask = _load_mask(image_folder, image_stem)
        if mask is None:
            logger.warning("No mask found for %s, skipping", img_path)
            continue

        img_rgb = _read_image_rgb(img_path)
        crops = extract_all_crops(img_rgb, mask, crop_size, padding_pct, mask_bg)

        if not crops:
            logger.warning("No cells in mask for %s", img_path)
            continue

        preds = classify_cells(model, crops, transform, device, confidence_threshold, batch_size)
        for p in preds:
            p["image_path"] = image_stem
            all_rows.append(p)

        n_good = sum(1 for p in preds if p["predicted_verdict"] == "good")
        logger.info(
            "%s: %d cells -> %d good, %d bad",
            image_stem,
            len(preds),
            n_good,
            len(preds) - n_good,
        )

    df = pd.DataFrame(all_rows, columns=["image_path", "mask_index", "predicted_verdict", "confidence"])

    csv_path = output_dir / "predictions.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Predictions CSV -> %s (%d cells)", csv_path, len(df))

    return df


def compute_filtered_areas(
    data_dir: Path,
    predictions_df: pd.DataFrame,
    output_path: Path,
    config_pixel_to_um: float | None = None,
    compute_diameters: bool = True,
) -> Path:
    """Compute cell areas (and optionally diameters) for 'good' cells.

    Pixel scale is resolved **per image**: first by auto-detecting from
    OME-TIFF / TIFF metadata, then falling back to ``config_pixel_to_um``.
    Different images may therefore have different scales.

    When ``compute_diameters`` is True, each cell's major and minor axis
    lengths are measured by fitting an ellipse to the cell region via
    ``skimage.measure.regionprops``.  These correspond to the long and
    short diameters of the cell.
    """
    from skimage.measure import regionprops_table

    from cell_size.metadata import resolve_pixel_scale

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    good_df = predictions_df[predictions_df["predicted_verdict"] == "good"]
    records: list[dict] = []
    has_any_um = False

    for image_name, group in good_df.groupby("image_path"):
        image_folder = _find_image_folder(data_dir, str(image_name))
        if image_folder is None:
            continue

        mask = _load_mask(image_folder, str(image_name))
        if mask is None:
            continue

        img_path = _find_source_image(image_folder, str(image_name))
        pixel_to_um = None
        if img_path is not None:
            pixel_to_um = resolve_pixel_scale(img_path, config_pixel_to_um)
        elif config_pixel_to_um is not None:
            pixel_to_um = config_pixel_to_um

        if pixel_to_um is not None:
            has_any_um = True

        props_lut: dict[int, dict] = {}
        if compute_diameters:
            props = regionprops_table(
                mask.astype(np.int32),
                properties=("label", "area", "major_axis_length", "minor_axis_length"),
            )
            for i in range(len(props["label"])):
                props_lut[int(props["label"][i])] = {
                    "area": int(props["area"][i]),
                    "major": float(props["major_axis_length"][i]),
                    "minor": float(props["minor_axis_length"][i]),
                }

        for _, row in group.iterrows():
            label = int(row["mask_index"])

            if label in props_lut:
                area_px = props_lut[label]["area"]
                major_px = props_lut[label]["major"]
                minor_px = props_lut[label]["minor"]
            else:
                area_px = int((mask == label).sum())
                major_px = None
                minor_px = None

            rec: dict = {
                "image_path": image_name,
                "mask_index": label,
                "area_px": area_px,
            }
            if pixel_to_um is not None:
                rec["area_um2"] = round(area_px * pixel_to_um**2, 4)

            if compute_diameters and major_px is not None:
                rec["major_axis_px"] = round(major_px, 2)
                rec["minor_axis_px"] = round(minor_px, 2)
                if pixel_to_um is not None:
                    rec["major_axis_um"] = round(major_px * pixel_to_um, 4)
                    rec["minor_axis_um"] = round(minor_px * pixel_to_um, 4)

            records.append(rec)

    cols = ["image_path", "mask_index", "area_px"]
    if has_any_um:
        cols.append("area_um2")
    if compute_diameters:
        cols.extend(["major_axis_px", "minor_axis_px"])
        if has_any_um:
            cols.extend(["major_axis_um", "minor_axis_um"])

    df = pd.DataFrame(records, columns=cols)
    df.to_csv(output_path, index=False)
    logger.info("Filtered areas CSV -> %s (%d good cells)", output_path, len(df))
    return output_path


def _find_source_image(image_folder: Path, image_name: str) -> Path | None:
    """Locate the source image file inside a per-image folder."""
    for ext in IMAGE_EXTENSIONS:
        p = image_folder / f"{image_name}{ext}"
        if p.is_file():
            return p
    return None


def _find_processed_images(data_dir: Path) -> list[Path]:
    """Find all images in per-image folders that have a corresponding mask."""
    found: list[Path] = []
    for folder in sorted(data_dir.rglob("*")):
        if not folder.is_dir():
            continue
        stem = folder.name
        for ext in IMAGE_EXTENSIONS:
            img = folder / f"{stem}{ext}"
            if img.is_file():
                has_mask = any(
                    (folder / f"{stem}{s}").is_file() for s in MASK_SUFFIXES
                )
                if has_mask:
                    found.append(img)
                break
    return found


def _find_image_folder(data_dir: Path, image_name: str) -> Path | None:
    """Locate the per-image folder for a given image name."""
    for folder in data_dir.rglob(image_name):
        if folder.is_dir():
            return folder
    return None
