"""Batch inference: classify every cell in a segmented dataset."""

from __future__ import annotations

import logging
import re
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
NUCLEUS_MASK_SUFFIXES = ("_nucleus_mask.tif", "_nucleus_mask.tiff", "_nucleus_mask.npy")
IMAGE_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")
FROG_ID_RE = re.compile(r"^TIFF_AH_(\d+)_\d+$")


def _extract_frog_id(image_name: str) -> int | None:
    """Extract frog id from image name, e.g. TIFF_AH_001_04 -> 1."""
    m = FROG_ID_RE.match(str(image_name))
    if m is None:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _safe_ratio(numerator: float | int | None, denominator: float | int | None, ndigits: int = 4) -> float:
    """Division-safe ratio; returns NaN if inputs are missing/invalid/zero denominator."""
    if numerator is None or denominator is None:
        return np.nan
    num = float(numerator)
    den = float(denominator)
    if not np.isfinite(num) or not np.isfinite(den) or den == 0.0:
        return np.nan
    return round(num / den, ndigits)


def _frog_aggregate_columns(filtered_df: pd.DataFrame) -> list[str]:
    preferred_metrics = [
        "area_px",
        "area_um2",
        "major_axis_px",
        "minor_axis_px",
        "cell_axis_ratio",
        "major_axis_um",
        "minor_axis_um",
        "nucleus_area_px",
        "nucleus_major_axis_px",
        "nucleus_minor_axis_px",
        "nucleus_axis_ratio",
        "nc_ratio",
        "nucleus_area_um2",
        "nucleus_major_axis_um",
        "nucleus_minor_axis_um",
    ]
    numeric_metrics = [c for c in preferred_metrics if c in filtered_df.columns]
    if not numeric_metrics:
        numeric_metrics = [
            c for c in filtered_df.columns
            if pd.api.types.is_numeric_dtype(filtered_df[c]) and c not in {"frog_id", "mask_index"}
        ]
    out_cols = ["frog_id", "n_images", "n_cells"]
    for metric in numeric_metrics:
        out_cols.extend([f"{metric}_mean", f"{metric}_std"])
    return out_cols


def _build_frog_aggregated_metrics(filtered_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Aggregate per-cell filtered area metrics into one row per frog."""
    out_cols = _frog_aggregate_columns(filtered_df)
    if filtered_df.empty:
        return pd.DataFrame(columns=out_cols), []

    work = filtered_df.copy()
    if "frog_id" not in work.columns:
        work["frog_id"] = work["image_path"].map(_extract_frog_id)

    unparsed_images = sorted(
        work.loc[work["frog_id"].isna(), "image_path"].dropna().astype(str).unique().tolist()
    )
    work = work.loc[work["frog_id"].notna()].copy()
    if work.empty:
        return pd.DataFrame(columns=out_cols), unparsed_images

    work["frog_id"] = work["frog_id"].astype(int)
    grouped = work.groupby("frog_id", sort=True)

    if "mask_index" in work.columns:
        agg_df = grouped.agg(
            n_images=("image_path", "nunique"),
            n_cells=("mask_index", "count"),
        )
    else:
        agg_df = grouped.agg(
            n_images=("image_path", "nunique"),
            n_cells=("image_path", "size"),
        )

    expected_cols = _frog_aggregate_columns(work)
    numeric_metrics = [c[:-5] for c in expected_cols if c.endswith("_mean")]
    for metric in numeric_metrics:
        agg_df[f"{metric}_mean"] = grouped[metric].mean()
        agg_df[f"{metric}_std"] = grouped[metric].std(ddof=1)

    agg_df = agg_df.reset_index()
    return agg_df[expected_cols], unparsed_images


def _write_frog_aggregated_metrics(filtered_df: pd.DataFrame, output_path: Path) -> Path:
    """Write per-frog aggregated erythrocyte metrics CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frog_df, unparsed = _build_frog_aggregated_metrics(filtered_df)
    if unparsed:
        sample = ", ".join(unparsed[:5])
        logger.warning(
            "Could not parse frog_id for %d image(s); excluded from frog aggregation. Examples: %s",
            len(unparsed),
            sample,
        )

    frog_df.to_csv(output_path, index=False)
    logger.info("Frog aggregated metrics CSV -> %s (%d frogs)", output_path, len(frog_df))
    return output_path


def _load_mask(image_folder: Path, image_stem: str) -> np.ndarray | None:
    """Try to load a membrane mask from the per-image folder."""
    for suffix in MASK_SUFFIXES:
        mp = image_folder / f"{image_stem}{suffix}"
        if mp.is_file():
            if mp.suffix == ".npy":
                return np.load(str(mp))
            else:
                import tifffile
                return tifffile.imread(str(mp))
    return None


def _load_nucleus_mask(image_folder: Path, image_stem: str) -> np.ndarray | None:
    """Try to load a nucleus mask from the per-image folder."""
    for suffix in NUCLEUS_MASK_SUFFIXES:
        mp = image_folder / f"{image_stem}{suffix}"
        if mp.is_file():
            if mp.suffix == ".npy":
                return np.load(str(mp))
            else:
                import tifffile
                return tifffile.imread(str(mp))
    return None


def match_nuclei_to_cells(
    cell_masks: np.ndarray,
    nuc_masks: np.ndarray,
) -> dict[int, int | None]:
    """Map each cell label to its best-matching nucleus label.

    For each cell, the nucleus with the largest pixel overlap inside that
    cell is selected. Cells with no overlapping nucleus map to ``None``.
    When multiple nuclei overlap a cell, only the largest is kept.
    """
    matches: dict[int, int | None] = {}
    for cell_label in np.unique(cell_masks):
        if cell_label == 0:
            continue
        nuc_in_cell = nuc_masks[cell_masks == cell_label]
        nuc_in_cell = nuc_in_cell[nuc_in_cell > 0]
        if len(nuc_in_cell) == 0:
            matches[cell_label] = None
            continue
        values, counts = np.unique(nuc_in_cell, return_counts=True)
        matches[cell_label] = int(values[counts.argmax()])
    return matches


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
        use_mlp_head=bool(ckpt.get("use_mlp_head", False)),
        use_efficient_probing=bool(ckpt.get("use_efficient_probing", False)),
        efficient_probing_cfg=dict(ckpt.get("efficient_probing", {})),
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
    selective_rejection_enabled: bool = False,
    selective_t_bad: float = 0.09,
    selective_t_good: float = 0.51,
) -> list[dict]:
    """Classify a list of (label, crop_image) tuples.

    Returns a list of dicts with keys:
    ``mask_index``, ``predicted_verdict``, ``confidence``, ``accepted``.
    """
    if not crops:
        return []
    if selective_rejection_enabled and not (0.0 <= selective_t_bad < selective_t_good <= 1.0):
        raise ValueError(
            "Invalid selective rejection thresholds: require 0 <= t_bad < t_good <= 1; "
            f"got t_bad={selective_t_bad}, t_good={selective_t_good}"
        )

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
            if selective_rejection_enabled:
                if prob <= selective_t_bad:
                    verdict = "bad"
                    accepted = True
                elif prob >= selective_t_good:
                    verdict = "good"
                    accepted = True
                else:
                    verdict = "rejected"
                    accepted = False
            else:
                if prob >= confidence_threshold:
                    verdict = "good"
                else:
                    verdict = "bad"
                accepted = True

            results.append({
                "mask_index": label,
                "predicted_verdict": verdict,
                "confidence": round(prob, 4),
                "accepted": accepted,
            })

    return results


def _resolve_selective_rejection_cfg(cfg: Any) -> tuple[bool, float, float]:
    """Read and validate selective-rejection settings from classifier config."""
    sel_cfg = getattr(cfg, "selective_rejection", None)
    enabled = bool(getattr(sel_cfg, "enabled", False)) if sel_cfg is not None else False
    t_bad = float(getattr(sel_cfg, "t_bad", 0.09)) if sel_cfg is not None else 0.09
    t_good = float(getattr(sel_cfg, "t_good", 0.51)) if sel_cfg is not None else 0.51
    if enabled and not (0.0 <= t_bad < t_good <= 1.0):
        raise ValueError(
            "Invalid classifier.selective_rejection thresholds: require 0 <= t_bad < t_good <= 1; "
            f"got t_bad={t_bad}, t_good={t_good}"
        )
    return enabled, t_bad, t_good


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
    selective_enabled, selective_t_bad, selective_t_good = _resolve_selective_rejection_cfg(cfg)
    padding_pct = float(cfg.crop_padding_pct)
    mask_bg = bool(cfg.mask_background)
    batch_size = int(cfg.batch_size)

    if selective_enabled:
        logger.info(
            "Selective rejection enabled: t_bad=%.3f, t_good=%.3f "
            "(classifier.confidence_threshold=%.3f is ignored)",
            selective_t_bad,
            selective_t_good,
            confidence_threshold,
        )

    all_images = _find_processed_images(data_dir)
    logger.info("Found %d processed images for inference", len(all_images))

    all_rows: list[dict] = []
    unparsed_frog_images: set[str] = set()

    for img_path in all_images:
        image_stem = img_path.stem
        image_folder = img_path.parent
        frog_id = _extract_frog_id(image_stem)
        if frog_id is None and image_stem not in unparsed_frog_images:
            unparsed_frog_images.add(image_stem)
            logger.warning("Could not parse frog_id from image name '%s'", image_stem)

        mask = _load_mask(image_folder, image_stem)
        if mask is None:
            logger.warning("No mask found for %s, skipping", img_path)
            continue

        img_rgb = _read_image_rgb(img_path)
        crops = extract_all_crops(img_rgb, mask, crop_size, padding_pct, mask_bg)

        if not crops:
            logger.warning("No cells in mask for %s", img_path)
            continue

        preds = classify_cells(
            model,
            crops,
            transform,
            device,
            confidence_threshold=confidence_threshold,
            batch_size=batch_size,
            selective_rejection_enabled=selective_enabled,
            selective_t_bad=selective_t_bad,
            selective_t_good=selective_t_good,
        )
        for p in preds:
            p["image_path"] = image_stem
            p["frog_id"] = frog_id
            all_rows.append(p)

        n_good = sum(1 for p in preds if p["predicted_verdict"] == "good")
        n_bad = sum(1 for p in preds if p["predicted_verdict"] == "bad")
        n_rejected = sum(1 for p in preds if p["predicted_verdict"] == "rejected")
        logger.info(
            "%s: %d cells -> %d good, %d bad, %d rejected",
            image_stem,
            len(preds),
            n_good,
            n_bad,
            n_rejected,
        )

    df = pd.DataFrame(
        all_rows,
        columns=["image_path", "mask_index", "predicted_verdict", "confidence", "accepted", "frog_id"],
    )

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

    When a nucleus mask (``<stem>_nucleus_mask.*``) is found alongside
    the membrane mask, nucleus measurements (area, major/minor axis,
    N/C ratio) are automatically added.
    """
    from skimage.measure import regionprops_table

    from cell_size.metadata import resolve_pixel_scale

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    good_df = predictions_df[predictions_df["predicted_verdict"] == "good"]
    records: list[dict] = []
    has_any_um = False
    has_any_nucleus = False
    unparsed_frog_images: set[str] = set()

    for image_name, group in good_df.groupby("image_path"):
        frog_id = _extract_frog_id(str(image_name))
        if frog_id is None and image_name not in unparsed_frog_images:
            unparsed_frog_images.add(str(image_name))
            logger.warning("Could not parse frog_id from image name '%s'", image_name)

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

        # Cell regionprops
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

        # Nucleus matching (if nucleus mask exists)
        nuc_mask = _load_nucleus_mask(image_folder, str(image_name))
        nuc_matches: dict[int, int | None] = {}
        nuc_props_lut: dict[int, dict] = {}
        if nuc_mask is not None:
            has_any_nucleus = True
            nuc_matches = match_nuclei_to_cells(mask, nuc_mask)
            nuc_props = regionprops_table(
                nuc_mask.astype(np.int32),
                properties=("label", "area", "major_axis_length", "minor_axis_length"),
            )
            for i in range(len(nuc_props["label"])):
                nuc_props_lut[int(nuc_props["label"][i])] = {
                    "area": int(nuc_props["area"][i]),
                    "major": float(nuc_props["major_axis_length"][i]),
                    "minor": float(nuc_props["minor_axis_length"][i]),
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
                "frog_id": frog_id,
                "mask_index": label,
                "area_px": area_px,
            }
            if pixel_to_um is not None:
                rec["area_um2"] = round(area_px * pixel_to_um**2, 4)

            if compute_diameters and major_px is not None:
                rec["major_axis_px"] = round(major_px, 2)
                rec["minor_axis_px"] = round(minor_px, 2)
                rec["cell_axis_ratio"] = _safe_ratio(major_px, minor_px)
                if pixel_to_um is not None:
                    rec["major_axis_um"] = round(major_px * pixel_to_um, 4)
                    rec["minor_axis_um"] = round(minor_px * pixel_to_um, 4)
            elif compute_diameters:
                rec["major_axis_px"] = np.nan
                rec["minor_axis_px"] = np.nan
                rec["cell_axis_ratio"] = np.nan

            # Nucleus columns
            if nuc_mask is not None:
                nuc_label = nuc_matches.get(label)
                if nuc_label is not None and nuc_label in nuc_props_lut:
                    np_info = nuc_props_lut[nuc_label]
                    rec["nucleus_area_px"] = np_info["area"]
                    rec["nucleus_major_axis_px"] = round(np_info["major"], 2)
                    rec["nucleus_minor_axis_px"] = round(np_info["minor"], 2)
                    rec["nucleus_axis_ratio"] = _safe_ratio(np_info["major"], np_info["minor"])
                    rec["nc_ratio"] = round(np_info["area"] / max(area_px, 1), 4)
                    if pixel_to_um is not None:
                        rec["nucleus_area_um2"] = round(np_info["area"] * pixel_to_um**2, 4)
                        rec["nucleus_major_axis_um"] = round(np_info["major"] * pixel_to_um, 4)
                        rec["nucleus_minor_axis_um"] = round(np_info["minor"] * pixel_to_um, 4)
                else:
                    rec["nucleus_area_px"] = None
                    rec["nucleus_major_axis_px"] = None
                    rec["nucleus_minor_axis_px"] = None
                    rec["nucleus_axis_ratio"] = np.nan
                    rec["nc_ratio"] = None

            records.append(rec)

    cols = ["image_path", "frog_id", "mask_index", "area_px"]
    if has_any_um:
        cols.append("area_um2")
    if compute_diameters:
        cols.extend(["major_axis_px", "minor_axis_px", "cell_axis_ratio"])
        if has_any_um:
            cols.extend(["major_axis_um", "minor_axis_um"])
    if has_any_nucleus:
        cols.extend([
            "nucleus_area_px",
            "nucleus_major_axis_px",
            "nucleus_minor_axis_px",
            "nucleus_axis_ratio",
            "nc_ratio",
        ])
        if has_any_um:
            cols.extend(["nucleus_area_um2", "nucleus_major_axis_um", "nucleus_minor_axis_um"])

    df = pd.DataFrame(records, columns=cols)
    df.to_csv(output_path, index=False)
    logger.info("Filtered areas CSV -> %s (%d good cells)", output_path, len(df))
    frog_agg_path = output_path.parent / "frog_aggregated_metrics.csv"
    _write_frog_aggregated_metrics(df, frog_agg_path)
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
