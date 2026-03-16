"""Hydra entry-point for batch cell quality classification (inference)."""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from cell_size.classifier.crop_extractor import _read_image_rgb
from cell_size.classifier.inference import (
    _find_processed_images,
    _load_mask,
    compute_filtered_areas,
    run_inference,
)
from cell_size.classifier.visualization import generate_filtered_overlay

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="classify", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Classify cells in a segmented dataset as good or bad.

    Pipeline: load model -> iterate images -> classify all cells ->
    write predictions CSV -> filtered areas -> filtered overlays.
    """
    logger.info("=== Cell Quality Classifier - Inference ===")

    if cfg.checkpoint is None:
        logger.error("No checkpoint provided. Set checkpoint=/path/to/best_model.pt")
        return
    if cfg.data_dir is None:
        logger.error("No data directory provided. Set data_dir=/path/to/segmented/data")
        return

    data_dir = Path(cfg.data_dir).resolve()
    output_dir = Path(cfg.output_dir).resolve()
    checkpoint_path = Path(cfg.checkpoint).resolve()

    logger.info("Step 1/3: Running batch inference")
    predictions_df = run_inference(data_dir, checkpoint_path, output_dir, cfg.classifier)

    if predictions_df.empty:
        logger.warning("No predictions generated. Check data_dir contents.")
        return

    n_good = (predictions_df["predicted_verdict"] == "good").sum()
    n_bad = (predictions_df["predicted_verdict"] == "bad").sum()
    logger.info("Inference complete: %d good, %d bad cells across all images", n_good, n_bad)

    if cfg.compute_filtered_areas:
        logger.info("Step 2/3: Computing filtered cell areas (good cells only)")
        config_pixel_to_um = float(cfg.pixel_to_um) if cfg.pixel_to_um is not None else None
        areas_path = output_dir / "filtered_areas.csv"
        diameters = bool(cfg.compute_diameters) if "compute_diameters" in cfg else True
        compute_filtered_areas(
            data_dir, predictions_df, areas_path, config_pixel_to_um, diameters,
        )
    else:
        logger.info("Step 2/3: Skipping filtered areas (disabled)")

    if cfg.generate_filtered_overlays:
        logger.info("Step 3/3: Generating filtered overlays")
        overlays_dir = output_dir / "overlays"
        overlays_dir.mkdir(parents=True, exist_ok=True)

        all_images = _find_processed_images(data_dir)
        for img_path in all_images:
            image_stem = img_path.stem
            img_preds = predictions_df[predictions_df["image_path"] == image_stem]
            if img_preds.empty:
                continue

            mask = _load_mask(img_path.parent, image_stem)
            if mask is None:
                continue

            img_rgb = _read_image_rgb(img_path)
            overlay_path = overlays_dir / f"{image_stem}_filtered_overlay.png"
            generate_filtered_overlay(img_rgb, mask, img_preds, overlay_path)
    else:
        logger.info("Step 3/3: Skipping filtered overlays (disabled)")

    logger.info("=== Inference complete ===")
    logger.info("Output directory: %s", output_dir)


if __name__ == "__main__":
    main()
