"""Hydra entry-point that orchestrates batch segmentation."""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from cell_size.csv_writer import CatalogCSV, make_catalog_row, write_cell_areas_csv
from cell_size.io_utils import (
    get_relative_path,
    is_already_processed,
    organize_image_folder,
    read_image,
    save_mask,
    scan_images,
)
from cell_size.metadata import resolve_pixel_scale
from cell_size.segmenter import Segmenter
from cell_size.visualization import generate_area_histogram, generate_overlay

logger = logging.getLogger(__name__)


def _process_single_image(
    image_path: Path,
    data_dir: Path,
    segmenter: Segmenter,
    cfg: DictConfig,
    catalog: CatalogCSV,
) -> None:
    """Full pipeline for one image: read -> segment -> organise -> save."""
    logger.info("Processing: %s", image_path)

    channels = list(cfg.data.channels) if cfg.data.channels is not None else None
    img = read_image(image_path, channels=channels)

    masks = segmenter.segment(img, cfg.segmentation)

    if int(masks.max()) == 0:
        logger.warning("No cells detected in %s, skipping", image_path.name)
        return

    folder = organize_image_folder(image_path, data_dir)

    new_image_path = folder / image_path.name
    mask_path = save_mask(
        masks,
        folder / (image_path.stem + "_mask"),
        mask_format=cfg.output.mask_format,
    )

    pixel_scale = resolve_pixel_scale(new_image_path, cfg.data.pixel_to_um)

    if cfg.output.compute_cell_areas and pixel_scale is not None:
        areas_csv = folder / (image_path.stem + "_areas.csv")
        write_cell_areas_csv(masks, pixel_scale, areas_csv)

    if cfg.output.generate_overlays:
        overlay_path = folder / (image_path.stem + "_overlay.png")
        img_for_overlay = read_image(new_image_path, channels=channels)
        generate_overlay(img_for_overlay, masks, overlay_path)

    if cfg.output.generate_plots:
        hist_path = folder / (image_path.stem + "_histogram.png")
        generate_area_histogram(masks, hist_path, pixel_scale=pixel_scale)

    resize_val = int(cfg.segmentation.resize)
    rel_path = get_relative_path(folder, data_dir)
    row = make_catalog_row(rel_path, new_image_path, mask_path, resize_val, cfg.cell_type)
    catalog.add(row)


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Batch cell segmentation driven by Hydra configuration."""
    logger.info("=== Cell Size Estimator ===")
    logger.info("Segmentation target: %s", cfg.segmentation.target)
    logger.info("Data directory: %s", cfg.data.data_dir)

    data_dir = Path(cfg.data.data_dir).resolve()
    force = bool(cfg.force)

    images = scan_images(data_dir, cfg.data.file_types, cfg.data.recursive)
    if not images:
        logger.error("No images found -- check data.data_dir and data.file_types")
        return

    segmenter = Segmenter(cfg.model)
    catalog = CatalogCSV()

    skipped = 0
    processed = 0
    failed = 0

    for image_path in images:
        if not force and is_already_processed(image_path, cfg.output.mask_format):
            logger.info("Skipping (already processed): %s", image_path.name)
            skipped += 1
            continue

        try:
            _process_single_image(image_path, data_dir, segmenter, cfg, catalog)
            processed += 1
        except Exception:
            logger.exception("Failed to process %s", image_path.name)
            failed += 1

    csv_path = Path(cfg.output.csv_path)
    if not csv_path.is_absolute():
        csv_path = data_dir / csv_path

    if len(catalog) > 0:
        catalog.write(csv_path)

    logger.info(
        "=== Done: %d processed, %d skipped, %d failed ===",
        processed,
        skipped,
        failed,
    )


if __name__ == "__main__":
    main()
