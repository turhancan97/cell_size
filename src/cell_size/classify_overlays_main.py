"""Hydra entry-point for regenerating classifier overlays from saved predictions."""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from cell_size.classifier.inference import generate_filtered_overlays_from_predictions

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="classify_overlays", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Regenerate filtered overlays without rerunning classifier inference."""
    logger.info("=== Cell Quality Classifier - Overlay Regeneration ===")

    if cfg.data_dir is None:
        logger.error("No data directory provided. Set data_dir=/path/to/segmented/data")
        return

    data_dir = Path(to_absolute_path(str(cfg.data_dir))).resolve()
    output_dir = Path(to_absolute_path(str(cfg.output_dir))).resolve()
    predictions_csv = cfg.predictions_csv or (output_dir / "predictions.csv")
    predictions_csv = Path(to_absolute_path(str(predictions_csv))).resolve()

    if not predictions_csv.is_file():
        logger.error("Predictions CSV not found: %s", predictions_csv)
        return

    predictions_df = pd.read_csv(predictions_csv)
    overlays_dir = generate_filtered_overlays_from_predictions(data_dir, predictions_df, output_dir)
    logger.info("Filtered overlays written to %s", overlays_dir)
    logger.info("=== Overlay regeneration complete ===")


if __name__ == "__main__":
    main()
