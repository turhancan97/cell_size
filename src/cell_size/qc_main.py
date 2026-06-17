"""Hydra entry-point for morphology QC filtering of filtered_areas.csv."""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from cell_size.classifier.morphology_qc import run_morphology_qc_from_csv

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="qc_filter", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Apply morphology QC to an existing filtered_areas.csv file."""
    logger.info("=== Morphology QC Filter ===")

    input_csv = Path(to_absolute_path(str(cfg.input_csv))).resolve()
    if not input_csv.is_file():
        logger.error("Input CSV not found: %s", input_csv)
        return

    output_dir = cfg.output_dir
    output_path = (
        Path(to_absolute_path(str(output_dir))).resolve()
        if output_dir is not None
        else input_csv.parent
    )

    if not bool(cfg.morphology_qc.enabled):
        logger.info("Morphology QC disabled; no outputs written")
        return

    result = run_morphology_qc_from_csv(input_csv, output_path, cfg.morphology_qc)
    logger.info(
        "QC complete: %d passed, %d rejected",
        len(result.clean_df),
        len(result.rejected_df),
    )
    logger.info("Output directory: %s", output_path)


if __name__ == "__main__":
    main()
