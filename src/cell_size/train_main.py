"""Hydra entry-point for cell quality classifier training."""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from cell_size.classifier.consensus import apply_majority_vote, load_and_merge
from cell_size.classifier.crop_extractor import extract_crops, split_dataset
from cell_size.classifier.trainer import train
from cell_size.classifier.visualization import plot_confusion_matrix

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Train a cell quality classifier.

    Pipeline: merge feedback CSVs -> consensus -> extract crops -> split ->
    train -> evaluate -> save checkpoint + confusion matrix.
    """
    logger.info("=== Cell Quality Classifier - Training ===")

    if not cfg.feedback_csvs:
        logger.error("No feedback CSVs provided. Set feedback_csvs=['/path/to/csv']")
        return
    if cfg.data_dir is None:
        logger.error("No data directory provided. Set data_dir=/path/to/segmented/data")
        return

    data_dir = Path(cfg.data_dir).resolve()
    output_dir = Path(cfg.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    crops_dir = output_dir / cfg.classifier.crops_dir
    splits_ready = all(
        (crops_dir / split / cls).is_dir() and any((crops_dir / split / cls).iterdir())
        for split in ("train", "val", "test")
        for cls in ("good", "bad")
    )

    if splits_ready:
        logger.info(
            "Steps 1-3/5: Skipping crop extraction & splitting "
            "(train/val/test directories already exist in %s)",
            crops_dir,
        )
    else:
        csv_paths = [Path(p) for p in cfg.feedback_csvs]

        logger.info("Step 1/5: Loading and merging %d feedback CSV(s)", len(csv_paths))
        merged_df = load_and_merge(csv_paths)
        merged_df = apply_majority_vote(merged_df)

        logger.info("Step 2/5: Extracting cell crops -> %s", crops_dir)
        n_crops = extract_crops(merged_df, data_dir, crops_dir, cfg.classifier)

        if n_crops == 0:
            logger.error("No crops extracted. Check data_dir and feedback CSV image_path values.")
            return

        split_ratio = list(cfg.classifier.split_ratio)
        seed = int(cfg.classifier.seed)
        logger.info("Step 3/5: Splitting dataset (ratio=%s, seed=%d)", split_ratio, seed)
        split_dataset(crops_dir, split_ratio, seed)

    logger.info("Step 4/5: Training classifier")
    result = train(crops_dir, output_dir, cfg.classifier)

    logger.info("Step 5/5: Generating evaluation artifacts")
    if result.all_test_labels and result.all_test_preds:
        cm_path = output_dir / "confusion_matrix.png"
        plot_confusion_matrix(result.all_test_labels, result.all_test_preds, cm_path)

    logger.info("=== Training complete ===")
    if result.best_checkpoint:
        logger.info("Best checkpoint: %s (val F1=%.4f)", result.best_checkpoint, result.best_val_f1)
    if result.test_metrics:
        m = result.test_metrics
        logger.info(
            "Test metrics: acc=%.4f  prec=%.4f  rec=%.4f  f1=%.4f",
            m.accuracy,
            m.precision,
            m.recall,
            m.f1,
        )


if __name__ == "__main__":
    main()
