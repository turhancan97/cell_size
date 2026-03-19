"""Hydra entry-point for cell quality classifier training."""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from cell_size.classifier.consensus import apply_majority_vote, load_and_merge
from cell_size.classifier.crop_extractor import extract_crops, split_dataset
from cell_size.classifier.experiment_tracking import (
    append_experiments_csv,
    build_record,
    write_run_results_csv,
)
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
    base_output_dir = Path(cfg.output_dir).resolve()
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # Use the (deterministic) run name provided by launcher / wandb config.
    run_name = str(cfg.classifier.wandb.run_name) if cfg.classifier.wandb.run_name else "run"
    run_dir = base_output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Keep crop datasets separate for different masking strategies so resume and
    # k-fold splits don't accidentally mix images.
    mask_bg = bool(cfg.classifier.mask_background)
    crops_dir = base_output_dir / cfg.classifier.crops_dir / (
        "mask_bg_true" if mask_bg else "mask_bg_false"
    )
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
    import datetime as dt  # noqa: PLC0415

    started_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

    status = "success"
    result = None
    try:
        result = train(crops_dir, run_dir, cfg.classifier)
    except Exception:
        status = "failed"
        logger.exception("Training failed")
        # Continue to write a failed record row.

    logger.info("Step 5/5: Generating evaluation artifacts")
    cm_path = None
    if result is not None and result.all_test_labels and result.all_test_preds:
        cm_path = run_dir / "confusion_matrix.png"
        plot_confusion_matrix(result.all_test_labels, result.all_test_preds, cm_path)

    logger.info("=== Training complete ===")
    if result is not None and result.best_checkpoint:
        logger.info("Best checkpoint: %s (val F1=%.4f)", result.best_checkpoint, result.best_val_f1)
    if result is not None and result.test_metrics:
        m = result.test_metrics
        logger.info(
            "Test metrics: acc=%.4f  prec=%.4f  rec=%.4f  f1=%.4f",
            m.accuracy,
            m.precision,
            m.recall,
            m.f1,
        )

    # Write per-run results.csv and append to central experiments.csv
    import os  # noqa: PLC0415
    slurm_job_id = os.environ.get("SLURM_JOB_ID")

    test_m = result.test_metrics if result is not None else None
    record = build_record(
        run_name=run_name,
        encoder=str(cfg.classifier.encoder),
        freeze_encoder=bool(cfg.classifier.freeze_encoder),
        learning_rate=float(cfg.classifier.learning_rate),
        confidence_threshold=float(cfg.classifier.confidence_threshold),
        seed=int(cfg.classifier.seed),
        cross_validation=bool(cfg.classifier.cross_validation.enabled),
        k_folds=int(cfg.classifier.cross_validation.k_folds) if cfg.classifier.cross_validation.enabled else None,
        best_val_f1=float(result.best_val_f1) if result is not None else None,
        test_accuracy=float(test_m.accuracy) if test_m is not None else None,
        test_precision=float(test_m.precision) if test_m is not None else None,
        test_recall=float(test_m.recall) if test_m is not None else None,
        test_f1=float(test_m.f1) if test_m is not None else None,
        best_checkpoint_path=str(result.best_checkpoint) if result is not None and result.best_checkpoint else None,
        confusion_matrix_path=str(cm_path) if cm_path is not None else None,
        slurm_job_id=slurm_job_id,
        status=status,
        started_at=started_at,
        finished_at=None,
    )

    try:
        write_run_results_csv(run_dir, record)
        append_experiments_csv(base_output_dir, record)
    except Exception:
        logger.exception("Failed to write results CSV(s)")

    logger.info("Run directory: %s", run_dir)
    logger.info("Central experiments CSV: %s", base_output_dir / "experiments.csv")


if __name__ == "__main__":
    main()
