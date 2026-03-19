"""Helpers for recording training run results to CSV.

This module is used by `cell-size-train` so that every run produces:
- a per-run `results.csv` inside `output_dir/<run_name>/`
- a central `experiments.csv` appended under the base output_dir

Because SLURM jobs can run concurrently, appending to the central CSV uses a
file lock (Linux `fcntl`).
"""

from __future__ import annotations

import csv
import datetime as dt
import os
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def write_run_results_csv(run_dir: Path, record: dict[str, Any]) -> Path:
    """Write a single-row `results.csv` in *run_dir*."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "results.csv"

    fieldnames = list(record.keys())
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerow(record)
    return out


def write_epoch_results_csv(run_dir: Path, rows: list[dict[str, Any]]) -> Path:
    """Write `epoch_results.csv` in *run_dir* (one row per epoch)."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "epoch_results.csv"

    if not rows:
        # Still write an empty file with a header for consistency.
        with out.open("w", newline="") as f:
            f.write("epoch\n")
        return out

    fieldnames: list[str] = []
    # Preserve key order from first row, then add any new keys encountered.
    for k in rows[0].keys():
        fieldnames.append(k)
    for r in rows[1:]:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return out


def append_experiments_csv(base_output_dir: Path, record: dict[str, Any]) -> Path:
    """Append a single row to `base_output_dir/experiments.csv` with file locking."""
    base_output_dir = Path(base_output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    out = base_output_dir / "experiments.csv"

    # Late import: fcntl is Linux/Unix only (fine for SLURM nodes).
    import fcntl  # noqa: PLC0415

    # We lock a dedicated lockfile to avoid portability issues with locking the CSV itself.
    lock_path = base_output_dir / ".experiments.csv.lock"
    lock_path.touch(exist_ok=True)

    fieldnames = list(record.keys())
    with lock_path.open("r+") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        try:
            file_exists = out.is_file()
            needs_header = True
            if file_exists and out.stat().st_size > 0:
                needs_header = False

            with out.open("a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                if needs_header:
                    w.writeheader()
                w.writerow(record)
        finally:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)

    return out


def build_record(
    *,
    run_name: str,
    encoder: str,
    freeze_encoder: bool,
    learning_rate: float,
    confidence_threshold: float,
    seed: int,
    cross_validation: bool,
    k_folds: int | None,
    best_val_f1: float | None,
    test_accuracy: float | None,
    test_precision: float | None,
    test_recall: float | None,
    test_f1: float | None,
    best_checkpoint_path: str | None,
    confusion_matrix_path: str | None,
    slurm_job_id: str | None,
    status: str,
    started_at: str | None = None,
    finished_at: str | None = None,
) -> dict[str, Any]:
    """Create a fixed-schema record for results tracking."""
    return {
        "run_name": run_name,
        "encoder": encoder,
        "freeze_encoder": bool(freeze_encoder),
        "learning_rate": float(learning_rate),
        "confidence_threshold": float(confidence_threshold),
        "seed": int(seed),
        "cross_validation": bool(cross_validation),
        "k_folds": int(k_folds) if k_folds is not None else None,
        "best_val_f1": float(best_val_f1) if best_val_f1 is not None else None,
        "test_accuracy": float(test_accuracy) if test_accuracy is not None else None,
        "test_precision": float(test_precision) if test_precision is not None else None,
        "test_recall": float(test_recall) if test_recall is not None else None,
        "test_f1": float(test_f1) if test_f1 is not None else None,
        "best_checkpoint_path": best_checkpoint_path,
        "confusion_matrix_path": confusion_matrix_path,
        "slurm_job_id": slurm_job_id,
        "status": status,
        "started_at": started_at or _utc_now_iso(),
        "finished_at": finished_at or _utc_now_iso(),
        "hostname": os.uname().nodename if hasattr(os, "uname") else None,
        "pid": os.getpid(),
    }

