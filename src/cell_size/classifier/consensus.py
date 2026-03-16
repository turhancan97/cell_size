"""Multi-CSV merge and majority-vote consensus for reviewer feedback."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"dataset", "image_path", "mask_index", "verdict"}
GROUP_KEYS = ["dataset", "image_path", "mask_index"]


def load_and_merge(csv_paths: list[str | Path]) -> pd.DataFrame:
    """Read and concatenate multiple feedback CSVs into one DataFrame.

    Every CSV must contain at least the columns defined in
    ``REQUIRED_COLUMNS``.  Extra columns are preserved.
    """
    frames: list[pd.DataFrame] = []
    for p in csv_paths:
        p = Path(p)
        if not p.is_file():
            logger.warning("Feedback CSV not found, skipping: %s", p)
            continue
        df = pd.read_csv(p, skipinitialspace=True)
        df.columns = df.columns.str.strip()
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"CSV {p} is missing required columns: {missing}")
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No valid feedback CSVs found.")

    merged = pd.concat(frames, ignore_index=True)
    logger.info("Merged %d feedback CSVs -> %d total rows", len(frames), len(merged))
    return merged


def apply_majority_vote(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate multi-reviewer labels with majority vote.

    Groups by ``(dataset, image_path, mask_index)`` and picks the most
    common verdict.  Ties default to ``"bad"`` (conservative).

    Returns a DataFrame with one row per unique cell, containing at minimum
    the ``GROUP_KEYS`` plus ``verdict``.
    """
    def _resolve(group: pd.DataFrame) -> str:
        counts = group["verdict"].str.lower().value_counts()
        if len(counts) == 1:
            return counts.index[0]
        if counts.iloc[0] == counts.iloc[1]:
            return "bad"
        return counts.index[0]

    resolved = (
        df.groupby(GROUP_KEYS, sort=False)
        .apply(_resolve, include_groups=False)
        .reset_index(name="verdict")
    )

    n_good = (resolved["verdict"] == "good").sum()
    n_bad = (resolved["verdict"] == "bad").sum()
    logger.info(
        "Majority-vote consensus: %d unique cells (%d good, %d bad)",
        len(resolved),
        n_good,
        n_bad,
    )
    return resolved
