"""Sample/frog identifier extraction from project image names."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

SAMPLE_ID_RE = re.compile(r"^TIFF_AH_([A-Za-z0-9]+)_[A-Za-z0-9]+$")


def extract_frog_id(image_name: str) -> str | None:
    """Extract the sample token from names like TIFF_AH_001_04 or TIFF_AH_030K_12."""
    stem = Path(str(image_name)).stem
    m = SAMPLE_ID_RE.match(stem)
    if m is None:
        return None
    return m.group(1)


def with_frog_id(
    df: pd.DataFrame,
    *,
    image_col: str = "image_path",
    frog_col: str = "frog_id",
    overwrite: bool = True,
) -> pd.DataFrame:
    """Return a copy of ``df`` with string frog IDs derived from ``image_col``."""
    out = df.copy()
    if image_col not in out.columns:
        if frog_col in out.columns:
            out[frog_col] = out[frog_col].astype("string")
        else:
            out[frog_col] = pd.Series(pd.NA, index=out.index, dtype="string")
        return out

    parsed = out[image_col].map(extract_frog_id).astype("string")
    if overwrite or frog_col not in out.columns:
        out[frog_col] = parsed
        return out

    existing = out[frog_col].astype("string")
    missing = existing.isna() | existing.str.strip().isin({"", "nan", "None", "<NA>"})
    out[frog_col] = existing.mask(missing, parsed)
    return out
