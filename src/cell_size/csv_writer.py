"""CSV writers for the catalog (one row per image) and per-image cell areas."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CatalogRow:
    """A single row in the catalog CSV."""

    relative_path: str
    image_name: str
    file_type: str
    mask_name: str
    resize: int
    cell_type: str
    timestamp: str


class CatalogCSV:
    """Accumulates per-image rows and writes the catalog CSV at the end."""

    COLUMNS = [
        "Relative_Path",
        "Image_Name",
        "File_Type",
        "Mask_Name",
        "Resize",
        "Cell_Type",
        "Timestamp",
    ]

    def __init__(self) -> None:
        self._rows: list[CatalogRow] = []

    def add(self, row: CatalogRow) -> None:
        self._rows.append(row)

    def write(self, path: str | Path) -> Path:
        """Write all accumulated rows to *path* as a CSV file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(
            [
                {
                    "Relative_Path": r.relative_path,
                    "Image_Name": r.image_name,
                    "File_Type": r.file_type,
                    "Mask_Name": r.mask_name,
                    "Resize": r.resize,
                    "Cell_Type": r.cell_type,
                    "Timestamp": r.timestamp,
                }
                for r in self._rows
            ],
            columns=self.COLUMNS,
        )
        df.to_csv(path, index=False)
        logger.info("Catalog CSV written -> %s  (%d images)", path, len(self._rows))
        return path

    def __len__(self) -> int:
        return len(self._rows)


def make_catalog_row(
    relative_path: str,
    image_path: Path,
    mask_path: Path,
    resize: int,
    cell_type: str,
) -> CatalogRow:
    """Helper to construct a :class:`CatalogRow` from processing results."""
    return CatalogRow(
        relative_path=relative_path,
        image_name=image_path.stem,
        file_type=image_path.suffix.lstrip("."),
        mask_name=mask_path.stem,
        resize=resize,
        cell_type=cell_type,
        timestamp=datetime.now().isoformat(timespec="seconds"),
    )


def write_cell_areas_csv(
    masks: np.ndarray,
    pixel_scale: Optional[float],
    output_path: str | Path,
) -> Path:
    """Write a per-cell area CSV for a single image.

    Columns: Cell_ID, Area_px, Area_um2 (if *pixel_scale* is available).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    unique_labels, counts = np.unique(masks, return_counts=True)
    # label 0 is background
    cell_mask = unique_labels > 0
    labels = unique_labels[cell_mask]
    areas_px = counts[cell_mask]

    records: list[dict] = []
    for label, area in zip(labels, areas_px):
        row: dict = {"Cell_ID": int(label), "Area_px": int(area)}
        if pixel_scale is not None:
            row["Area_um2"] = round(float(area) * pixel_scale**2, 4)
        records.append(row)

    columns = ["Cell_ID", "Area_px"]
    if pixel_scale is not None:
        columns.append("Area_um2")

    df = pd.DataFrame(records, columns=columns)
    df.to_csv(output_path, index=False)
    logger.info("Cell areas CSV written -> %s  (%d cells)", output_path, len(df))
    return output_path
