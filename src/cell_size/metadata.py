"""Pixel-scale metadata extraction from microscopy images."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import tifffile

logger = logging.getLogger(__name__)


def get_pixel_scale(image_path: str | Path) -> Optional[float]:
    """Attempt to read the pixel-to-micrometer scale from image metadata.

    Supports OME-TIFF (``PhysicalSizeX`` attribute) and standard TIFF
    resolution tags. Returns micrometers-per-pixel, or ``None`` if the
    scale cannot be determined.
    """
    image_path = Path(image_path)

    if image_path.suffix.lower() not in {".tif", ".tiff"}:
        logger.debug("Not a TIFF file, cannot extract pixel scale: %s", image_path.name)
        return None

    try:
        scale = _try_ome_tiff(image_path)
        if scale is not None:
            return scale
    except Exception as exc:
        logger.debug("OME-TIFF metadata extraction failed: %s", exc)

    try:
        scale = _try_tiff_resolution(image_path)
        if scale is not None:
            return scale
    except Exception as exc:
        logger.debug("TIFF resolution tag extraction failed: %s", exc)

    logger.debug("No pixel scale found in metadata for %s", image_path.name)
    return None


def _try_ome_tiff(image_path: Path) -> Optional[float]:
    """Extract PhysicalSizeX from OME-XML metadata embedded in a TIFF."""
    with tifffile.TiffFile(str(image_path)) as tif:
        if not tif.ome_metadata:
            return None

        root = ET.fromstring(tif.ome_metadata)
        ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}

        for pixels_elem in root.iter():
            tag = pixels_elem.tag
            if tag.endswith("}Pixels") or tag == "Pixels":
                phys_x = pixels_elem.attrib.get("PhysicalSizeX")
                unit = pixels_elem.attrib.get("PhysicalSizeXUnit", "µm")
                if phys_x is not None:
                    scale = float(phys_x)
                    if unit.lower() in ("µm", "um", "micron", "micrometer"):
                        logger.info(
                            "OME-TIFF pixel scale: %.6f µm/px (%s)", scale, image_path.name
                        )
                        return scale
                    elif unit.lower() in ("nm", "nanometer"):
                        scale /= 1000.0
                        logger.info(
                            "OME-TIFF pixel scale (converted from nm): %.6f µm/px (%s)",
                            scale,
                            image_path.name,
                        )
                        return scale
                    else:
                        logger.warning("Unknown PhysicalSizeXUnit '%s', assuming µm", unit)
                        return scale
    return None


def _try_tiff_resolution(image_path: Path) -> Optional[float]:
    """Extract pixel scale from standard TIFF XResolution/YResolution tags."""
    with tifffile.TiffFile(str(image_path)) as tif:
        page = tif.pages[0]
        tags = page.tags

        res_unit_tag = tags.get("ResolutionUnit")
        x_res_tag = tags.get("XResolution")

        if x_res_tag is None:
            return None

        x_res = x_res_tag.value
        if isinstance(x_res, tuple):
            x_res = x_res[0] / x_res[1] if x_res[1] != 0 else 0

        if x_res <= 0:
            return None

        pixels_per_unit = float(x_res)
        unit_um = 1.0

        if res_unit_tag is not None:
            res_unit = res_unit_tag.value
            if res_unit == 2:  # inch
                unit_um = 25400.0
            elif res_unit == 3:  # centimeter
                unit_um = 10000.0
            else:
                return None
        else:
            return None

        scale = unit_um / pixels_per_unit
        logger.info("TIFF resolution tag pixel scale: %.6f µm/px (%s)", scale, image_path.name)
        return scale


def resolve_pixel_scale(
    image_path: str | Path,
    config_pixel_to_um: Optional[float] = None,
) -> Optional[float]:
    """Determine pixel scale: auto-detect first, then fall back to config value."""
    scale = get_pixel_scale(image_path)
    if scale is not None:
        return scale

    if config_pixel_to_um is not None:
        logger.info(
            "Using configured pixel scale: %.6f µm/px for %s",
            config_pixel_to_um,
            Path(image_path).name,
        )
        return config_pixel_to_um

    logger.warning("No pixel scale available for %s", Path(image_path).name)
    return None
