"""Gradio end-to-end demo: upload image -> segment -> classify -> view results."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from skimage.measure import regionprops_table

from cell_size.classifier.crop_extractor import _read_image_rgb, extract_all_crops
from cell_size.classifier.inference import (
    _build_inference_transform,
    classify_cells,
    load_checkpoint,
    match_nuclei_to_cells,
)
from cell_size.classifier.visualization import generate_filtered_overlay
from cell_size.io_utils import read_image
from cell_size.metadata import resolve_pixel_scale
from cell_size.segmenter import Segmenter
from cell_size.visualization import generate_overlay

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Segmentation presets
# ---------------------------------------------------------------------------

_SEG_PRESETS = {
    "membrane": {
        "target": "membrane",
        "channel_map": [None, 1, 2],
        "threshold_channel": None,
        "threshold_value": 0.5,
    },
    "nucleus": {
        "target": "nucleus",
        "channel_map": [0, 0, 0],
        "threshold_channel": 0,
        "threshold_value": 0.5,
    },
}

# ---------------------------------------------------------------------------
# Cached model singletons (avoid re-loading on every call)
# ---------------------------------------------------------------------------

_segmenter: Segmenter | None = None
_classifier_model = None
_classifier_ckpt: dict | None = None
_classifier_path: str | None = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_segmenter() -> Segmenter:
    global _segmenter
    if _segmenter is None:
        model_cfg = OmegaConf.create({
            "model_type": "cpsam",
            "custom_model_path": None,
            "gpu": torch.cuda.is_available(),
            "batch_size": 32,
        })
        _segmenter = Segmenter(model_cfg)
    return _segmenter


def _get_classifier(checkpoint_path: str):
    global _classifier_model, _classifier_ckpt, _classifier_path
    if checkpoint_path and checkpoint_path != _classifier_path:
        _classifier_model, _classifier_ckpt = load_checkpoint(checkpoint_path, _device)
        _classifier_path = checkpoint_path
    return _classifier_model, _classifier_ckpt


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    image_path: str | None,
    seg_target: str,
    resize: int,
    flow_threshold: float,
    cellprob_threshold: float,
    min_cell_size: int,
    niter: int,
    checkpoint_path: str,
    confidence_threshold: float,
    selective_rejection_enabled: bool,
    selective_t_bad: float,
    selective_t_good: float,
    pixel_to_um: float | None,
    also_segment_nuclei: bool = True,
):
    """End-to-end pipeline: segment -> classify -> measure."""
    if image_path is None:
        raise gr.Error("Please upload an image first.")

    image_path = Path(image_path)

    # --- 1. Read image ---
    img = read_image(image_path)
    logger.info("Loaded image: %s  shape=%s", image_path.name, img.shape)

    # --- 2. Segment (membrane / selected target) ---
    segmenter = _get_segmenter()
    preset = _SEG_PRESETS.get(seg_target, _SEG_PRESETS["membrane"])
    seg_cfg = OmegaConf.create({
        **preset,
        "flow_threshold": flow_threshold,
        "cellprob_threshold": cellprob_threshold,
        "tile_norm_blocksize": 0,
        "resize": resize,
        "min_cell_size": min_cell_size,
        "niter": niter,
    })
    masks = segmenter.segment(img, seg_cfg)
    n_cells = int(masks.max())
    logger.info("Segmented %d cells (target=%s)", n_cells, seg_target)

    # --- 2b. Also segment nuclei (if requested and target is membrane) ---
    nuc_masks: np.ndarray | None = None
    if also_segment_nuclei and seg_target == "membrane":
        nuc_preset = _SEG_PRESETS["nucleus"]
        nuc_cfg = OmegaConf.create({
            **nuc_preset,
            "flow_threshold": flow_threshold,
            "cellprob_threshold": cellprob_threshold,
            "tile_norm_blocksize": 0,
            "resize": resize,
            "min_cell_size": min_cell_size,
            "niter": niter,
        })
        nuc_masks = segmenter.segment(img, nuc_cfg)
        n_nuc = int(nuc_masks.max())
        logger.info("Segmented %d nuclei", n_nuc)

    # --- 3. Segmentation overlay ---
    tmp_dir = Path(tempfile.mkdtemp())
    seg_overlay_path = tmp_dir / "seg_overlay.png"
    generate_overlay(img, masks, seg_overlay_path)

    # --- 4. Classify (if checkpoint provided) ---
    preds_df = pd.DataFrame(columns=["mask_index", "predicted_verdict", "confidence", "accepted"])
    filtered_overlay_img = None
    areas_df = pd.DataFrame()
    preds_csv_path = None
    areas_csv_path = None

    if checkpoint_path and Path(checkpoint_path).is_file() and n_cells > 0:
        if selective_rejection_enabled and not (0.0 <= selective_t_bad < selective_t_good <= 1.0):
            raise gr.Error(
                "Invalid selective rejection thresholds. "
                f"Require 0 <= t_bad < t_good <= 1, got t_bad={selective_t_bad}, t_good={selective_t_good}."
            )
        if selective_rejection_enabled:
            logger.info(
                "Demo selective rejection enabled: t_bad=%.3f, t_good=%.3f "
                "(confidence_threshold=%.3f ignored)",
                selective_t_bad,
                selective_t_good,
                confidence_threshold,
            )

        model, ckpt = _get_classifier(checkpoint_path)
        crop_size = ckpt.get("crop_size", 224)
        transform = _build_inference_transform(crop_size)

        img_rgb = _read_image_rgb(image_path)
        crops = extract_all_crops(img_rgb, masks, crop_size=crop_size)

        preds = classify_cells(
            model, crops, transform, _device,
            confidence_threshold=confidence_threshold,
            batch_size=32,
            selective_rejection_enabled=selective_rejection_enabled,
            selective_t_bad=selective_t_bad,
            selective_t_good=selective_t_good,
        )
        preds_df = pd.DataFrame(preds)

        preds_csv_path = str(tmp_dir / "predictions.csv")
        preds_df.to_csv(preds_csv_path, index=False)

        filt_overlay_path = tmp_dir / "filtered_overlay.jpg"
        generate_filtered_overlay(
            img, masks, preds_df, filt_overlay_path, nuc_masks=nuc_masks,
        )
        filtered_overlay_img = str(filt_overlay_path)

        areas_df = _compute_good_cell_areas(
            masks, preds_df, image_path, pixel_to_um, nuc_masks=nuc_masks,
        )
        areas_csv_path = str(tmp_dir / "filtered_areas.csv")
        areas_df.to_csv(areas_csv_path, index=False)

        n_good = (preds_df["predicted_verdict"] == "good").sum()
        n_bad = (preds_df["predicted_verdict"] == "bad").sum()
        n_rejected = (preds_df["predicted_verdict"] == "rejected").sum()
        logger.info("Classification: %d good, %d bad, %d rejected", n_good, n_bad, n_rejected)
    elif n_cells == 0:
        logger.warning("No cells segmented -- skipping classification.")
    else:
        logger.info("No classifier checkpoint provided -- skipping classification.")

    return (
        str(seg_overlay_path),
        filtered_overlay_img,
        preds_df,
        areas_df,
        preds_csv_path,
        areas_csv_path,
    )


def _compute_good_cell_areas(
    masks: np.ndarray,
    preds_df: pd.DataFrame,
    image_path: Path,
    config_pixel_to_um: float | None,
    nuc_masks: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compute area and major/minor axis for cells predicted as good.

    When *nuc_masks* is provided, nucleus measurements and N/C ratio are
    added for each cell that has a matching nucleus.
    """
    pixel_to_um = resolve_pixel_scale(image_path, config_pixel_to_um)

    good_labels = set(
        preds_df.loc[preds_df["predicted_verdict"] == "good", "mask_index"].astype(int)
    )
    if not good_labels:
        return pd.DataFrame()

    props = regionprops_table(
        masks.astype(np.int32),
        properties=("label", "area", "major_axis_length", "minor_axis_length"),
    )

    # Nucleus matching
    nuc_matches: dict[int, int | None] = {}
    nuc_props_lut: dict[int, dict] = {}
    if nuc_masks is not None:
        nuc_matches = match_nuclei_to_cells(masks, nuc_masks)
        nuc_props = regionprops_table(
            nuc_masks.astype(np.int32),
            properties=("label", "area", "major_axis_length", "minor_axis_length"),
        )
        for i in range(len(nuc_props["label"])):
            nuc_props_lut[int(nuc_props["label"][i])] = {
                "area": int(nuc_props["area"][i]),
                "major": float(nuc_props["major_axis_length"][i]),
                "minor": float(nuc_props["minor_axis_length"][i]),
            }

    records: list[dict] = []
    for i in range(len(props["label"])):
        label = int(props["label"][i])
        if label not in good_labels:
            continue
        area_px = int(props["area"][i])
        rec: dict = {
            "mask_index": label,
            "area_px": area_px,
            "major_axis_px": round(float(props["major_axis_length"][i]), 2),
            "minor_axis_px": round(float(props["minor_axis_length"][i]), 2),
        }
        if pixel_to_um is not None:
            rec["area_um2"] = round(area_px * pixel_to_um**2, 4)
            rec["major_axis_um"] = round(rec["major_axis_px"] * pixel_to_um, 4)
            rec["minor_axis_um"] = round(rec["minor_axis_px"] * pixel_to_um, 4)

        if nuc_masks is not None:
            nuc_label = nuc_matches.get(label)
            if nuc_label is not None and nuc_label in nuc_props_lut:
                ninfo = nuc_props_lut[nuc_label]
                rec["nucleus_area_px"] = ninfo["area"]
                rec["nucleus_major_axis_px"] = round(ninfo["major"], 2)
                rec["nucleus_minor_axis_px"] = round(ninfo["minor"], 2)
                rec["nc_ratio"] = round(ninfo["area"] / max(area_px, 1), 4)
                if pixel_to_um is not None:
                    rec["nucleus_area_um2"] = round(ninfo["area"] * pixel_to_um**2, 4)
                    rec["nucleus_major_axis_um"] = round(ninfo["major"] * pixel_to_um, 4)
                    rec["nucleus_minor_axis_um"] = round(ninfo["minor"] * pixel_to_um, 4)
            else:
                rec["nucleus_area_px"] = None
                rec["nucleus_major_axis_px"] = None
                rec["nucleus_minor_axis_px"] = None
                rec["nc_ratio"] = None

        records.append(rec)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------


def _preview_upload(file_obj) -> tuple:
    """Convert any uploaded image (including TIFF) to a displayable preview.

    Returns (preview_numpy_array, original_filepath).
    """
    if file_obj is None:
        return None, None
    fpath = file_obj if isinstance(file_obj, str) else file_obj.name
    try:
        img_rgb = _read_image_rgb(Path(fpath))
        return img_rgb, fpath
    except Exception as exc:
        logger.warning("Preview failed for %s: %s", fpath, exc)
        return None, fpath


def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="Cell Size Estimator",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown("# Cell Size Estimator -- Interactive Demo")
        gr.Markdown(
            "Upload a microscopy image, configure segmentation and classification, "
            "then run the full pipeline to view and download results."
        )

        original_path = gr.State(value=None)

        with gr.Row():
            # ---- Left column: inputs ----
            with gr.Column(scale=1):
                gr.Markdown("### Input")
                file_upload = gr.File(
                    label="Upload image (TIFF, PNG, JPG, BMP)",
                    file_types=[".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"],
                )
                image_preview = gr.Image(
                    label="Image preview",
                    interactive=False,
                    height=250,
                )

                gr.Markdown("### Segmentation")
                seg_target = gr.Dropdown(
                    choices=["membrane", "nucleus"],
                    value="membrane",
                    label="Segmentation target",
                )
                with gr.Accordion("Advanced segmentation parameters", open=False):
                    resize = gr.Slider(
                        minimum=0, maximum=2000, value=0, step=50,
                        label="Resize (0 = no resize)",
                    )
                    flow_threshold = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.4, step=0.05,
                        label="Flow threshold",
                    )
                    cellprob_threshold = gr.Slider(
                        minimum=-6.0, maximum=6.0, value=0.0, step=0.1,
                        label="Cell probability threshold",
                    )
                    min_cell_size = gr.Slider(
                        minimum=0, maximum=500, value=15, step=5,
                        label="Min cell size (px)",
                    )
                    niter = gr.Number(
                        value=500, label="niter", precision=0,
                    )

                also_segment_nuclei = gr.Checkbox(
                    value=True,
                    label="Also segment nuclei (adds nucleus measurements)",
                )

                gr.Markdown("### Classification")
                checkpoint_path = gr.Textbox(
                    label="Classifier checkpoint path (.pt)",
                    placeholder="/path/to/best_model.pt",
                )
                confidence_threshold = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.7, step=0.05,
                    label="Confidence threshold",
                )
                selective_rejection_enabled = gr.Checkbox(
                    value=False,
                    label="Enable selective rejection",
                    info="When enabled: p<=t_bad => bad, p>=t_good => good, otherwise rejected.",
                )
                selective_t_bad = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.09, step=0.01,
                    label="Selective threshold t_bad",
                )
                selective_t_good = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.51, step=0.01,
                    label="Selective threshold t_good",
                )

                gr.Markdown("### Scale")
                pixel_to_um = gr.Number(
                    value=None,
                    label="Pixel-to-\u00b5m (optional, auto-detected from TIFF metadata)",
                    precision=6,
                )

                run_btn = gr.Button("Run Pipeline", variant="primary", size="lg")

            # ---- Right column: outputs ----
            with gr.Column(scale=2):
                gr.Markdown("### Results")
                with gr.Tabs():
                    with gr.Tab("Segmentation Overlay"):
                        seg_overlay_output = gr.Image(label="Segmentation overlay")
                    with gr.Tab("Filtered Overlay"):
                        filt_overlay_output = gr.Image(
                            label="Filtered overlay (good=green, bad=orange, rejected=magenta)",
                        )

                with gr.Tabs():
                    with gr.Tab("All Predictions"):
                        preds_table = gr.Dataframe(
                            label="Per-cell predictions",
                            headers=["mask_index", "predicted_verdict", "confidence", "accepted"],
                            interactive=False,
                        )
                    with gr.Tab("Filtered Areas (good cells)"):
                        areas_table = gr.Dataframe(
                            label="Areas and diameters for good cells",
                            interactive=False,
                        )

                gr.Markdown("### Downloads")
                with gr.Row():
                    preds_csv_file = gr.File(label="predictions.csv")
                    areas_csv_file = gr.File(label="filtered_areas.csv")

        # ---- Wire upload preview ----
        file_upload.change(
            fn=_preview_upload,
            inputs=[file_upload],
            outputs=[image_preview, original_path],
        )

        # ---- Wire pipeline ----
        run_btn.click(
            fn=run_pipeline,
            inputs=[
                original_path,
                seg_target,
                resize,
                flow_threshold,
                cellprob_threshold,
                min_cell_size,
                niter,
                checkpoint_path,
                confidence_threshold,
                selective_rejection_enabled,
                selective_t_bad,
                selective_t_good,
                pixel_to_um,
                also_segment_nuclei,
            ],
            outputs=[
                seg_overlay_output,
                filt_overlay_output,
                preds_table,
                areas_table,
                preds_csv_file,
                areas_csv_file,
            ],
        )

    return app


def main(share: bool = False, server_name: str = "0.0.0.0", server_port: int = 7860):
    app = build_app()
    app.launch(share=share, server_name=server_name, server_port=server_port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cell Size Estimator Demo")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    parser.add_argument("--server-name", default="0.0.0.0", help="Server bind address")
    parser.add_argument("--server-port", type=int, default=7860, help="Server port")
    args = parser.parse_args()

    main(share=args.share, server_name=args.server_name, server_port=args.server_port)
