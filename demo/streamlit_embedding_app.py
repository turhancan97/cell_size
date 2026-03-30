"""Interactive Streamlit explorer for crop-level embeddings.

This app visualizes classifier crop embeddings in 2D/3D with selectable
methods (PCA, t-SNE, UMAP), model checkpoints, and uncertainty-reject policy.
"""

from __future__ import annotations

import hashlib
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

try:
    import umap  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    umap = None


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cell_size.classifier.dataset import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from cell_size.classifier.models import EfficientProbingViTClassifier, build_model, get_classifier_module  # noqa: E402


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEFAULT_SAMPLE_CAP = 5000
MAX_TSNE_SAMPLES = 8000
MAX_ALL_SAMPLES_WITHOUT_CAP = 20000
DEFAULT_MAX_PLOT_POINTS = 30000


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ImageFolderWithPaths(ImageFolder):
    """ImageFolder that also returns file path and dataset index."""

    def __getitem__(self, index: int):  # type: ignore[override]
        image, target = super().__getitem__(index)
        path, _ = self.samples[index]
        return image, target, path, index


def discover_mask_modes(crops_root: Path) -> list[str]:
    if not crops_root.is_dir():
        return []
    return sorted([p.name for p in crops_root.iterdir() if p.is_dir()])


def discover_splits(mask_mode_dir: Path) -> list[str]:
    if not mask_mode_dir.is_dir():
        return []

    candidates = []
    for p in mask_mode_dir.iterdir():
        if not p.is_dir():
            continue
        if (p / "good").is_dir() and (p / "bad").is_dir():
            candidates.append(p.name)

    preferred = ["train", "val", "test"]
    found_pref = [x for x in preferred if x in candidates]
    others = sorted([x for x in candidates if x not in preferred])
    return found_pref + others


@st.cache_data(show_spinner=False)
def discover_checkpoints(search_root: str) -> list[str]:
    root = Path(search_root).expanduser().resolve()
    if not root.exists():
        return []

    found = sorted(root.rglob("best_model.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p.resolve()) for p in found]


def format_checkpoint_display(checkpoint_path: str, search_root: str) -> str:
    """Show checkpoint paths relative to the selected search root in UI."""
    ckpt = Path(checkpoint_path).expanduser().resolve()
    root = Path(search_root).expanduser().resolve()
    try:
        return ckpt.relative_to(root).as_posix()
    except ValueError:
        return ckpt.name


@st.cache_data(show_spinner=False)
def dataset_signature(split_dir_str: str) -> dict[str, Any]:
    split_dir = Path(split_dir_str)
    h = hashlib.sha256()
    n_files = 0
    latest_mtime = 0

    for p in sorted(split_dir.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        n_files += 1
        stt = p.stat()
        latest_mtime = max(latest_mtime, stt.st_mtime_ns)
        rel = p.relative_to(split_dir).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(str(stt.st_size).encode("utf-8"))
        h.update(str(stt.st_mtime_ns).encode("utf-8"))

    return {
        "n_files": n_files,
        "latest_mtime_ns": latest_mtime,
        "sha256": h.hexdigest(),
    }


def get_head_module(model: nn.Module, encoder: str) -> nn.Module:
    return get_classifier_module(model, encoder)


def load_checkpoint_model(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    model = build_model(
        encoder=ckpt["encoder"],
        pretrained=False,
        freeze_encoder=False,
        use_mlp_head=bool(ckpt.get("use_mlp_head", False)),
        use_efficient_probing=bool(ckpt.get("use_efficient_probing", False)),
        efficient_probing_cfg=dict(ckpt.get("efficient_probing", {})),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt


def build_transform(crop_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def compute_features_and_logits(
    split_dir: Path,
    checkpoint_path: Path,
    sample_cap: int,
    seed: int,
    num_workers: int,
    batch_size: int,
    device: torch.device,
    mask_mode: str,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    model, ckpt = load_checkpoint_model(checkpoint_path, device)
    encoder = str(ckpt["encoder"])
    crop_size = int(ckpt.get("crop_size", 224))

    dataset = ImageFolderWithPaths(str(split_dir), transform=build_transform(crop_size))
    if "good" not in dataset.class_to_idx or "bad" not in dataset.class_to_idx:
        raise ValueError(f"Expected classes good/bad in {split_dir}, got {dataset.class_to_idx}")

    rng = np.random.default_rng(seed)
    if sample_cap > 0 and len(dataset) > sample_cap:
        indices = np.sort(rng.choice(len(dataset), size=sample_cap, replace=False)).tolist()
        data_for_loader = Subset(dataset, indices)
    else:
        data_for_loader = dataset

    loader = DataLoader(
        data_for_loader,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    feature_batches: list[np.ndarray] = []
    logit_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []
    path_list: list[str] = []

    if isinstance(model, EfficientProbingViTClassifier):
        with torch.no_grad():
            for images, targets, paths, _ in loader:
                images_dev = images.to(device)
                probe_feats = model.extract_probe_features(images_dev)
                feats = probe_feats.cpu().numpy()
                logits = model.classifier(probe_feats).squeeze(1).cpu().numpy()
                if feats.ndim > 2:
                    feats = feats.reshape(feats.shape[0], -1)

                logit_batches.append(logits)
                feature_batches.append(feats)
                target_batches.append(targets.numpy())
                path_list.extend([str(p) for p in paths])
    else:
        head = get_head_module(model, encoder)
        cache: dict[str, torch.Tensor] = {}

        def _hook(_module, inputs, _output):
            cache["features"] = inputs[0].detach().cpu()

        handle = head.register_forward_hook(_hook)
        try:
            with torch.no_grad():
                for images, targets, paths, _ in loader:
                    logits = model(images.to(device)).squeeze(1).cpu().numpy()
                    feats_tensor = cache.get("features")
                    if feats_tensor is None:
                        raise RuntimeError("Failed to capture penultimate features from model hook.")
                    feats = feats_tensor.numpy()
                    if feats.ndim > 2:
                        feats = feats.reshape(feats.shape[0], -1)

                    logit_batches.append(logits)
                    feature_batches.append(feats)
                    target_batches.append(targets.numpy())
                    path_list.extend([str(p) for p in paths])
        finally:
            handle.remove()

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    logits_all = np.concatenate(logit_batches, axis=0)
    features_all = np.concatenate(feature_batches, axis=0)
    targets_all = np.concatenate(target_batches, axis=0)

    good_idx = int(dataset.class_to_idx["good"])
    y_true = (targets_all == good_idx).astype(int)
    p_good = 1.0 / (1.0 + np.exp(-logits_all))
    y_pred = (p_good >= 0.5).astype(int)

    df = pd.DataFrame({
        "row_id": np.arange(len(path_list), dtype=int),
        "path": path_list,
        "true_id": y_true,
        "true_label": np.where(y_true == 1, "good", "bad"),
        "logit": logits_all.astype(float),
        "p_good": p_good.astype(float),
        "pred_id": y_pred,
        "pred_label": np.where(y_pred == 1, "good", "bad"),
    })

    meta = {
        "encoder": encoder,
        "use_mlp_head": bool(ckpt.get("use_mlp_head", False)),
        "use_efficient_probing": bool(ckpt.get("use_efficient_probing", False)),
        "efficient_probing": dict(ckpt.get("efficient_probing", {})),
        "crop_size": crop_size,
        "checkpoint": str(checkpoint_path),
        "mask_mode": mask_mode,
        "split_dir": str(split_dir),
        "sampled_count": len(df),
    }
    return df, features_all, meta


def compute_embedding(
    features: np.ndarray,
    method: str,
    viz_dims: int,
    seed: int,
    pca_n_components: int,
    tsne_perplexity: float,
    umap_n_neighbors: int,
    umap_min_dist: float,
) -> np.ndarray:
    n_samples, n_features = features.shape
    if n_samples < 2:
        raise ValueError("Need at least 2 samples to compute embeddings.")

    if method == "PCA":
        n_comp = max(viz_dims, int(pca_n_components))
        n_comp = min(n_comp, n_samples, n_features)
        transformed = PCA(n_components=n_comp, random_state=seed).fit_transform(features)
        return transformed[:, :viz_dims]

    if method == "t-SNE":
        pre_dims = min(50, n_features, n_samples - 1)
        if pre_dims >= 2:
            x = PCA(n_components=pre_dims, random_state=seed).fit_transform(features)
        else:
            x = features

        max_perplexity = max(1.0, float(n_samples - 1))
        perplexity = min(float(tsne_perplexity), max_perplexity)
        perplexity = max(1.0, perplexity)

        return TSNE(
            n_components=viz_dims,
            random_state=seed,
            init="pca",
            learning_rate="auto",
            perplexity=perplexity,
        ).fit_transform(x)

    if method == "UMAP":
        if umap is None:
            raise RuntimeError("UMAP is not available (install umap-learn).")
        model = umap.UMAP(
            n_components=viz_dims,
            n_neighbors=int(umap_n_neighbors),
            min_dist=float(umap_min_dist),
            random_state=seed,
        )
        return model.fit_transform(features)

    raise ValueError(f"Unknown method: {method}")


def apply_reject_policy(df: pd.DataFrame, t_bad: float, t_good: float) -> pd.DataFrame:
    if t_bad >= t_good:
        raise ValueError("t_bad must be smaller than t_good.")

    out = df.copy()
    p = out["p_good"].to_numpy(dtype=float)

    accepted = (p <= t_bad) | (p >= t_good)
    accepted_pred = np.where(p <= t_bad, "bad", np.where(p >= t_good, "good", "rejected"))

    out["accepted"] = accepted
    out["accepted_label"] = accepted_pred
    out["accepted_id"] = np.where(accepted_pred == "good", 1, np.where(accepted_pred == "bad", 0, -1))
    out["accepted_status"] = np.where(accepted, "accepted", "rejected")

    def _conf(row: pd.Series) -> str:
        if row["accepted_label"] == "rejected":
            return "REJECT"
        if row["true_label"] == "good" and row["accepted_label"] == "good":
            return "TP"
        if row["true_label"] == "bad" and row["accepted_label"] == "bad":
            return "TN"
        if row["true_label"] == "bad" and row["accepted_label"] == "good":
            return "FP"
        return "FN"

    out["confusion_class"] = out.apply(_conf, axis=1)
    out["uncertainty"] = 1.0 - np.abs(2.0 * out["p_good"].to_numpy(dtype=float) - 1.0)
    return out


def accepted_metrics(df: pd.DataFrame) -> dict[str, float]:
    n_total = int(len(df))
    acc_df = df[df["accepted"]].copy()
    n_accepted = int(len(acc_df))
    n_rejected = n_total - n_accepted
    coverage = float(n_accepted / n_total) if n_total > 0 else float("nan")

    out: dict[str, float] = {
        "n_total": float(n_total),
        "n_accepted": float(n_accepted),
        "n_rejected": float(n_rejected),
        "coverage": coverage,
        "rejected_good": float(((~df["accepted"]) & (df["true_label"] == "good")).sum()),
        "rejected_bad": float(((~df["accepted"]) & (df["true_label"] == "bad")).sum()),
    }

    if n_accepted == 0:
        out.update({
            "accuracy": float("nan"),
            "recall": float("nan"),
            "precision": float("nan"),
            "f1": float("nan"),
            "specificity": float("nan"),
            "tn": 0.0,
            "fp": 0.0,
            "fn": 0.0,
            "tp": 0.0,
        })
        return out

    y_true = acc_df["true_id"].to_numpy(dtype=int)
    y_pred = acc_df["accepted_id"].to_numpy(dtype=int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")

    out.update({
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": specificity,
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    })
    return out


def load_rgb(path: str) -> np.ndarray:
    with Image.open(path) as im:
        return np.array(im.convert("RGB"))


def build_plot(
    df: pd.DataFrame,
    viz_dims: int,
    color_mode: str,
) -> Any:
    color_col_map = {
        "True label": "true_label",
        "Predicted label": "pred_label",
        "Accepted/Rejected": "accepted_status",
        "Confusion class": "confusion_class",
    }
    color_col = color_col_map[color_mode]

    color_maps = {
        "true_label": {"good": "#2ca02c", "bad": "#ff7f0e"},
        "pred_label": {"good": "#2ca02c", "bad": "#ff7f0e"},
        "accepted_status": {"accepted": "#1f77b4", "rejected": "#7f7f7f"},
        "confusion_class": {
            "TP": "#2ca02c",
            "TN": "#1f77b4",
            "FP": "#d62728",
            "FN": "#9467bd",
            "REJECT": "#7f7f7f",
        },
    }

    common_kwargs = {
        "color": color_col,
        "color_discrete_map": color_maps.get(color_col, None),
        "custom_data": [
            "row_id",
            "path",
            "true_label",
            "pred_label",
            "accepted_label",
            "p_good",
            "accepted_status",
            "confusion_class",
        ],
    }

    if viz_dims == 3:
        fig = px.scatter_3d(df, x="emb_1", y="emb_2", z="emb_3", **common_kwargs)
        fig.update_traces(marker={"size": 4, "opacity": 0.85})
    else:
        fig = px.scatter(df, x="emb_1", y="emb_2", **common_kwargs)
        fig.update_traces(marker={"size": 7, "opacity": 0.85})

    # Keep metadata out of cursor tooltip; details are shown in the fixed side panel.
    fig.update_traces(hoverinfo="skip")
    fig.update_layout(height=760, legend_title_text=color_col, clickmode="event+select")
    return fig


def extract_selected_row_ids(plot_event: Any) -> list[int]:
    """Extract selected row IDs from Streamlit plotly selection payload."""
    if plot_event is None:
        return []

    points: list[dict[str, Any]] = []
    if isinstance(plot_event, dict):
        points = list(plot_event.get("selection", {}).get("points", []) or [])
    else:
        selection = getattr(plot_event, "selection", None)
        if selection is not None:
            points = list(getattr(selection, "points", []) or [])

    out: list[int] = []
    for point in points:
        custom = point.get("customdata", [])
        if isinstance(custom, (list, tuple)) and len(custom) > 0:
            try:
                out.append(int(custom[0]))
            except Exception:
                continue
    return out


def main() -> None:
    st.set_page_config(page_title="Cell Embedding Explorer", layout="wide")
    set_seed(42)

    st.title("Cell Crop Embedding Explorer")
    st.caption("Interactive 2D/3D embedding viewer with model selection and uncertainty rejection.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.caption(f"Device: `{device}`")

    with st.sidebar:
        st.header("Data and Model")
        crops_root_raw = st.text_input(
            "Crops root",
            value=str((REPO_ROOT / "classifier_output" / "crops").resolve()),
        )
        crops_root = Path(crops_root_raw).expanduser().resolve()

        mask_modes = discover_mask_modes(crops_root)
        if not mask_modes:
            st.error(f"No mask modes found in {crops_root}")
            st.stop()
        mask_mode = st.selectbox("Mask mode", mask_modes, index=0)

        split_options = discover_splits(crops_root / mask_mode)
        if not split_options:
            st.error(f"No valid split folders (good/bad) found in {crops_root / mask_mode}")
            st.stop()
        split = st.selectbox("Split", split_options, index=0)

        ckpt_search_root_raw = st.text_input(
            "Checkpoint search root",
            value=str((REPO_ROOT / "classifier_output").resolve()),
        )
        checkpoint_candidates = discover_checkpoints(ckpt_search_root_raw)
        if not checkpoint_candidates:
            st.error(f"No `best_model.pt` found under {ckpt_search_root_raw}")
            st.stop()
        checkpoint = st.selectbox(
            "Checkpoint",
            checkpoint_candidates,
            format_func=lambda p: format_checkpoint_display(p, ckpt_search_root_raw),
        )

        st.header("Embedding Method")
        methods = ["PCA", "t-SNE"]
        if umap is not None:
            methods.append("UMAP")
        else:
            st.warning("UMAP disabled: `umap-learn` not installed.")
        method = st.selectbox("Method", methods, index=0)

        dim_label = st.radio("Visualization space", options=["2D", "3D"], horizontal=True, index=0)
        viz_dims = 2 if dim_label == "2D" else 3

        pca_n_components = st.slider("PCA n_components", min_value=2, max_value=50, value=10, step=1)
        tsne_perplexity = st.slider("t-SNE perplexity", min_value=2.0, max_value=100.0, value=30.0, step=1.0)
        umap_n_neighbors = st.slider("UMAP n_neighbors", min_value=2, max_value=100, value=15, step=1)
        umap_min_dist = st.slider("UMAP min_dist", min_value=0.0, max_value=0.99, value=0.1, step=0.01)

        st.header("Sampling and Policy")
        sample_cap = st.number_input("Sample cap (0 = all)", min_value=0, value=DEFAULT_SAMPLE_CAP, step=100)
        seed = st.number_input("Random seed", min_value=0, value=42, step=1)
        batch_size = st.number_input("Batch size", min_value=1, value=128, step=1)
        num_workers = st.number_input("DataLoader workers", min_value=0, value=0, step=1)
        max_plot_points = st.number_input(
            "Max plot points",
            min_value=1000,
            max_value=200000,
            value=DEFAULT_MAX_PLOT_POINTS,
            step=1000,
        )

        t_bad = st.slider("t_bad (predict bad if p <= t_bad)", min_value=0.0, max_value=0.49, value=0.20, step=0.01)
        t_good = st.slider("t_good (predict good if p >= t_good)", min_value=0.51, max_value=1.0, value=0.80, step=0.01)
        if t_bad >= t_good:
            st.error("Invalid thresholds: t_bad must be < t_good.")
            st.stop()

        color_mode = st.selectbox(
            "Color by",
            options=["True label", "Predicted label", "Accepted/Rejected", "Confusion class"],
            index=0,
        )
        auto_recompute = st.checkbox(
            "Auto recompute on every control change",
            value=False,
            help="If disabled, press Run / Refresh to apply changed settings.",
        )
        run_refresh = st.button("Run / Refresh", type="primary")

    current_params = {
        "crops_root": str(crops_root),
        "mask_mode": str(mask_mode),
        "split": str(split),
        "checkpoint": str(checkpoint),
        "method": str(method),
        "viz_dims": int(viz_dims),
        "pca_n_components": int(pca_n_components),
        "tsne_perplexity": float(tsne_perplexity),
        "umap_n_neighbors": int(umap_n_neighbors),
        "umap_min_dist": float(umap_min_dist),
        "sample_cap": int(sample_cap),
        "seed": int(seed),
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "max_plot_points": int(max_plot_points),
        "t_bad": float(t_bad),
        "t_good": float(t_good),
        "color_mode": str(color_mode),
    }

    if "applied_params" not in st.session_state:
        st.session_state.applied_params = None
    if auto_recompute or run_refresh or st.session_state.applied_params is None:
        st.session_state.applied_params = current_params
    elif st.session_state.applied_params != current_params:
        st.info("Controls changed. Click `Run / Refresh` to recompute.")

    params = st.session_state.applied_params
    if params is None:
        st.stop()

    crops_root = Path(params["crops_root"]).resolve()
    mask_mode = str(params["mask_mode"])
    split = str(params["split"])
    checkpoint = str(params["checkpoint"])
    method = str(params["method"])
    viz_dims = int(params["viz_dims"])
    pca_n_components = int(params["pca_n_components"])
    tsne_perplexity = float(params["tsne_perplexity"])
    umap_n_neighbors = int(params["umap_n_neighbors"])
    umap_min_dist = float(params["umap_min_dist"])
    sample_cap = int(params["sample_cap"])
    seed = int(params["seed"])
    batch_size = int(params["batch_size"])
    num_workers = int(params["num_workers"])
    max_plot_points = int(params["max_plot_points"])
    t_bad = float(params["t_bad"])
    t_good = float(params["t_good"])
    color_mode = str(params["color_mode"])

    split_dir = (crops_root / mask_mode / split).resolve()
    if not split_dir.is_dir():
        st.error(f"Split directory not found: {split_dir}")
        st.stop()

    split_sig = dataset_signature(str(split_dir))
    if int(sample_cap) == 0 and int(split_sig.get("n_files", 0)) > MAX_ALL_SAMPLES_WITHOUT_CAP:
        st.error(
            "For stability, `Sample cap = 0 (all)` is disabled on very large splits. "
            f"Found {int(split_sig.get('n_files', 0))} images; set `Sample cap` to <= {MAX_ALL_SAMPLES_WITHOUT_CAP}."
        )
        st.stop()

    with st.spinner("Loading features and predictions..."):
        feature_df, features, model_meta = compute_features_and_logits(
            split_dir=split_dir,
            checkpoint_path=Path(checkpoint),
            sample_cap=int(sample_cap),
            seed=int(seed),
            num_workers=int(num_workers),
            batch_size=int(batch_size),
            device=device,
            mask_mode=mask_mode,
        )

    if len(feature_df) < 2:
        st.error("Need at least 2 samples for embedding visualization. Increase sample cap or choose another split.")
        st.stop()

    if method == "t-SNE" and len(feature_df) > MAX_TSNE_SAMPLES:
        st.error(
            "t-SNE is disabled for very large sample counts to avoid UI crashes. "
            f"Current samples: {len(feature_df)}; allowed max: {MAX_TSNE_SAMPLES}. "
            "Lower `Sample cap` and rerun."
        )
        st.stop()

    with st.spinner("Computing embedding..."):
        try:
            embedding = compute_embedding(
                features=features,
                method=method,
                viz_dims=viz_dims,
                seed=int(seed),
                pca_n_components=int(pca_n_components),
                tsne_perplexity=float(tsne_perplexity),
                umap_n_neighbors=int(umap_n_neighbors),
                umap_min_dist=float(umap_min_dist),
            )
        except Exception as exc:
            st.error(f"Embedding computation failed: {exc}")
            st.stop()

    viz_df = apply_reject_policy(feature_df, float(t_bad), float(t_good))
    viz_df["emb_1"] = embedding[:, 0]
    viz_df["emb_2"] = embedding[:, 1]
    if viz_dims == 3:
        viz_df["emb_3"] = embedding[:, 2]

    emb_cols = ["emb_1", "emb_2"] + (["emb_3"] if viz_dims == 3 else [])
    finite_mask = np.isfinite(viz_df[emb_cols].to_numpy(dtype=float)).all(axis=1)
    if not bool(np.all(finite_mask)):
        dropped = int((~finite_mask).sum())
        viz_df = viz_df.loc[finite_mask].copy()
        st.warning(f"Dropped {dropped} non-finite embedding points before plotting.")
    if viz_df.empty:
        st.error("No finite embedding points to plot for the current settings.")
        st.stop()

    metrics = accepted_metrics(viz_df)
    info_cols = st.columns(6)
    info_cols[0].metric("Samples", int(metrics["n_total"]))
    info_cols[1].metric("Accepted", int(metrics["n_accepted"]))
    info_cols[2].metric("Rejected", int(metrics["n_rejected"]))
    info_cols[3].metric("Coverage", f"{metrics['coverage']:.3f}")
    info_cols[4].metric("Recall*", f"{metrics['recall']:.3f}" if np.isfinite(metrics["recall"]) else "n/a")
    info_cols[5].metric(
        "Specificity*",
        f"{metrics['specificity']:.3f}" if np.isfinite(metrics["specificity"]) else "n/a",
    )
    st.caption("* Accepted-only metrics under reject policy.")

    with st.expander("Run Metadata", expanded=False):
        st.write({
            "checkpoint": checkpoint,
            "encoder": model_meta["encoder"],
            "use_mlp_head": model_meta.get("use_mlp_head", False),
            "use_efficient_probing": model_meta.get("use_efficient_probing", False),
            "efficient_probing": model_meta.get("efficient_probing", {}),
            "crop_size": model_meta["crop_size"],
            "split_dir": str(split_dir),
            "mask_mode": mask_mode,
            "method": method,
            "viz_dims": viz_dims,
            "device": str(device),
        })

    plot_df = viz_df
    if len(viz_df) > int(max_plot_points):
        rng = np.random.default_rng(int(seed))
        keep = np.sort(rng.choice(len(viz_df), size=int(max_plot_points), replace=False))
        plot_df = viz_df.iloc[keep].copy()
        st.warning(
            f"Plot downsampled to {len(plot_df)} points (from {len(viz_df)}) for UI stability. "
            "Metrics above still use all loaded samples."
        )

    fig = build_plot(plot_df, viz_dims=viz_dims, color_mode=color_mode)

    st.subheader("Embedding Plot")
    plot_event: Any | None = None
    try:
        plot_event = st.plotly_chart(
            fig,
            use_container_width=True,
            key="embedding_plot",
            on_select="rerun",
            selection_mode=("points",),
        )
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, key="embedding_plot")
        st.info("Point-click selection is unavailable in this Streamlit version. Use `Select Top Uncertain`.")

    if "selected_row_ids" not in st.session_state:
        st.session_state.selected_row_ids = []

    clicked_ids = extract_selected_row_ids(plot_event)
    if clicked_ids:
        st.session_state.selected_row_ids = sorted(set(clicked_ids))

    valid_ids = set(plot_df["row_id"].astype(int).tolist())
    st.session_state.selected_row_ids = [x for x in st.session_state.selected_row_ids if x in valid_ids]

    action_cols = st.columns([1, 1, 5])
    with action_cols[0]:
        if st.button("Clear Selection"):
            st.session_state.selected_row_ids = []
    with action_cols[1]:
        if st.button("Select Top Uncertain"):
            top = plot_df.sort_values("uncertainty", ascending=False).head(20)["row_id"].astype(int).tolist()
            st.session_state.selected_row_ids = sorted(set(st.session_state.selected_row_ids + top))

    selected_df = viz_df[viz_df["row_id"].isin(st.session_state.selected_row_ids)].copy()

    left, right = st.columns([1.1, 1.3])
    with left:
        st.subheader("Selected Image Preview")
        if selected_df.empty:
            st.info("Click points in the plot, or use `Select Top Uncertain`.")
        else:
            last_row_id = st.session_state.selected_row_ids[-1]
            row = viz_df[viz_df["row_id"] == last_row_id].iloc[0]
            img_path = row["path"]
            if Path(img_path).is_file():
                st.image(load_rgb(img_path), caption=img_path, use_container_width=True)
            else:
                st.warning(f"Image file not found: {img_path}")
            st.write({
                "row_id": int(row["row_id"]),
                "true_label": row["true_label"],
                "pred_label": row["pred_label"],
                "accepted_label": row["accepted_label"],
                "confusion_class": row["confusion_class"],
                "p_good": float(row["p_good"]),
                "uncertainty": float(row["uncertainty"]),
            })

    with right:
        st.subheader("Selected Points")
        cols = [
            "row_id",
            "path",
            "true_label",
            "pred_label",
            "accepted_label",
            "accepted_status",
            "confusion_class",
            "p_good",
            "uncertainty",
        ]
        if selected_df.empty:
            st.info("No selected points yet.")
        else:
            selected_show = selected_df[cols].sort_values("row_id")
            st.dataframe(selected_show, use_container_width=True, height=360)
            csv_bytes = selected_show.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Selected CSV",
                data=csv_bytes,
                file_name="selected_embedding_points.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
