"""Training loop with early stopping, weighted loss, metrics, and optional WandB."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from torchvision.datasets import ImageFolder

from cell_size.classifier.dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_dataset,
    get_eval_transforms,
    get_train_transforms,
)
from cell_size.classifier.models import build_model
from cell_size.classifier.experiment_tracking import write_epoch_results_csv

logger = logging.getLogger(__name__)


@dataclass
class TrainMetrics:
    loss: float = 0.0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0


@dataclass
class TrainResult:
    best_checkpoint: Path | None = None
    best_val_f1: float = 0.0
    test_metrics: TrainMetrics | None = None
    all_test_labels: list[int] = field(default_factory=list)
    all_test_preds: list[int] = field(default_factory=list)


def _compute_pos_weight(dataset: ImageFolder) -> torch.Tensor:
    """Compute pos_weight for BCEWithLogitsLoss from class distribution.

    ImageFolder sorts classes alphabetically, so typically bad=0, good=1.
    ``pos_weight = n_negative / n_positive``.
    """
    targets = np.array(dataset.targets)
    good_idx = dataset.class_to_idx.get("good", 1)
    n_pos = (targets == good_idx).sum()
    n_neg = len(targets) - n_pos
    weight = n_neg / max(n_pos, 1)
    logger.info("Class distribution: %d good, %d bad -> pos_weight=%.2f", n_pos, n_neg, weight)
    return torch.tensor([weight], dtype=torch.float32)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    good_idx: int,
) -> TrainMetrics:
    """Run one epoch (train or eval depending on whether optimizer is given)."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running_loss = 0.0
    all_labels: list[int] = []
    all_preds: list[int] = []

    ctx = torch.no_grad() if not is_train else torch.enable_grad()
    desc = "train" if is_train else "val"
    with ctx:
        for images, targets in tqdm(loader, desc=desc, leave=False):
            images = images.to(device)
            binary_targets = (targets == good_idx).float().unsqueeze(1).to(device)

            logits = model(images)
            loss = criterion(logits, binary_targets)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).long().squeeze(1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend((targets == good_idx).long().tolist())

    n = len(all_labels)
    avg_loss = running_loss / max(n, 1)
    return TrainMetrics(
        loss=avg_loss,
        accuracy=accuracy_score(all_labels, all_preds),
        precision=precision_score(all_labels, all_preds, zero_division=0),
        recall=recall_score(all_labels, all_preds, zero_division=0),
        f1=f1_score(all_labels, all_preds, zero_division=0),
    )


def train(
    crops_dir: Path,
    output_dir: Path,
    cfg: Any,
) -> TrainResult:
    """Full training pipeline: build model, train, evaluate, save checkpoint.

    Supports standard train/val/test split and optional k-fold cross-validation.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    crop_size = int(cfg.crop_size)
    device = torch.device("cuda" if cfg.gpu and torch.cuda.is_available() else "cpu")
    logger.info("Training device: %s", device)

    _init_wandb(cfg)

    if cfg.cross_validation.enabled:
        return _train_kfold(crops_dir, output_dir, cfg, device, crop_size)

    return _train_standard(crops_dir, output_dir, cfg, device, crop_size)


def _train_standard(
    crops_dir: Path,
    output_dir: Path,
    cfg: Any,
    device: torch.device,
    crop_size: int,
) -> TrainResult:
    """Standard train/val/test training."""
    train_ds = build_dataset(crops_dir, "train", crop_size)
    val_ds = build_dataset(crops_dir, "val", crop_size)

    good_idx = train_ds.class_to_idx.get("good", 1)
    pos_weight = _compute_pos_weight(train_ds).to(device)

    model = build_model(cfg.encoder, cfg.pretrained, cfg.freeze_encoder).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(cfg.learning_rate),
        weight_decay=float(cfg.weight_decay),
    )

    train_loader = DataLoader(
        train_ds, batch_size=int(cfg.batch_size), shuffle=True, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=int(cfg.batch_size), shuffle=False, num_workers=4, pin_memory=True,
    )

    best_val_f1 = 0.0
    patience_counter = 0
    patience = int(cfg.early_stopping_patience)
    checkpoint_path = output_dir / "best_model.pt"

    n_samples = int(cfg.wandb.n_samples) if cfg.wandb.enabled else 0
    sample_indices = (
        _select_fixed_samples(val_ds, n_samples, good_idx, int(cfg.seed))
        if n_samples > 0
        else []
    )

    epoch_rows: list[dict[str, Any]] = []

    for epoch in range(1, int(cfg.epochs) + 1):
        train_m = _run_epoch(model, train_loader, criterion, optimizer, device, good_idx)
        val_m = _run_epoch(model, val_loader, criterion, None, device, good_idx)

        epoch_rows.append({
            "epoch": epoch,
            "train_loss": train_m.loss,
            "train_accuracy": train_m.accuracy,
            "train_precision": train_m.precision,
            "train_recall": train_m.recall,
            "train_f1": train_m.f1,
            "val_loss": val_m.loss,
            "val_accuracy": val_m.accuracy,
            "val_precision": val_m.precision,
            "val_recall": val_m.recall,
            "val_f1": val_m.f1,
        })

        logger.info(
            "Epoch %d/%d  train_loss=%.4f train_f1=%.4f  val_loss=%.4f val_f1=%.4f",
            epoch,
            cfg.epochs,
            train_m.loss,
            train_m.f1,
            val_m.loss,
            val_m.f1,
        )
        _log_wandb(epoch, train_m, val_m)
        _log_wandb_images(epoch, model, val_ds, sample_indices, device, good_idx)

        if val_m.f1 > best_val_f1:
            best_val_f1 = val_m.f1
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "encoder": cfg.encoder,
                "val_f1": best_val_f1,
                "class_to_idx": train_ds.class_to_idx,
                "crop_size": crop_size,
            }, str(checkpoint_path))
            logger.info("Saved best model (val_f1=%.4f) -> %s", best_val_f1, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping at epoch %d (patience=%d)", epoch, patience)
                break

    # Write per-epoch metrics for this run
    try:
        write_epoch_results_csv(output_dir, epoch_rows)
    except Exception:
        logger.exception("Failed to write epoch_results.csv")

    result = TrainResult(best_checkpoint=checkpoint_path, best_val_f1=best_val_f1)

    test_dir = crops_dir / "test"
    if test_dir.is_dir():
        model.load_state_dict(torch.load(str(checkpoint_path), weights_only=False)["model_state_dict"])
        test_ds = build_dataset(crops_dir, "test", crop_size)
        test_loader = DataLoader(
            test_ds, batch_size=int(cfg.batch_size), shuffle=False, num_workers=4,
        )
        test_m = _run_epoch(model, test_loader, criterion, None, device, good_idx)
        result.test_metrics = test_m
        logger.info(
            "Test  acc=%.4f  prec=%.4f  rec=%.4f  f1=%.4f",
            test_m.accuracy,
            test_m.precision,
            test_m.recall,
            test_m.f1,
        )

        all_labels, all_preds = _collect_predictions(model, test_loader, device, good_idx)
        result.all_test_labels = all_labels
        result.all_test_preds = all_preds
        logger.info("\n%s", classification_report(
            all_labels, all_preds, target_names=["bad", "good"], zero_division=0,
        ))

    _finish_wandb()
    return result


def _train_kfold(
    crops_dir: Path,
    output_dir: Path,
    cfg: Any,
    device: torch.device,
    crop_size: int,
) -> TrainResult:
    """K-fold cross-validation training."""
    k = int(cfg.cross_validation.k_folds)
    seed = int(cfg.seed)

    full_ds = ImageFolder(
        str(crops_dir / "train"),
        transform=get_train_transforms(crop_size),
    )
    good_idx = full_ds.class_to_idx.get("good", 1)
    targets = np.array(full_ds.targets)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    fold_f1s: list[float] = []
    best_fold_f1 = 0.0
    best_checkpoint = output_dir / "best_model.pt"

    epoch_rows: list[dict[str, Any]] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(targets, targets), 1):
        logger.info("=== Fold %d/%d ===", fold, k)

        train_subset = Subset(full_ds, train_idx.tolist())

        eval_ds = ImageFolder(
            str(crops_dir / "train"),
            transform=get_eval_transforms(crop_size),
        )
        val_subset = Subset(eval_ds, val_idx.tolist())

        n_pos = (targets[train_idx] == good_idx).sum()
        n_neg = len(train_idx) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)

        model = build_model(cfg.encoder, cfg.pretrained, cfg.freeze_encoder).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=float(cfg.learning_rate),
            weight_decay=float(cfg.weight_decay),
        )

        train_loader = DataLoader(
            train_subset, batch_size=int(cfg.batch_size), shuffle=True, num_workers=4,
        )
        val_loader = DataLoader(
            val_subset, batch_size=int(cfg.batch_size), shuffle=False, num_workers=4,
        )

        fold_best_f1 = 0.0
        patience_counter = 0
        patience = int(cfg.early_stopping_patience)

        for epoch in range(1, int(cfg.epochs) + 1):
            train_m = _run_epoch(model, train_loader, criterion, optimizer, device, good_idx)
            val_m = _run_epoch(model, val_loader, criterion, None, device, good_idx)

            _log_wandb_kfold_epoch(epoch, fold, train_m, val_m)

            epoch_rows.append({
                "fold": fold,
                "epoch": epoch,
                "train_loss": train_m.loss,
                "train_accuracy": train_m.accuracy,
                "train_precision": train_m.precision,
                "train_recall": train_m.recall,
                "train_f1": train_m.f1,
                "val_loss": val_m.loss,
                "val_accuracy": val_m.accuracy,
                "val_precision": val_m.precision,
                "val_recall": val_m.recall,
                "val_f1": val_m.f1,
            })

            if val_m.f1 > fold_best_f1:
                fold_best_f1 = val_m.f1
                patience_counter = 0
                if fold_best_f1 > best_fold_f1:
                    best_fold_f1 = fold_best_f1
                    torch.save({
                        "epoch": epoch,
                        "fold": fold,
                        "model_state_dict": model.state_dict(),
                        "encoder": cfg.encoder,
                        "val_f1": best_fold_f1,
                        "class_to_idx": full_ds.class_to_idx,
                        "crop_size": crop_size,
                    }, str(best_checkpoint))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        fold_f1s.append(fold_best_f1)
        logger.info("Fold %d best val F1: %.4f", fold, fold_best_f1)

        try:
            import wandb

            if _wandb_run is not None:
                wandb.log({
                    "fold": fold,
                    f"fold{fold}/best_val_f1": fold_best_f1,
                    "kfold/best_so_far_val_f1": best_fold_f1,
                })
        except Exception:
            pass

    mean_f1 = float(np.mean(fold_f1s))
    std_f1 = float(np.std(fold_f1s))
    logger.info("K-fold results: mean_f1=%.4f +/- %.4f", mean_f1, std_f1)

    # Write per-epoch metrics for this run (includes fold column)
    try:
        write_epoch_results_csv(output_dir, epoch_rows)
    except Exception:
        logger.exception("Failed to write epoch_results.csv")

    try:
        import wandb

        if _wandb_run is not None:
            wandb.log({
                "kfold/mean_f1": mean_f1,
                "kfold/std_f1": std_f1,
            })
    except Exception:
        pass

    result = TrainResult(best_checkpoint=best_checkpoint, best_val_f1=best_fold_f1)

    test_dir = crops_dir / "test"
    if test_dir.is_dir() and best_checkpoint.is_file():
        ckpt = torch.load(str(best_checkpoint), weights_only=False)
        model = build_model(cfg.encoder, cfg.pretrained, False).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        pos_weight = _compute_pos_weight(full_ds).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        test_ds = build_dataset(crops_dir, "test", crop_size)
        test_loader = DataLoader(
            test_ds, batch_size=int(cfg.batch_size), shuffle=False, num_workers=4,
        )
        test_m = _run_epoch(model, test_loader, criterion, None, device, good_idx)
        result.test_metrics = test_m

        all_labels, all_preds = _collect_predictions(model, test_loader, device, good_idx)
        result.all_test_labels = all_labels
        result.all_test_preds = all_preds
        logger.info("Test F1: %.4f", test_m.f1)

    _finish_wandb()
    return result


def _collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    good_idx: int,
) -> tuple[list[int], list[int]]:
    """Collect ground-truth and predicted labels from a DataLoader."""
    model.eval()
    all_labels: list[int] = []
    all_preds: list[int] = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            logits = model(images)
            preds = (torch.sigmoid(logits) >= 0.5).long().squeeze(1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend((targets == good_idx).long().tolist())
    return all_labels, all_preds


# ---------------------------------------------------------------------------
# Optional WandB integration
# ---------------------------------------------------------------------------

_wandb_run = None


def _make_run_name(cfg: Any) -> str:
    """Build a human-readable WandB run name from training config."""
    from datetime import datetime

    encoder = cfg.encoder
    lr = float(cfg.learning_rate)
    frozen = "frozen" if cfg.freeze_encoder else "finetune"
    bs = int(cfg.batch_size)
    ts = datetime.now().strftime("%m%d-%H%M")
    return f"{encoder}_{frozen}_lr{lr}_bs{bs}_{ts}"


def _init_wandb(cfg: Any) -> None:
    global _wandb_run
    if not cfg.wandb.enabled:
        return
    try:
        import wandb

        run_name = cfg.wandb.run_name if cfg.wandb.run_name else _make_run_name(cfg)
        _wandb_run = wandb.init(
            project=cfg.wandb.project,
            name=run_name,
            config={
                "encoder": cfg.encoder,
                "crop_size": cfg.crop_size,
                "epochs": cfg.epochs,
                "batch_size": cfg.batch_size,
                "learning_rate": cfg.learning_rate,
                "freeze_encoder": cfg.freeze_encoder,
            },
        )
        logger.info("WandB run initialised: %s", _wandb_run.url)
    except ImportError:
        logger.warning("wandb not installed; install with: pip install wandb")
    except Exception as exc:
        logger.warning("WandB init failed: %s", exc)


def _log_wandb(epoch: int, train_m: TrainMetrics, val_m: TrainMetrics) -> None:
    if _wandb_run is None:
        return
    try:
        import wandb
        wandb.log({
            "epoch": epoch,
            "train/loss": train_m.loss,
            "train/f1": train_m.f1,
            "train/accuracy": train_m.accuracy,
            "val/loss": val_m.loss,
            "val/f1": val_m.f1,
            "val/accuracy": val_m.accuracy,
            "val/precision": val_m.precision,
            "val/recall": val_m.recall,
        })
    except Exception:
        pass


def _log_wandb_kfold_epoch(
    epoch: int,
    fold: int,
    train_m: TrainMetrics,
    val_m: TrainMetrics,
) -> None:
    """Log k-fold metrics to WandB with fold-prefixed keys."""
    if _wandb_run is None:
        return
    try:
        import wandb

        fold_prefix = f"fold{fold}"
        wandb.log({
            "epoch": epoch,
            "fold": fold,
            f"{fold_prefix}/train/loss": train_m.loss,
            f"{fold_prefix}/train/f1": train_m.f1,
            f"{fold_prefix}/train/accuracy": train_m.accuracy,
            f"{fold_prefix}/val/loss": val_m.loss,
            f"{fold_prefix}/val/f1": val_m.f1,
            f"{fold_prefix}/val/accuracy": val_m.accuracy,
            f"{fold_prefix}/val/precision": val_m.precision,
            f"{fold_prefix}/val/recall": val_m.recall,
        })
    except Exception:
        pass


def _denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalization and convert to uint8 HWC numpy array."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = tensor.cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


_sample_indices: list[int] | None = None


def _select_fixed_samples(
    dataset: ImageFolder,
    n_samples: int,
    good_idx: int,
    seed: int = 42,
) -> list[int]:
    """Select a fixed balanced set of sample indices from the dataset.

    Returns ``n_samples`` indices (half good, half bad) that stay constant
    across epochs so prediction evolution is visible.
    """
    global _sample_indices
    if _sample_indices is not None:
        return _sample_indices

    rng = np.random.RandomState(seed)
    targets = np.array(dataset.targets)

    n_per_class = n_samples // 2
    good_mask = targets == good_idx
    bad_mask = ~good_mask

    good_pool = np.where(good_mask)[0]
    bad_pool = np.where(bad_mask)[0]

    n_good = min(n_per_class, len(good_pool))
    n_bad = min(n_samples - n_good, len(bad_pool))

    selected_good = rng.choice(good_pool, size=n_good, replace=False).tolist() if n_good > 0 else []
    selected_bad = rng.choice(bad_pool, size=n_bad, replace=False).tolist() if n_bad > 0 else []

    _sample_indices = selected_good + selected_bad
    return _sample_indices


def _log_wandb_images(
    epoch: int,
    model: nn.Module,
    dataset: ImageFolder,
    sample_indices: list[int],
    device: torch.device,
    good_idx: int,
) -> None:
    """Log sample images with predictions to WandB as tables."""
    if _wandb_run is None or not sample_indices:
        return
    try:
        import wandb
    except ImportError:
        return

    class_names = {0: "bad", 1: "good"}
    model.eval()

    columns = ["image", "true_label", "predicted_label", "confidence", "correct"]
    samples_table = wandb.Table(columns=columns)
    misclassified_table = wandb.Table(columns=columns)

    with torch.no_grad():
        for idx in sample_indices:
            img_tensor, target = dataset[idx]
            true_binary = 1 if target == good_idx else 0
            true_label = class_names[true_binary]

            logit = model(img_tensor.unsqueeze(0).to(device))
            prob = float(torch.sigmoid(logit).item())
            pred_binary = 1 if prob >= 0.5 else 0
            pred_label = class_names[pred_binary]
            correct = pred_binary == true_binary

            img_np = _denormalize(img_tensor)
            caption = f"true={true_label} pred={pred_label} conf={prob:.3f}"
            wb_img = wandb.Image(img_np, caption=caption)

            row = [wb_img, true_label, pred_label, round(prob, 4), correct]
            samples_table.add_data(*row)
            if not correct:
                misclassified_table.add_data(*row)

    wandb.log({
        "val_samples": samples_table,
        "misclassified": misclassified_table,
        "epoch": epoch,
    })


def _finish_wandb() -> None:
    global _wandb_run, _sample_indices
    if _wandb_run is None:
        return
    try:
        import wandb
        wandb.finish()
    except Exception:
        pass
    _wandb_run = None
    _sample_indices = None
