from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("sklearn")

from cell_size.classifier import trainer


def _base_cfg(*, train_with_val: bool, cv_enabled: bool) -> SimpleNamespace:
    return SimpleNamespace(
        crop_size=224,
        gpu=False,
        train_with_val=train_with_val,
        cross_validation=SimpleNamespace(enabled=cv_enabled, k_folds=5),
        wandb=SimpleNamespace(enabled=False),
    )


def test_train_with_val_incompatible_with_cross_validation(tmp_path: Path) -> None:
    cfg = _base_cfg(train_with_val=True, cv_enabled=True)
    with pytest.raises(ValueError, match="incompatible with cross-validation"):
        trainer.train(tmp_path, tmp_path / "out", cfg)


def test_train_with_val_requires_test_split(tmp_path: Path) -> None:
    cfg = SimpleNamespace(train_with_val=True)
    with pytest.raises(FileNotFoundError, match="requires a test split"):
        trainer._train_standard(  # noqa: SLF001 - test private helper behavior
            crops_dir=tmp_path,
            output_dir=tmp_path / "out",
            cfg=cfg,
            device=torch.device("cpu"),
            crop_size=224,
        )


def test_make_run_name_marks_train_with_val_mode() -> None:
    cfg = SimpleNamespace(
        encoder="resnet18",
        learning_rate=0.001,
        freeze_encoder=True,
        use_efficient_probing=False,
        use_mlp_head=False,
        batch_size=32,
        train_with_val=True,
    )
    name = trainer._make_run_name(cfg)  # noqa: SLF001 - test private helper behavior
    assert "trval_testmon" in name
