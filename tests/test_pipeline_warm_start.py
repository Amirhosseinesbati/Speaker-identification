from pathlib import Path

import pytest
import torch

from src.pipelines.steps import _load_warm_start_checkpoint


def _config(checkpoint_path: Path, *, fold: int = 0) -> dict:
    return {
        "model": {"encoder_type": "campp"},
        "data": {
            "split": {"scheme": "kfold", "folds": 3, "fold": fold, "seed": 42}
        },
        "training": {"warm_start_checkpoint": str(checkpoint_path)},
    }


def _write_checkpoint(path: Path, model: torch.nn.Module) -> dict:
    class_map = {"unknown": 0, "speaker-a": 1}
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_map": class_map,
        "config": _config(path),
        "epoch": 12,
        "val_macro_f1": 0.91,
    }, path)
    return class_map


def test_warm_start_restores_weights_and_provenance(tmp_path: Path) -> None:
    source = torch.nn.Linear(3, 2)
    with torch.no_grad():
        source.weight.fill_(0.25)
        source.bias.fill_(-0.5)
    checkpoint_path = tmp_path / "source.pt"
    class_map = _write_checkpoint(checkpoint_path, source)

    target = torch.nn.Linear(3, 2)
    receipt = _load_warm_start_checkpoint(
        target, _config(checkpoint_path), class_map
    )

    assert receipt is not None
    assert receipt["source_epoch"] == 12
    assert len(receipt["sha256"]) == 64
    assert torch.equal(target.weight, source.weight)
    assert torch.equal(target.bias, source.bias)


def test_warm_start_rejects_split_mismatch(tmp_path: Path) -> None:
    source = torch.nn.Linear(3, 2)
    checkpoint_path = tmp_path / "source.pt"
    class_map = _write_checkpoint(checkpoint_path, source)

    with pytest.raises(ValueError, match="split mismatch"):
        _load_warm_start_checkpoint(
            torch.nn.Linear(3, 2),
            _config(checkpoint_path, fold=1),
            class_map,
        )
