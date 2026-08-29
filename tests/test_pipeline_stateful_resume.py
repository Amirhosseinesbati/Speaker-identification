from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from src.pipelines.steps import _restore_training_state_for_resume


def _source_config() -> dict:
    return {
        "model": {"encoder_type": "campp", "speaker_target_scope": "known"},
        "data": {
            "audio_dir": "data/audio",
            "split": {"scheme": "kfold", "folds": 3, "fold": 0, "seed": 42},
        },
        "audio": {"sample_rate": 16000, "ood_batch_ratio": 0.5},
        "augmentation": {"enabled": True, "rir_probability": 0.6},
        "hardware": {"profiles": {"gpu": {"batch_size": 48}}},
        "training": {
            "epochs": 200,
            "learning_rate": 1e-3,
            "weight_decay": 0.01,
            "schedule": "cosine",
            "freeze_epochs": 20,
            "early_stopping_patience": 20,
        },
    }


def _history(last_epoch: int) -> list[dict]:
    return [
        {
            "epoch": epoch,
            "train_loss": 2.0 / epoch,
            "val_loss": 1.0 + epoch / 100.0,
            "val_macro_f1": 0.90 + epoch / 1000.0,
            "val_ema_macro_f1": 0.89 + epoch / 1000.0,
            "val_ood_acc": 0.9,
            "val_speaker_acc": 0.9,
        }
        for epoch in range(1, last_epoch + 1)
    ]


def _write_checkpoint(
    path: Path,
    *,
    include_scheduler: bool = True,
    checkpoint_history: list[dict] | None = None,
) -> tuple[dict, torch.nn.Module, torch.optim.Optimizer, object]:
    class_map = {"unknown": 0, "speaker-a": 1}
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        model(torch.ones(2, 3)).sum().backward()
        optimizer.step()
        scheduler.step()

    payload = {
        "epoch": 3,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": _source_config(),
        "class_map": class_map,
        "training_history": (
            checkpoint_history if checkpoint_history is not None else _history(3)
        ),
        "val_macro_f1": 0.903,
    }
    if include_scheduler:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(payload, path)
    return class_map, model, optimizer, scheduler


def _target_state(config: dict):
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    return model, optimizer, scheduler


def _resume_config(checkpoint: Path) -> dict:
    config = copy.deepcopy(_source_config())
    config["training"]["resume_checkpoint"] = str(checkpoint)
    config["training"]["early_stopping_patience"] = 12
    return config


def test_stateful_resume_restores_all_training_state(tmp_path: Path) -> None:
    checkpoint = tmp_path / "source.pt"
    class_map, source_model, source_optimizer, source_scheduler = _write_checkpoint(
        checkpoint,
    )
    config = _resume_config(checkpoint)
    model, optimizer, scheduler = _target_state(config)

    receipt, history = _restore_training_state_for_resume(
        model, optimizer, scheduler, config, class_map,
    )

    assert receipt is not None
    assert receipt["source_epoch"] == 3
    assert receipt["start_epoch"] == 4
    assert receipt["ema_restored"] is False
    assert len(receipt["sha256"]) == 64
    assert len(receipt["source_history_sha256"]) == 64
    assert [row["epoch"] for row in history] == [1, 2, 3]
    assert torch.equal(model.weight, source_model.weight)
    assert torch.equal(model.bias, source_model.bias)
    assert (
        optimizer.state_dict()["state"].keys()
        == source_optimizer.state_dict()["state"].keys()
    )
    assert scheduler.last_epoch == source_scheduler.last_epoch == 3


def test_stateful_resume_uses_external_history_and_truncates_to_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "source.pt"
    class_map, _, _, _ = _write_checkpoint(
        checkpoint, checkpoint_history=_history(2),
    )
    history_path = tmp_path / "terminal_latest.pt"
    torch.save({"training_history": _history(5)}, history_path)
    config = _resume_config(checkpoint)
    config["training"]["resume_history_path"] = str(history_path)
    model, optimizer, scheduler = _target_state(config)

    receipt, history = _restore_training_state_for_resume(
        model, optimizer, scheduler, config, class_map,
    )

    assert receipt is not None
    assert [row["epoch"] for row in history] == [1, 2, 3]
    assert receipt["source_history_epochs"] == 3
    assert receipt["source_history_path"] == str(history_path.resolve())
    assert len(receipt["source_history_file_sha256"]) == 64


def test_stateful_resume_rejects_contract_change(tmp_path: Path) -> None:
    checkpoint = tmp_path / "source.pt"
    class_map, _, _, _ = _write_checkpoint(checkpoint)
    config = _resume_config(checkpoint)
    config["augmentation"]["rir_probability"] = 0.8
    model, optimizer, scheduler = _target_state(config)

    with pytest.raises(ValueError, match="scientific-contract mismatch"):
        _restore_training_state_for_resume(
            model, optimizer, scheduler, config, class_map,
        )


def test_stateful_resume_rejects_checkpoint_without_scheduler(tmp_path: Path) -> None:
    checkpoint = tmp_path / "source.pt"
    class_map, _, _, _ = _write_checkpoint(
        checkpoint, include_scheduler=False,
    )
    config = _resume_config(checkpoint)
    model, optimizer, scheduler = _target_state(config)

    with pytest.raises(ValueError, match="lacks scheduler_state_dict"):
        _restore_training_state_for_resume(
            model, optimizer, scheduler, config, class_map,
        )
