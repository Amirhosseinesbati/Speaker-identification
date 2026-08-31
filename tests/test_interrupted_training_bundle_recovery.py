import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.recover_interrupted_training_bundle import main


def test_recovery_builds_oof_bound_manifest_and_is_idempotent(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    profile = "interrupted"
    checkpoint_dir = tmp_path / profile
    bundle_dir = checkpoint_dir / "campp_best_bundle"
    bundle_dir.mkdir(parents=True)
    config = {
        "model": {"competition_num_known": 1, "encoder_type": "campp"},
        "data": {"split": {"scheme": "kfold", "folds": 3, "fold": 0, "seed": 42}},
        "training": {"seed": 42, "deterministic_algorithms": True},
    }
    class_map = {"unknown": 0, "known": 1}
    history = [
        {"epoch": 1, "val_macro_f1": 0.8, "val_ema_macro_f1": 0.79},
        {"epoch": 2, "val_macro_f1": 0.81, "val_ema_macro_f1": 0.80},
    ]
    selected = {
        "epoch": 2,
        "weight_variant": "raw",
        "val_macro_f1": 0.81,
        "model_state_dict": {"weight": torch.tensor([1.0])},
        "config": config,
        "class_map": class_map,
    }
    latest = dict(selected, training_history=history)
    torch.save(selected, checkpoint_dir / "campp_best.pt")
    torch.save(selected, checkpoint_dir / "campp_best_raw.pt")
    torch.save(latest, checkpoint_dir / "campp_latest.pt")
    np.savez_compressed(bundle_dir / "oof_predictions.npz", files=np.array(["a.wav"]))

    argv = [
        "recover_interrupted_training_bundle.py",
        "--profile", profile,
        "--checkpoint-root", str(tmp_path),
    ]
    monkeypatch.setattr("sys.argv", argv)
    assert main() == 0
    first = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest_bytes = (bundle_dir / "manifest.json").read_bytes()
    assert first["oof_predictions_sha256"]
    assert main() == 0
    assert (bundle_dir / "manifest.json").read_bytes() == manifest_bytes
    assert '"status": "already_valid"' in capsys.readouterr().out


def test_recovery_requires_explicit_permission_to_regenerate_oof(
    tmp_path: Path, monkeypatch,
) -> None:
    profile = "interrupted"
    checkpoint_dir = tmp_path / profile
    checkpoint_dir.mkdir(parents=True)
    config = {
        "model": {"competition_num_known": 1, "encoder_type": "campp"},
        "data": {"split": {"scheme": "kfold", "folds": 3, "fold": 0, "seed": 42}},
        "training": {"seed": 42, "deterministic_algorithms": True},
    }
    checkpoint = {
        "epoch": 1,
        "weight_variant": "raw",
        "val_macro_f1": 0.8,
        "model_state_dict": {"weight": torch.tensor([1.0])},
        "config": config,
        "class_map": {"unknown": 0, "known": 1},
    }
    latest = dict(
        checkpoint,
        training_history=[{"epoch": 1, "val_macro_f1": 0.8}],
    )
    torch.save(checkpoint, checkpoint_dir / "campp_best.pt")
    torch.save(checkpoint, checkpoint_dir / "campp_best_raw.pt")
    torch.save(latest, checkpoint_dir / "campp_latest.pt")
    monkeypatch.setattr("sys.argv", [
        "recover_interrupted_training_bundle.py",
        "--profile", profile,
        "--checkpoint-root", str(tmp_path),
    ])

    with pytest.raises(RuntimeError, match="--regenerate-oof"):
        main()


def test_regenerate_oof_accepts_list_backed_fold_splits(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    profile = "interrupted"
    checkpoint_dir = tmp_path / profile
    checkpoint_dir.mkdir(parents=True)
    config = {
        "model": {"competition_num_known": 1, "encoder_type": "campp"},
        "data": {"split": {"scheme": "kfold", "folds": 3, "fold": 0, "seed": 42}},
        "training": {"seed": 42, "deterministic_algorithms": True},
    }
    checkpoint = {
        "epoch": 1,
        "weight_variant": "raw",
        "val_macro_f1": 0.8,
        "model_state_dict": {"weight": torch.tensor([1.0])},
        "config": config,
        "class_map": {"unknown": 0, "known": 1},
    }
    latest = dict(
        checkpoint,
        training_history=[{"epoch": 1, "val_macro_f1": 0.8}],
    )
    torch.save(checkpoint, checkpoint_dir / "campp_best.pt")
    torch.save(checkpoint, checkpoint_dir / "campp_best_raw.pt")
    torch.save(latest, checkpoint_dir / "campp_latest.pt")
    monkeypatch.setattr(
        "scripts.recover_interrupted_training_bundle.rebuild_exact_splits",
        lambda *_: ([(None, "validation")], {}),
    )

    def fake_evaluate(**kwargs) -> None:
        assert kwargs["val_df"] == "validation"
        bundle_dir = checkpoint_dir / "campp_best_bundle"
        bundle_dir.mkdir()
        np.savez_compressed(
            bundle_dir / "oof_predictions.npz", files=np.array(["a.wav"])
        )
        from src.model_artifacts import create_training_bundle
        create_training_bundle(
            checkpoint_dir / "campp_best.pt",
            config,
            checkpoint["class_map"],
            latest["training_history"],
            {"selected_weight_variant": "raw"},
        )

    monkeypatch.setattr(
        "scripts.recover_interrupted_training_bundle.evaluate_model.entrypoint",
        fake_evaluate,
    )
    monkeypatch.setattr("sys.argv", [
        "recover_interrupted_training_bundle.py",
        "--profile", profile,
        "--checkpoint-root", str(tmp_path),
        "--regenerate-oof",
    ])
    assert main() == 0
    assert '"status": "recovered_oof_and_bundle"' in capsys.readouterr().out
