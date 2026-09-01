from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/preregister_ecapa_se_bn_followup.py"
P8_PROFILE = ROOT / (
    "configs/experiments/"
    "p8-ecapa-frozen-known446-ood-complement-oof-f0.yaml"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("p9_preregister", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_parent_checkpoint(path: Path, raw: float = 0.92) -> None:
    config = yaml.safe_load(P8_PROFILE.read_text(encoding="utf-8"))
    # Resolved runtime configs stored by the pipeline omit file-loader metadata;
    # profile binding must therefore also be proven from logging.checkpoint_dir.
    config.pop("_meta")
    history = [
        {
            "epoch": 1,
            "val_macro_f1": raw - 0.01,
            "val_known_acc": 0.91,
            "val_ood_f1": 0.90,
            "val_logit_avg_macro_f1": raw - 0.02,
            "val_ema_macro_f1": raw - 0.03,
        },
        {
            "epoch": 2,
            "val_macro_f1": raw,
            "val_known_acc": 0.93,
            "val_ood_f1": 0.92,
            "val_logit_avg_macro_f1": raw - 0.01,
            "val_ema_macro_f1": raw - 0.015,
        },
    ]
    torch.save(
        {
            "model_state_dict": {"weight": torch.ones(1)},
            "config": config,
            "class_map": {str(index): index for index in range(1001)},
            "epoch": 2,
            "weight_variant": "raw",
            "training_history": history,
            "val_macro_f1": raw,
        },
        path,
    )


def test_preregister_locks_parent_receipt_and_single_variable(tmp_path: Path) -> None:
    module = _load_module()
    checkpoint = tmp_path / "ecapa_best_raw.pt"
    output_profile = tmp_path / "p9.yaml"
    output_contract = tmp_path / "p9.prereg.json"
    _write_parent_checkpoint(checkpoint)

    receipt = module.preregister(
        checkpoint, P8_PROFILE, output_profile, output_contract
    )
    profile = yaml.safe_load(output_profile.read_text(encoding="utf-8"))
    contract = json.loads(output_contract.read_text(encoding="utf-8"))

    assert receipt["parent_epoch"] == 2
    assert receipt["parent_raw_macro_f1"] == 0.92
    assert receipt["standalone_gate"] == module.P8_STANDALONE_GATE
    assert receipt["profile_sha256"] == module.sha256_file(output_profile)
    assert contract["profile_sha256"] == receipt["profile_sha256"]
    assert contract["parent"]["checkpoint_sha256"] == module.sha256_file(
        checkpoint
    )
    assert contract["single_variable"] == "enable_all_group_ecapa_se_bn_adapter"
    assert contract["no_hyperparameter_sweep"] is True
    assert contract["no_leaderboard_selection"] is True

    assert profile["_meta"]["name"] == module.P9_PROFILE
    assert profile["data"]["split"] == module.EXPECTED_SPLIT
    ecapa = profile["model"]["encoder_config"]["ecapa"]
    assert ecapa["freeze_encoder"] is True
    assert ecapa["unfreeze_last_n_blocks"] == 0
    assert ecapa["adapter_mode"] == "se_bn"
    training = profile["training"]
    assert training["warm_start_checkpoint"] == checkpoint.as_posix()
    assert training["learning_rate"] == 0.0
    assert training["encoder_lr"] == 1e-5
    assert training["epochs"] == 45
    assert training["early_stopping_start_epoch"] == 5
    assert training["early_stopping_patience"] == 12


def test_preregister_rejects_parent_that_already_passed_p8_gate(
    tmp_path: Path,
) -> None:
    module = _load_module()
    checkpoint = tmp_path / "ecapa_best_raw.pt"
    _write_parent_checkpoint(checkpoint, raw=module.P8_STANDALONE_GATE)

    with pytest.raises(RuntimeError, match="conditional P9 trigger is false"):
        module.preregister(
            checkpoint,
            P8_PROFILE,
            tmp_path / "p9.yaml",
            tmp_path / "p9.prereg.json",
        )
