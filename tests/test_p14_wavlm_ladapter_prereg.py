from __future__ import annotations

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
P13B = ROOT / "configs/experiments/p13b-wavlm-baseplus-layerweighted-deterministic-known446-ood-complement-oof-f0.yaml"
P14 = ROOT / "configs/experiments/p14-wavlm-baseplus-ladapter-known446-ood-complement-oof-f0.yaml"
PREREG = ROOT / "configs/analyses/p14-wavlm-baseplus-ladapter-known446-ood-complement-oof-f0.prereg.json"


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_p14_changes_only_the_preregistered_adapter_dimensions() -> None:
    p13b = _load_yaml(P13B)
    p14 = _load_yaml(P14)

    for section in ("data", "audio"):
        assert p14[section] == p13b[section]
    assert p14["model"]["competition_num_known"] == p13b["model"]["competition_num_known"]
    assert p14["model"]["speaker_target_scope"] == p13b["model"]["speaker_target_scope"]
    assert p14["model"]["speaker_head_type"] == p13b["model"]["speaker_head_type"]
    assert p14["model"]["speaker_head_config"] == p13b["model"]["speaker_head_config"]
    assert p14["training"]["loss"] == p13b["training"]["loss"]
    for key in ("seed", "epochs", "early_stopping_start_epoch", "early_stopping_patience", "weight_decay", "schedule", "warmup_ratio", "min_lr_ratio", "ema_enabled", "ema_decay"):
        assert p14["training"][key] == p13b["training"][key]

    wavlm = p14["model"]["encoder_config"]["wavlm"]
    assert wavlm["layer_aggregation"] == "layer_adapter"
    assert wavlm["layer_adapter_dim"] == 512
    assert wavlm["layer_adapter_activation"] == "relu"
    assert wavlm["layer_adapter_layer_norm"] is True
    assert wavlm["layer_adapter_tune_backbone_layer_norms"] is True
    assert wavlm["frozen_backbone_eval"] is False
    assert p14["experiment"]["operational_preflight"]["require_layer_weight_count"] == 12
    assert p14["training"]["learning_rate"] == 5.0e-4
    assert p14["training"]["encoder_lr"] == 5.0e-4


def test_p14_preregistration_locks_parameter_scope_and_gate() -> None:
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    assert prereg["status"].startswith("preregistered_before_p13b_terminal_result")
    assert prereg["leaderboard_used_for_selection"] is False
    assert prereg["parameter_contract"] == {
        "wavlm_total_parameters": 94381936,
        "wavlm_trainable_parameters": 36864,
        "wavlm_trainable_scope": "transformer encoder layer_norm parameters only",
        "layer_adapter_count": 12,
        "layer_adapter_parameters": 4737024,
        "layer_weight_count": 12,
        "layer_adapter_activation": "relu",
        "layer_adapter_output_dim": 512,
        "runtime_layerdrop": 0.0,
    }
    gate = prereg["fold0_gate"]
    assert gate["standalone_min_raw_macro_f1"] == 0.9269211906147802
    assert gate["fixed_50_50_probability_blend_with_p0_min_macro_gain"] == 0.002
    assert gate["max_known_accuracy_drop_vs_p0"] == 0.001
    assert gate["max_ood_f1_drop_vs_p0"] == 0.001
    assert gate["min_p0_error_rescue_rate"] == 0.25
    assert prereg["selection"]["futility_check_epoch"] == 15
    assert prereg["selection"]["futility_min_best_raw_macro_f1"] == 0.88
