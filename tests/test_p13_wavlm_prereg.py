from __future__ import annotations

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs" / "experiments" / (
    "p13-wavlm-baseplus-layerweighted-known446-ood-complement-oof-f0.yaml"
)
PREREG = ROOT / "configs" / "analyses" / (
    "p13-wavlm-baseplus-layerweighted-known446-ood-complement-oof-f0.prereg.json"
)


def test_p13_profile_matches_preregistered_ssl_frontend() -> None:
    profile = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    wavlm = profile["model"]["encoder_config"]["wavlm"]
    training = profile["training"]
    gate = prereg["fold0_gate"]

    assert prereg["status"] == (
        "preregistered_before_weight_acquisition_and_fold0_metric_evaluation"
    )
    assert prereg["leaderboard_used_for_selection"] is False
    assert profile["data"]["split"] == {
        "scheme": "kfold", "folds": 3, "fold": 0, "seed": 42,
    }
    assert profile["model"]["speaker_target_scope"] == "known"
    assert wavlm["base_model"] == "microsoft/wavlm-base-plus"
    assert wavlm["freeze_encoder"] is True
    assert wavlm["layer_aggregation"] == "weighted_sum"
    assert wavlm["pooling_type"] == "attentive"
    assert profile["audio"]["num_train_windows"] == 2
    assert profile["audio"]["max_eval_windows"] == 8
    active = profile["hardware"]["mode"]
    assert profile["hardware"]["profiles"][active]["batch_size"] == 28
    assert training["epochs"] == 40
    assert training["early_stopping_start_epoch"] == 10
    assert training["early_stopping_patience"] == 8
    assert gate["standalone_min_raw_macro_f1"] == 0.9269211906147802
    assert gate["fixed_50_50_probability_blend_with_p0_min_macro_gain"] == 0.002
    assert gate["max_known_accuracy_drop_vs_p0"] == 0.001
    assert gate["max_ood_f1_drop_vs_p0"] == 0.001
    assert gate["min_p0_error_rescue_rate"] == 0.25
    assert gate["require_backbone_zero_trainable_parameters"] is True
    assert gate["require_layer_weight_count"] == 13
    resolution = prereg["operational_resolution_before_fold0_metric_evaluation"]
    assert resolution["training_probe"]["selected_batch_size"] == 28
    assert resolution["evaluation_probe"]["measured_batch_size"] == 28
    assert resolution["model_invariants"]["wavlm_trainable_parameters"] == 0
    assert resolution["model_invariants"]["layer_weight_count"] == 13


def test_p13_changes_no_posthoc_decision_dimension() -> None:
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    change = prereg["locked_change"]

    assert change["decision"] == "Raw probability-average direct argmax"
    assert change["tuned_thresholds_or_blends"] == 0
    assert change["ema_role"] == "diagnostic_only"
    assert change["logit_average_role"] == "diagnostic_only"
    assert prereg["fold0_gate"]["all_checks_required"] is True
