from __future__ import annotations

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
NAME = "p13b-wavlm-baseplus-layerweighted-deterministic-known446-ood-complement-oof-f0"
PROFILE = ROOT / "configs" / "experiments" / f"{NAME}.yaml"
PREREG = ROOT / "configs" / "analyses" / f"{NAME}.prereg.json"


def test_p13b_is_a_single_runtime_change_from_p13() -> None:
    profile = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
    p13 = yaml.safe_load((
        ROOT / "configs" / "experiments" /
        "p13-wavlm-baseplus-layerweighted-known446-ood-complement-oof-f0.yaml"
    ).read_text(encoding="utf-8"))
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))

    wavlm = profile["model"]["encoder_config"]["wavlm"]
    assert wavlm["frozen_backbone_eval"] is True
    assert wavlm["freeze_encoder"] is True
    assert wavlm["layer_aggregation"] == "weighted_sum"
    assert profile["data"] == p13["data"]
    assert profile["audio"] == p13["audio"]
    assert profile["training"] == p13["training"]
    assert profile["model"]["speaker_head_config"] == p13["model"]["speaker_head_config"]
    assert profile["model"]["speaker_target_scope"] == p13["model"]["speaker_target_scope"]
    assert prereg["leaderboard_used_for_selection"] is False
    assert prereg["locked_change"]["tuned_thresholds_or_blends"] == 0


def test_p13b_locks_runtime_receipt_and_same_fold0_gate() -> None:
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    gate = prereg["fold0_gate"]

    assert prereg["status"] == (
        "preregistered_before_p13_terminal_result_and_p13b_fold0_metric_evaluation"
    )
    assert prereg["selection"]["futility_check_epoch"] == 15
    assert prereg["selection"]["futility_min_best_raw_macro_f1"] == 0.88
    assert gate["standalone_min_raw_macro_f1"] == 0.9269211906147802
    assert gate["fixed_50_50_probability_blend_with_p0_min_macro_gain"] == 0.002
    assert gate["max_known_accuracy_drop_vs_p0"] == 0.001
    assert gate["max_ood_f1_drop_vs_p0"] == 0.001
    assert gate["require_training_probe_wavlm_eval_mode"] is True
