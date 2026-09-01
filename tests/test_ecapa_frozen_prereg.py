from __future__ import annotations

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/experiments/p8-ecapa-frozen-known446-ood-complement-oof-f0.yaml"
PREREG = ROOT / "configs/analyses/ecapa-frozen-known446-ood-complement-oof-f0.prereg.json"


def test_ecapa_frozen_profile_matches_preregistration() -> None:
    config = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))

    assert config["_meta"]["name"] == prereg["profile"]
    assert config["data"]["split"] == {
        "scheme": "kfold",
        "folds": 3,
        "fold": 0,
        "seed": 42,
    }
    assert config["model"]["speaker_target_scope"] == "known"
    assert config["model"]["competition_num_known"] == 446
    assert config["model"]["ood_head"] is True
    assert config["training"]["loss"]["proto"]["enabled"] is False

    encoder = config["model"]["encoder_config"]["ecapa"]
    assert config["model"]["encoder_type"] == "ecapa"
    assert encoder["freeze_encoder"] is True
    assert encoder["unfreeze_last_n_blocks"] == 0
    assert encoder["allow_hub_download"] is False
    assert encoder["pooling_type"] == "identity"

    source = config["experiment"]["source_provenance"]
    assert source["revision"] == prereg["intervention"]["encoder_revision"]
    assert config["experiment"]["decision_policy"] == "raw_probability_average_argmax"
    assert config["experiment"]["diagnostic_ensemble"] == (
        "fixed_probability_average_50_50_with_campp_control_fold0"
    )
    assert config["training"]["early_stopping_patience"] == 15
    assert config["training"]["epochs"] == 90


def test_all_promotion_gate_conditions_are_locked() -> None:
    config = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    assert config["experiment"]["preregistered_gate"] == {
        "standalone_min_raw_macro_f1": 0.9269211906147802,
        "fixed_50_50_min_macro_gain": 0.002,
        "max_known_accuracy_drop": 0.001,
        "max_ood_f1_drop": 0.001,
        "min_campp_error_rescue_rate": 0.25,
        "require_rescued_gt_introduced": True,
    }
    assert prereg["gate"]["all_conditions_required"] is True
    assert prereg["data_contract"]["no_leaderboard_selection"] is True
