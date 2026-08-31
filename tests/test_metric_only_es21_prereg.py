import json
from pathlib import Path

from src.experiment_config import load_profile


ROOT = Path(__file__).resolve().parents[1]
PROFILE = "p0-campp-metric-only-repro-es21-oof-f0"


def test_metric_only_es21_profile_is_one_variable_reproduction():
    config = load_profile(PROFILE)
    assert config["data"]["split"] == {
        "scheme": "kfold",
        "folds": 3,
        "fold": 0,
        "seed": 42,
    }
    assert config["model"]["ood_head"] is False
    assert config["model"]["num_unknown_clusters"] == 554
    assert config["model"]["unknown_cluster_path"].endswith(
        "unknown_clusters_oof_f0.json"
    )
    assert config["training"]["freeze_epochs"] == 20
    assert config["training"]["early_stopping_start_epoch"] == 21
    assert config["training"]["early_stopping_patience"] == 20
    assert config["training"]["epochs"] == 200
    assert config["training"]["loss"]["ood"]["weight"] == 0.0
    assert config["training"]["loss"]["proto"] == {
        "enabled": True,
        "scope": "metric",
        "weight": 0.05,
        "scale": 30.0,
        "margin": 0.2,
        "decay": 0.9,
    }


def test_clean_pair_prereg_quarantines_historical_forensic_result():
    prereg = json.loads(
        (
            ROOT
            / "configs"
            / "analyses"
            / "no-proto-es21-metric-only-es21-paired-f0-prereg.json"
        ).read_text(encoding="utf-8")
    )
    assert prereg["locked_primary_fusion"]["weights"] == [0.5, 0.5]
    assert prereg["locked_primary_fusion"]["search_dimensions"] == 0
    assert prereg["historical_forensic_only"]["selection_or_gate_eligible"] is False
    assert prereg["historical_forensic_only"]["historical_train_in_sample_rows"] == 1285
    assert prereg["expansion_gate"]["equal_fusion_macro_f1_min"] == 0.96
    assert prereg["expansion_gate"]["rescue_rate_vs_better_single_min"] == 0.25
