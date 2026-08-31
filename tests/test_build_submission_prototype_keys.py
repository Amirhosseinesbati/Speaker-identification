import json
from pathlib import Path

from scripts.build_submission import ROOT, _fusion_model_specs


def test_explicit_models_default_prototype_key_to_packaged_checkpoint(tmp_path) -> None:
    fusion = {
        "models": [
            {
                "name": "fold-zero",
                "source_checkpoint": "source.pt",
                "checkpoint": "campp_lme20_f0_best.pt",
                "base_encoder": "campp",
                "weight": 1.0,
            }
        ]
    }
    specs = _fusion_model_specs(fusion, tmp_path)
    assert specs[0]["prototype_key"] == "campp_lme20_f0"


def test_threefold_lme_config_has_unique_model_specific_prototypes() -> None:
    path = ROOT / "configs" / "submissions" / "campp-lme20-threefold-equal.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    keys = [model["prototype_key"] for model in config["models"]]
    assert keys == ["campp_lme20_f0", "campp_lme20_f1", "campp_lme20_f2"]
    assert set(keys) == set(config["prototype_sources"])
    assert set(keys) == set(config["evidence"]["prototype_sha256"])
    assert abs(sum(config["weights"]) - 1.0) < 1e-12
