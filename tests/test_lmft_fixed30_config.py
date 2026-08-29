from copy import deepcopy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "configs/experiments/p2-campp-known446-ood-lmft-m05-oof-f0.yaml"
FIXED30 = ROOT / "configs/experiments/p2-campp-known446-ood-lmft-m05-fixed30-oof-f0.yaml"


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_fixed30_changes_only_horizon_control_and_output_identity() -> None:
    original = deepcopy(_load(ORIGINAL))
    fixed30 = deepcopy(_load(FIXED30))

    assert fixed30["experiment"]["preregistered_baseline"]["epoch"] == 0
    assert fixed30["experiment"]["fixed_horizon"] == {
        "epochs": 30,
        "metric_early_stopping": False,
        "min_supporting_epochs": 2,
    }
    assert original["training"]["early_stopping_patience"] == 8
    assert fixed30["training"]["early_stopping_patience"] == 0

    for config in (original, fixed30):
        config.pop("_meta")
        config.pop("experiment")
        config["logging"]["checkpoint_dir"] = "<profile-checkpoints>"
        config["logging"]["log_dir"] = "<profile-logs>"
        config["training"]["early_stopping_patience"] = "<horizon-control>"

    assert fixed30 == original
