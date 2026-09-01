from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.experiment_config import load_profile
from src.pipelines.steps import _resume_contract


ROOT = Path(__file__).resolve().parents[1]
P13B = ROOT / "configs/experiments/p13b-wavlm-baseplus-layerweighted-deterministic-known446-ood-complement-oof-f0.yaml"
P13C = ROOT / "configs/experiments/p13c-wavlm-baseplus-layerweighted-deterministic-continuation-known446-ood-complement-oof-f0.yaml"
PREREG = ROOT / "configs/analyses/p13c-wavlm-baseplus-layerweighted-deterministic-continuation-known446-ood-complement-oof-f0.prereg.json"


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_p13c_changes_only_resume_and_early_stopping_dimensions() -> None:
    # Compare the exact fully-resolved contracts consumed by the pipeline.
    # Comparing the sparse profile YAMLs alone can miss a mismatch in an
    # inherited base-config field.
    source = load_profile(P13B.stem)
    continuation = load_profile(P13C.stem)

    assert _resume_contract(source) == _resume_contract(continuation)
    training = continuation["training"]
    assert training["epochs"] == source["training"]["epochs"] == 40
    assert training["early_stopping_start_epoch"] == 20
    assert training["early_stopping_patience"] == 10
    assert training["resume_checkpoint"].endswith("/latest_model.pt")
    assert training["resume_history_path"] == training["resume_checkpoint"]


def test_p13c_preregisters_a_slope_gate_and_no_fixed_futility() -> None:
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))

    assert prereg["status"].startswith("preregistered_before_p13b_terminal_result")
    assert prereg["leaderboard_used_for_selection"] is False
    activation = prereg["activation_gate"]
    assert activation["minimum_epoch15_raw_macro_f1"] == 0.10
    assert activation["require_raw_strict_increase_epochs"] == [13, 14, 15]
    assert activation["minimum_train_loss_drop_epoch13_to_epoch15"] == 0.50
    assert prereg["selection"]["fixed_metric_futility_threshold"] is None
    assert prereg["selection"]["early_stopping_patience"] == 10
