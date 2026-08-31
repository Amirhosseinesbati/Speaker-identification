from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

from scripts.audit_campp_inter_class_lme20 import (
    P6_CONTROL_CONFIG_SHA256,
    P6_CONTROL_PROFILE,
    P6_EARLY_STOPPING_PATIENCE,
    P6_EARLY_STOPPING_START_EPOCH,
    P6_MAXIMUM_EPOCHS,
    P6_TREATMENT_CONFIG_SHA256,
    P6_TREATMENT_PROFILE,
)
from src.experiment_config import load_profile
from src.pipelines.steps import _early_stopping_staleness


def _history(values: list[float]) -> list[dict]:
    return [
        {"epoch": epoch, "val_macro_f1": value}
        for epoch, value in enumerate(values, start=1)
    ]


def _normalise_pair(config: dict) -> dict:
    result = deepcopy(config)
    result.pop("experiment", None)
    result["logging"] = {"checkpoint_dir": "<profile>", "log_dir": "<profile>"}
    result["hardware"]["profiles"]["vastai_3090_campp"].pop(
        "description", None
    )
    return result


def test_delayed_staleness_does_not_consume_patience_before_start() -> None:
    values = [0.90] + [0.89] * 78 + [0.88] * 20
    assert _early_stopping_staleness(_history(values[:79]), 80) == 0
    assert _early_stopping_staleness(_history(values), 80) == 20


def test_delayed_staleness_resets_on_late_improvement() -> None:
    values = [0.90] + [0.89] * 84 + [0.91] + [0.90] * 7
    assert _early_stopping_staleness(_history(values), 80) == 7


def test_p6_delayed_pair_is_single_variable_and_hash_locked() -> None:
    control = load_profile(P6_CONTROL_PROFILE)
    treatment = load_profile(P6_TREATMENT_PROFILE)
    for config in (control, treatment):
        training = config["training"]
        assert training["epochs"] == P6_MAXIMUM_EPOCHS == 120
        assert (
            training["early_stopping_start_epoch"]
            == P6_EARLY_STOPPING_START_EPOCH
            == 80
        )
        assert (
            training["early_stopping_patience"]
            == P6_EARLY_STOPPING_PATIENCE
            == 20
        )

    normalised_control = _normalise_pair(control)
    normalised_treatment = _normalise_pair(treatment)
    normalised_treatment["training"]["loss"]["speaker"]["inter_class"][
        "enabled"
    ] = False
    assert normalised_treatment == normalised_control

    root = Path(__file__).resolve().parents[1]
    expected = {
        P6_CONTROL_PROFILE: P6_CONTROL_CONFIG_SHA256,
        P6_TREATMENT_PROFILE: P6_TREATMENT_CONFIG_SHA256,
    }
    for profile, digest in expected.items():
        path = root / "configs" / "experiments" / f"{profile}.yaml"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
