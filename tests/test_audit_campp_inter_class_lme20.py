from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.audit_campp_inter_class_lme20 import (
    P5_CROSS_FILE_TREATMENT_PROFILE,
    P6_CONTROL_PROFILE,
    P6_TREATMENT_PROFILE,
    acceptance_gate,
    assert_p6_single_objective_contract,
)
from src.experiment_config import load_profile


def _metrics(macro: float, *, known: float = 0.96, ood: float = 0.97) -> dict:
    return {
        "macro_f1": macro,
        "known_accuracy": known,
        "ood_f1": ood,
    }


def _passing_gate() -> dict:
    return acceptance_gate(
        p6_control_metrics=_metrics(0.950),
        p6_treatment_metrics=_metrics(0.953),
        p5_control_metrics=_metrics(0.949),
        p5_treatment_metrics=_metrics(0.951),
        external_metrics=_metrics(0.960),
        fusion_metrics=_metrics(0.963),
        rescue_rate=0.20,
        spread_ratio=0.95,
        energy_ratio=0.95,
    )


def test_p6_contract_is_source_matched_and_single_objective() -> None:
    p5_treatment = load_profile(P5_CROSS_FILE_TREATMENT_PROFILE)
    p6_control = load_profile(P6_CONTROL_PROFILE)
    p6_treatment = load_profile(P6_TREATMENT_PROFILE)
    assert_p6_single_objective_contract(
        p5_treatment, p6_control, p6_treatment
    )

    drifted = deepcopy(p6_treatment)
    drifted["training"]["encoder_lr"] *= 2
    with pytest.raises(RuntimeError, match="outside inter_class.enabled"):
        assert_p6_single_objective_contract(
            p5_treatment, p6_control, drifted
        )


def test_p6_acceptance_gate_passes_exact_locked_boundaries() -> None:
    gate = _passing_gate()
    assert gate["passed"] is True
    assert all(gate["checks"].values())


@pytest.mark.parametrize(
    ("field", "value", "failed_check"),
    [
        ("rescue_rate", 0.199, "minimum_rescue_rate"),
        ("spread_ratio", 0.949, "embedding_spread_ratio"),
        ("energy_ratio", 0.951, "exclusive_energy_ratio"),
    ],
)
def test_p6_acceptance_gate_rejects_mechanism_failures(
    field: str, value: float, failed_check: str
) -> None:
    kwargs = {
        "p6_control_metrics": _metrics(0.950),
        "p6_treatment_metrics": _metrics(0.953),
        "p5_control_metrics": _metrics(0.949),
        "p5_treatment_metrics": _metrics(0.951),
        "external_metrics": _metrics(0.960),
        "fusion_metrics": _metrics(0.963),
        "rescue_rate": 0.20,
        "spread_ratio": 0.95,
        "energy_ratio": 0.95,
    }
    kwargs[field] = value
    gate = acceptance_gate(**kwargs)
    assert gate["passed"] is False
    assert gate["checks"][failed_check] is False


def test_p6_acceptance_gate_rejects_score_or_guardrail_failures() -> None:
    gate = acceptance_gate(
        p6_control_metrics=_metrics(0.950),
        p6_treatment_metrics=_metrics(0.9519, known=0.9589),
        p5_control_metrics=_metrics(0.949),
        p5_treatment_metrics=_metrics(0.9520),
        external_metrics=_metrics(0.960),
        fusion_metrics=_metrics(0.9619, known=0.9589),
        rescue_rate=0.30,
        spread_ratio=1.0,
        energy_ratio=0.90,
    )
    assert gate["passed"] is False
    assert gate["checks"]["source_matched_macro_gain"] is False
    assert gate["checks"]["historical_p5_treatment_noninferiority"] is False
    assert gate["checks"]["fixed_fusion_macro_gain"] is False
    assert gate["checks"]["p6_control_known_guardrail"] is False
    assert gate["checks"]["fixed_fusion_known_guardrail"] is False
