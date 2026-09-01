from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREREG = (
    ROOT / "configs" / "analyses" / "p12-campp-duration-router-f0-prereg.json"
)


def test_p12_prereg_locks_structural_router_before_evaluation() -> None:
    contract = json.loads(PREREG.read_text(encoding="utf-8"))
    assert contract["status"] == (
        "preregistered_before_router_metric_evaluation"
    )
    assert contract["locked_router"] == {
        "short_if_duration_seconds_le": 8.0,
        "short_branch": "P11 Raw probability-average vector",
        "long_branch": "P0 Raw probability-average vector",
        "boundary_inclusive": True,
        "blend": False,
        "threshold_search_dimensions": 0,
        "decision": (
            "direct argmax of the selected 447-way probability vector"
        ),
        "ema_allowed": False,
        "logit_average_allowed": False,
        "calibration_allowed": False,
    }
    assert contract["gate"]["all_checks_required"] is True
    assert contract["gate"]["min_raw_macro_f1_gain_vs_p0"] == 0.002
    assert contract["gate"]["max_known_accuracy_drop_vs_p0"] == 0.001
    assert contract["gate"]["max_ood_f1_drop_vs_p0"] == 0.001
    assert contract["gate"]["min_routed_subset_p0_error_rescue_rate"] == 0.25
    assert contract["later_fold_contract_if_authorized"][
        "early_stopping_patience"
    ] == 8
    assert contract["later_fold_contract_if_authorized"][
        "submission_requires_complete_three_fold_oof"
    ] is True


def test_p12_prereg_discloses_prior_bins_and_forbids_same_fold_selection() -> None:
    contract = json.loads(PREREG.read_text(encoding="utf-8"))
    disclosure = contract["disclosed_prior_analysis"]
    assert disclosure["sha256"] == (
        "4ff0d67768c82239aac054fbd6536eb3605cf255492e03022d2e2470fe2767ad"
    )
    assert "not selected" in disclosure["selection_firewall"]
    assert contract["later_fold_contract_if_authorized"][
        "cutoff_tuning"
    ] is False
    assert any(
        "No duration cutoff sweep" in rule
        for rule in contract["prohibitions"]
    )
