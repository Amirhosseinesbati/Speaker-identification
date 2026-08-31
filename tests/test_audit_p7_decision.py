from __future__ import annotations

from copy import deepcopy

from scripts.audit_p7_decision import acceptance_gate


def _passing_evidence() -> dict:
    return {
        "matched_control": {
            "macro_f1": 0.9460,
            "known_accuracy": 0.9570,
            "ood_f1": 0.9580,
        },
        "immutable_external_control": {
            "macro_f1": 0.9469211906147802,
            "known_accuracy": 0.9573991031390134,
            "ood_f1": 0.9586206896551724,
        },
        "treatment": {
            "macro_f1": 0.9490,
            "known_accuracy": 0.9565,
            "ood_f1": 0.9582,
        },
        "fixed_fusion": {
            "macro_f1": 0.9491,
            "known_accuracy": 0.9570,
            "ood_f1": 0.9580,
        },
        "supporting_epochs": [31, 34],
        "immutable_external_errors": {
            "known_to_unknown": 24,
            "unknown_to_known": 11,
        },
        "treatment_errors": {
            "known_to_unknown": 19,
            "unknown_to_known": 10,
        },
        "clean_aug_ood_gap_ratio": 0.79,
        "speaker_logits_byte_identical": True,
        "fusion_weight": 0.5,
        "evaluated_fusion_weights": [0.5],
        "provenance_complete": True,
        "decision_policy": "raw_probability_average_lme20_direct_argmax",
        "leaderboard_tuning": False,
    }


def test_all_locked_gates_authorize_replication_only() -> None:
    report = acceptance_gate(_passing_evidence())
    assert report["passed"] is True
    assert report["decision"] == "accept_for_replication"
    assert all(report["checks"].values())


def test_one_isolated_peak_cannot_pass() -> None:
    evidence = _passing_evidence()
    evidence["supporting_epochs"] = [31, 31]
    report = acceptance_gate(evidence)
    assert report["passed"] is False
    assert report["checks"]["gain_supported_by_two_epochs"] is False


def test_each_control_and_each_guardrail_is_binding() -> None:
    evidence = _passing_evidence()
    evidence["matched_control"]["macro_f1"] = 0.9480
    evidence["treatment"]["known_accuracy"] = 0.9559
    evidence["treatment"]["ood_f1"] = 0.9569
    report = acceptance_gate(evidence)
    assert report["passed"] is False
    assert report["checks"]["treatment_gain_vs_matched_control"] is False
    assert report["checks"][
        "treatment_known_guardrail_vs_both_controls"
    ] is False
    assert report["checks"][
        "treatment_ood_guardrail_vs_both_controls"
    ] is False


def test_error_topology_and_mechanism_gates_are_binding() -> None:
    evidence = _passing_evidence()
    evidence["treatment_errors"]["known_to_unknown"] = 20
    evidence["treatment_errors"]["unknown_to_known"] = 12
    evidence["clean_aug_ood_gap_ratio"] = 0.81
    evidence["speaker_logits_byte_identical"] = False
    report = acceptance_gate(evidence)
    assert report["passed"] is False
    assert report["checks"]["known_to_unknown_reduced_by_twenty_percent"] is False
    assert report["checks"]["unknown_to_known_not_increased"] is False
    assert report["checks"]["clean_aug_gap_ratio"] is False
    assert report["checks"]["speaker_logits_byte_identical"] is False


def test_no_fusion_weight_search_or_leaderboard_tuning_can_pass() -> None:
    evidence = _passing_evidence()
    evidence["evaluated_fusion_weights"] = [0.25, 0.5, 0.75]
    evidence["leaderboard_tuning"] = True
    report = acceptance_gate(evidence)
    assert report["passed"] is False
    assert report["checks"]["fixed_fusion_contract"] is False
    assert report["checks"]["leaderboard_tuning_absent"] is False


def test_missing_or_nonfinite_evidence_fails_closed() -> None:
    for mutation in ("nan", "missing"):
        evidence = deepcopy(_passing_evidence())
        if mutation == "nan":
            evidence["treatment"]["macro_f1"] = float("nan")
        else:
            del evidence["fixed_fusion"]["ood_f1"]
        report = acceptance_gate(evidence)
        assert report["passed"] is False
        assert report["checks"]["finite_metric_inputs"] is False


def test_missing_error_count_and_boolean_metric_fail_closed() -> None:
    evidence = _passing_evidence()
    del evidence["treatment_errors"]["known_to_unknown"]
    evidence["treatment"]["known_accuracy"] = True
    report = acceptance_gate(evidence)
    assert report["passed"] is False
    assert report["checks"]["finite_metric_inputs"] is False
    assert report["checks"]["known_to_unknown_reduced_by_twenty_percent"] is False
