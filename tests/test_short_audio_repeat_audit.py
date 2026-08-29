import numpy as np

from scripts.audit_short_audio_repeat import (
    acceptance_gate,
    short_subset_summary,
)


def _row(macro=0.001, known=0.0, ood=0.0):
    return {
        "delta": {
            "macro_f1": macro,
            "known_accuracy": known,
            "ood_f1": ood,
        }
    }


def _aggregate(macro=0.001, known=0.0, ood=0.0):
    return {
        "delta": {
            "macro_f1": macro,
            "known_accuracy": known,
            "ood_f1": ood,
        }
    }


def test_acceptance_gate_requires_consistent_fold_direction_and_guardrails():
    passed = acceptance_gate(
        [_row(0.001), _row(0.002), _row(0.003)],
        _aggregate(0.0015, known=-0.0005, ood=-0.0005),
    )
    assert passed["passed"] is True

    inconsistent = acceptance_gate(
        [_row(0.001), _row(-1e-8), _row(0.003)],
        _aggregate(0.002),
    )
    assert inconsistent["passed"] is False
    assert inconsistent["conditions"]["all_three_folds_macro_positive"] is False

    guardrail = acceptance_gate(
        [_row(0.001), _row(0.002, known=-0.0011), _row(0.003)],
        _aggregate(0.002),
    )
    assert guardrail["passed"] is False
    assert guardrail["conditions"]["all_fold_known_guardrails"] is False


def test_short_subset_summary_counts_rescues_and_regressions():
    labels = np.array([1, 2, 0, 3, 4])
    baseline = np.array([0, 2, 1, 3, 4])
    candidate = np.array([1, 0, 0, 3, 2])
    short = np.array([True, True, True, False, False])

    result = short_subset_summary(labels, baseline, candidate, short)

    assert result == {
        "files": 3,
        "baseline_correct": 1,
        "candidate_correct": 2,
        "baseline_accuracy": 1 / 3,
        "candidate_accuracy": 2 / 3,
        "rescued_errors": 2,
        "introduced_errors": 1,
        "changed_predictions": 3,
    }
