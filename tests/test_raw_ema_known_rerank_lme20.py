import numpy as np

from scripts.audit_raw_ema_known_rerank_lme20 import (
    binary_locked_known_rerank,
    gate,
)


def test_known_rerank_preserves_binary_decisions_exactly():
    raw = np.array([0, 1, 2, 0, 4])
    probabilities = np.zeros((5, 447), dtype=np.float64)
    probabilities[0, 8] = 1.0
    probabilities[1, 3] = 1.0
    probabilities[2, 2] = 1.0
    probabilities[3, 5] = 1.0
    probabilities[4, 9] = 1.0

    candidate = binary_locked_known_rerank(raw, probabilities)

    np.testing.assert_array_equal(candidate, np.array([0, 3, 2, 0, 9]))
    np.testing.assert_array_equal(candidate == 0, raw == 0)


def _row(macro, ood=0.0, binary=0):
    return {
        "delta": {"macro_f1": macro, "ood_f1": ood},
        "binary_changed": binary,
    }


def test_gate_requires_useful_known_gain_and_exact_invariants():
    aggregate = {
        "delta": {
            "macro_f1": 0.0012,
            "known_accuracy": 0.0011,
            "ood_f1": 0.0,
        }
    }
    assert gate([_row(0.0), _row(0.001), _row(0.002)], aggregate)["passed"]

    negative = gate([_row(-1e-9), _row(0.001), _row(0.002)], aggregate)
    assert not negative["passed"]

    changed_binary = gate(
        [_row(0.0), _row(0.001, binary=1), _row(0.002)], aggregate
    )
    assert not changed_binary["passed"]
