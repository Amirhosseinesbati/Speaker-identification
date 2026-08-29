import numpy as np

from scripts.audit_raw_ema_lme20 import (
    final_decision,
    fixed_raw_ema_decision,
)


def _evidence():
    head = np.array([
        [0.20, 0.70, 0.10],
        [0.70, 0.20, 0.10],
    ])
    prototype = np.array([
        [0.10, 0.80, 0.10],
        [0.60, 0.25, 0.15],
    ])
    max_score = np.array([0.70, 0.30])
    return head, prototype, max_score


def test_fixed_self_ensemble_is_exactly_decision_equivalent():
    evidence = _evidence()
    baseline_probs, baseline_predictions = final_decision(*evidence)
    ensemble_probs, ensemble_predictions = fixed_raw_ema_decision(
        evidence, evidence
    )

    np.testing.assert_allclose(ensemble_probs, baseline_probs, atol=1e-12)
    np.testing.assert_array_equal(ensemble_predictions, baseline_predictions)


def test_fixed_ensemble_averages_all_three_evidence_channels():
    raw = _evidence()
    ema = (
        np.flip(raw[0], axis=1).copy(),
        np.flip(raw[1], axis=1).copy(),
        np.array([0.60, 0.80]),
    )
    actual_probs, actual_predictions = fixed_raw_ema_decision(raw, ema)
    expected_probs, expected_predictions = final_decision(
        0.5 * raw[0] + 0.5 * ema[0],
        0.5 * raw[1] + 0.5 * ema[1],
        0.5 * raw[2] + 0.5 * ema[2],
    )

    np.testing.assert_allclose(actual_probs, expected_probs, atol=1e-12)
    np.testing.assert_array_equal(actual_predictions, expected_predictions)


def test_fixed_ensemble_rejects_misaligned_snapshot_evidence():
    raw = _evidence()
    malformed = (raw[0][:-1], raw[1], raw[2])

    try:
        fixed_raw_ema_decision(raw, malformed)
    except RuntimeError as exc:
        assert "shape mismatch" in str(exc)
    else:
        raise AssertionError("misaligned evidence must be rejected")
