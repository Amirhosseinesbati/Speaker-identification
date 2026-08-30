from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.analyze_lme20_speaker_specific_threshold import (
    apply_speaker_specific_rejection,
    evaluate_against_reference,
    locked_lme20_predictions,
    speaker_specific_thresholds,
)
from scripts.analyze_lme20_asnorm_crossfit import (
    LOCKED_RAW_KAPPA,
    decision_predictions,
)


def _unit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def test_thresholds_are_maximum_cross_speaker_enrollment_cosines() -> None:
    embeddings = _unit(
        np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    )
    thresholds = speaker_specific_thresholds(
        embeddings, [np.array([0, 1]), np.array([2, 3])]
    )
    expected = float(embeddings[1] @ embeddings[3])
    np.testing.assert_allclose(thresholds, [expected, expected], atol=1e-7)


def test_rejection_uses_predicted_speaker_and_strict_boundary() -> None:
    predictions = np.array([1, 2, 0, 1], dtype=np.int64)
    scores = np.array(
        [[0.8, 0.2], [0.3, 0.4], [0.1, 0.1], [0.5, 0.2]], dtype=np.float64
    )
    output, rejected = apply_speaker_specific_rejection(
        predictions, scores, np.array([0.5, 0.5])
    )
    assert output.tolist() == [1, 0, 0, 0]
    assert rejected.tolist() == [False, True, False, True]


def test_threshold_builder_rejects_overlapping_groups() -> None:
    with pytest.raises(ValueError, match="overlap"):
        speaker_specific_thresholds(
            np.eye(3, dtype=np.float32), [np.array([0, 1]), np.array([1, 2])]
        )


def test_evaluation_uses_explicit_lme_reference_not_raw_head() -> None:
    folds = [
        SimpleNamespace(
            files=np.array(["a", "b"]),
            labels=np.array([1, 0], dtype=np.int64),
        )
    ]
    reference = [np.array([1, 0], dtype=np.int64)]
    candidate = [np.array([0, 0], dtype=np.int64)]
    result = evaluate_against_reference(folds, reference, candidate)
    assert result["aggregate"]["baseline"]["accuracy"] == 1.0
    assert result["aggregate"]["candidate"]["accuracy"] == 0.5
    assert result["aggregate"]["delta"]["accuracy"] == -0.5
    assert result["aggregate"]["introduced_errors"] == 1


def test_locked_predictions_use_the_validated_lme20_policy() -> None:
    rng = np.random.default_rng(42)
    head = rng.random((4, 447))
    head /= head.sum(axis=1, keepdims=True)
    scores = rng.normal(size=(4, 1000))
    expected = decision_predictions(
        head=head,
        scores=scores,
        probability_kappa=LOCKED_RAW_KAPPA,
        raw_max_scores=scores.max(axis=1),
    )
    np.testing.assert_array_equal(
        locked_lme20_predictions(head, scores), expected
    )
