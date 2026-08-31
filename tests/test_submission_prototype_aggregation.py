from __future__ import annotations

import numpy as np
import pytest
from scipy.special import logsumexp, softmax

from submission.inference import load_prototypes, prototype_logmeanexp_probs


def _unit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def test_logmeanexp_matches_bruteforce_and_collapses_unknown() -> None:
    test = _unit(np.array([[1.0, 0.2], [0.1, 1.0]]))
    enrollment = _unit(np.array([
        [1.0, 0.0], [0.8, 0.2],
        [0.0, 1.0],
        [-1.0, 0.0], [-0.8, 0.2],
    ]))
    speaker_ids = np.array([1, 1, 2, 3, 3], dtype=np.int64)
    beta, kappa = 20.0, 16.0

    actual, max_score = prototype_logmeanexp_probs(
        test, enrollment, speaker_ids, 3, beta=beta, kappa=kappa
    )

    similarities = test @ enrollment.T
    scores = np.stack([
        (logsumexp(beta * similarities[:, speaker_ids == speaker_id], axis=1)
         - np.log(np.sum(speaker_ids == speaker_id))) / beta
        for speaker_id in (1, 2, 3)
    ], axis=1)
    internal = np.zeros((len(test), 4), dtype=np.float64)
    internal[:, 1:] = softmax(kappa * scores, axis=1)
    expected = np.column_stack([
        internal[:, 0] + internal[:, 3], internal[:, 1], internal[:, 2]
    ])
    expected /= expected.sum(axis=1, keepdims=True)

    # The runtime promotes group reductions to float64 after the shared
    # float32 cosine matrix; SciPy may promote one operation slightly earlier.
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-6)
    np.testing.assert_allclose(max_score, scores.max(axis=1), atol=1e-7)


def test_logmeanexp_normalises_unequal_group_sizes() -> None:
    test = np.array([[1.0, 0.0]], dtype=np.float32)
    enrollment = np.array([
        [1.0, 0.0],
        [1.0, 0.0], [1.0, 0.0], [1.0, 0.0],
    ], dtype=np.float32)
    speaker_ids = np.array([1, 2, 2, 2], dtype=np.int64)
    probabilities, _ = prototype_logmeanexp_probs(
        test, enrollment, speaker_ids, 3, beta=20.0, kappa=16.0
    )
    np.testing.assert_allclose(probabilities[0, 1], probabilities[0, 2], atol=1e-12)


def test_load_prototypes_validates_dense_ids_and_unit_norm(tmp_path) -> None:
    directory = tmp_path / "prototypes"
    directory.mkdir()
    embeddings = np.eye(3, dtype=np.float32)
    np.savez_compressed(
        directory / "prototypes_campp.npz",
        embeddings=embeddings,
        speaker_ids=np.array([1, 2, 3], dtype=np.int64),
    )
    loaded = load_prototypes(str(directory), ["campp"], expected_groups=3)
    np.testing.assert_array_equal(loaded["campp"][0], embeddings)

    np.savez_compressed(
        directory / "prototypes_campp.npz",
        embeddings=embeddings,
        speaker_ids=np.array([1, 2, 4], dtype=np.int64),
    )
    with pytest.raises(RuntimeError, match="not dense"):
        load_prototypes(str(directory), ["campp"], expected_groups=3)


def test_load_prototypes_keeps_same_encoder_fold_spaces_separate(tmp_path) -> None:
    directory = tmp_path / "prototypes"
    directory.mkdir()
    speaker_ids = np.array([1, 2, 3], dtype=np.int64)
    fold0 = np.eye(3, dtype=np.float32)
    fold1 = np.roll(fold0, 1, axis=1).copy()
    np.savez_compressed(
        directory / "prototypes_campp_lme20_f0.npz",
        embeddings=fold0,
        speaker_ids=speaker_ids,
    )
    np.savez_compressed(
        directory / "prototypes_campp_lme20_f1.npz",
        embeddings=fold1,
        speaker_ids=speaker_ids,
    )
    loaded = load_prototypes(
        str(directory), ["campp_lme20_f0", "campp_lme20_f1"], expected_groups=3
    )
    np.testing.assert_array_equal(loaded["campp_lme20_f0"][0], fold0)
    np.testing.assert_array_equal(loaded["campp_lme20_f1"][0], fold1)
