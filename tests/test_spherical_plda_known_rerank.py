from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import scripts.evaluate_spherical_plda_known_rerank as sph


ROOT = Path(__file__).resolve().parents[1]


def test_prereg_locks_method_and_preserves_ood_decision() -> None:
    contract = json.loads(
        (
            ROOT
            / "configs"
            / "analyses"
            / "spherical-plda-known-rerank-prereg.json"
        ).read_text(encoding="utf-8")
    )
    policy = contract["fixed_phase_one_policy"]
    assert policy["em_iterations"] == 10
    assert policy["threshold_search"] is False
    assert policy["fusion_search"] is False
    assert "copy" in policy["ood_decision"]


def test_spherical_plda_fits_positive_variances_and_prefers_own_class() -> None:
    rng = np.random.default_rng(7)
    raw = np.vstack(
        [
            np.array([1.0, 0.0, 0.0]) + 0.03 * rng.normal(size=(5, 3)),
            np.array([0.0, 1.0, 0.0]) + 0.03 * rng.normal(size=(5, 3)),
        ]
    )
    labels = np.repeat([1, 2], 5)
    center = raw.mean(axis=0)
    values = sph.center_and_normalize(raw, center)
    b, w = sph.fit_spherical_plda(values, labels)
    assert b > 0.0
    assert w > 0.0
    centroids = np.vstack([values[labels == 1].mean(0), values[labels == 2].mean(0)])
    scores = sph.spherical_plda_score_matrix(
        centroids,
        np.array([5, 5]),
        values[[0, 5]],
        b,
        w,
    )
    assert scores.shape == (2, 2)
    assert scores[0, 0] > scores[1, 0]
    assert scores[1, 1] > scores[0, 1]


def test_known_rerank_cannot_change_binary_ood_decisions() -> None:
    probabilities = np.full((4, sph.NUM_CLASSES), 1e-9, dtype=np.float64)
    probabilities[0, 0] = 1.0
    probabilities[1, 1] = 1.0
    probabilities[2, 2] = 1.0
    probabilities[3, 0] = 1.0
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    scores = np.zeros((sph.NUM_KNOWN, 4), dtype=np.float64)
    scores[4, 1] = 3.0
    scores[7, 2] = 4.0
    baseline, candidate = sph.rerank_known_only(probabilities, scores)
    np.testing.assert_array_equal(baseline == 0, candidate == 0)
    np.testing.assert_array_equal(candidate, np.array([0, 5, 8, 0]))


def test_collapse_labels_maps_pseudo_unknowns_only() -> None:
    np.testing.assert_array_equal(
        sph.collapse_labels(np.array([0, 1, 446, 447, 1000])),
        np.array([0, 1, 446, 0, 0]),
    )
