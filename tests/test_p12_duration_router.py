from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import scripts.evaluate_p12_duration_router as router


ROOT = Path(__file__).resolve().parents[1]


def _record(
    files: list[str], labels: np.ndarray, predictions: np.ndarray
) -> dict[str, np.ndarray]:
    probabilities = np.full((len(files), 447), 1e-8, dtype=np.float64)
    probabilities[np.arange(len(files)), predictions] = 1.0
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return {
        "files": np.asarray(files),
        "labels": labels,
        "competition_probs": probabilities,
        "split_fold": np.asarray([0]),
        "split_folds": np.asarray([3]),
        "split_seed": np.asarray([42]),
    }


def _contract() -> dict:
    path = (
        ROOT
        / "configs"
        / "analyses"
        / "p12-campp-duration-router-f0-prereg.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def test_router_uses_exact_specialist_vector_at_or_below_eight_seconds(
    monkeypatch,
) -> None:
    files = ["a", "b", "c", "d"]
    labels = np.array([1, 2, 0, 3], dtype=np.int64)
    baseline = _record(files, labels, np.array([0, 2, 1, 3]))
    specialist = _record(files, labels, np.array([1, 0, 0, 2]))
    durations = np.array([1.0, 8.0, 8.01, 20.0])

    def simple_metrics(y_true, y_pred):
        accuracy = float(np.mean(y_true == y_pred))
        return {
            "macro_f1": accuracy,
            "accuracy": accuracy,
            "known_accuracy": accuracy,
            "ood_f1": accuracy,
        }

    monkeypatch.setattr(router, "metric_bundle", simple_metrics)
    result, output = router.evaluate_router(
        baseline, specialist, durations, _contract()
    )
    predictions = output["competition_probs"].argmax(axis=1)
    assert predictions.tolist() == [1, 0, 1, 3]
    assert output["short_specialist_mask"].tolist() == [True, True, False, False]
    np.testing.assert_allclose(
        output["competition_probs"][0],
        specialist["competition_probs"][0],
        rtol=0.0,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        output["competition_probs"][2],
        baseline["competition_probs"][2],
        rtol=0.0,
        atol=1e-7,
    )
    assert result["contract"]["blend"] is False
    assert result["routing"]["short_specialist_rows"] == 2


def test_router_rejects_invalid_probability_matrix() -> None:
    good = np.full((2, 447), 1.0 / 447.0)
    router.validate_probabilities(good)
    bad = good.copy()
    bad[0, 0] = np.nan
    try:
        router.validate_probabilities(bad)
    except RuntimeError as exc:
        assert "NaN/Inf" in str(exc)
    else:
        raise AssertionError("NaN probabilities must be rejected")
