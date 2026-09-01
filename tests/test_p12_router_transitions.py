from __future__ import annotations

import numpy as np
import pytest

from scripts.analyze_p12_router_transitions import analyze_transitions


def _record(files, labels, predictions):
    probabilities = np.full((len(files), 447), 1e-6, dtype=np.float64)
    probabilities[np.arange(len(files)), predictions] = 1.0
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return {
        "files": np.asarray(files),
        "labels": np.asarray(labels),
        "competition_probs": probabilities,
    }


def test_transition_forensics_counts_known_and_unknown() -> None:
    files = ["a.wav", "b.wav", "c.wav", "d.wav"]
    labels = np.array([1, 0, 2, 0])
    baseline = _record(files, labels, np.array([1, 3, 2, 0]))
    specialist = _record(files, labels, np.array([4, 0, 2, 5]))
    mask = np.array([True, True, False, False])
    router_probs = baseline["competition_probs"].copy()
    router_probs[mask] = specialist["competition_probs"][mask]
    routed = {
        "files": np.asarray(files),
        "labels": labels,
        "competition_probs": router_probs,
        "duration_seconds": np.array([2.0, 3.0, 12.0, 20.0]),
        "short_specialist_mask": mask,
    }

    report = analyze_transitions(baseline, specialist, routed)

    assert report["integrity"]["exact_hard_route_verified"] is True
    assert report["summary"] == {
        "prediction_changes": 2,
        "rescued_errors": 1,
        "introduced_errors": 1,
        "rescued_known": 0,
        "rescued_unknown": 1,
        "introduced_known": 1,
        "introduced_unknown": 0,
        "net_correct": 0,
    }
    assert report["introduced_errors"][0]["file"] == "a.wav"
    assert report["rescued_errors"][0]["file"] == "b.wav"


def test_transition_forensics_rejects_non_exact_router() -> None:
    files = ["a.wav"]
    labels = np.array([1])
    baseline = _record(files, labels, np.array([1]))
    specialist = _record(files, labels, np.array([2]))
    routed = {
        "files": np.asarray(files),
        "labels": labels,
        "competition_probs": baseline["competition_probs"].copy(),
        "duration_seconds": np.array([1.0]),
        "short_specialist_mask": np.array([True]),
    }

    with pytest.raises(RuntimeError, match="exact hard-routed"):
        analyze_transitions(baseline, specialist, routed)
