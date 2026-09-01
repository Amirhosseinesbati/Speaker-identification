from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import scripts.evaluate_ecapa_campp_fixed_pair as paired


ROOT = Path(__file__).resolve().parents[1]


def _record(files: list[str], labels: np.ndarray, probabilities: np.ndarray) -> dict:
    return {
        "files": np.asarray(files),
        "labels": labels,
        "competition_probs": probabilities,
        "split_fold": np.asarray([0]),
        "split_folds": np.asarray([3]),
        "split_seed": np.asarray([42]),
    }


def _probabilities(predictions: list[int], confidence: float = 0.9) -> np.ndarray:
    values = np.full((len(predictions), 447), (1.0 - confidence) / 446)
    values[np.arange(len(predictions)), predictions] = confidence
    return values


def _contract(campp_metrics: dict) -> dict:
    return {
        "profile": paired.PROFILE,
        "locked_comparator": {
            "profile": "p0-campp-known446-ood-control-oof-f0",
            "raw_macro_f1": campp_metrics["macro_f1"],
            "known_accuracy": campp_metrics["known_accuracy"],
            "ood_f1": campp_metrics["ood_f1"],
        },
        "gate": {
            "standalone_min_raw_macro_f1": 0.5,
            "fixed_50_50_min_macro_gain": 0.0,
            "max_known_accuracy_drop": 0.1,
            "max_ood_f1_drop": 0.1,
            "min_campp_error_rescue_rate": 0.0,
            "require_rescued_gt_introduced": True,
            "all_conditions_required": True,
        },
    }


def _accuracy_bundle(labels, predictions):
    score = float(np.mean(np.asarray(labels) == np.asarray(predictions)))
    return {
        "macro_f1": score,
        "accuracy": score,
        "known_accuracy": score,
        "ood_f1": score,
    }


def test_repository_prereg_has_no_legacy_absolute_macro_floor() -> None:
    contract = json.loads(
        (
            ROOT
            / "configs"
            / "analyses"
            / "ecapa-frozen-known446-ood-complement-oof-f0.prereg.json"
        ).read_text(encoding="utf-8")
    )
    assert contract["profile"] == paired.PROFILE
    assert "equal_fusion_macro_f1_min" not in contract["gate"]
    assert contract["gate"]["standalone_min_raw_macro_f1"] == pytest.approx(
        0.9269211906147802
    )
    assert paired.FUSION_WEIGHTS == (0.5, 0.5)


def test_pair_uses_campp_as_locked_comparator_and_rescue_reference(monkeypatch) -> None:
    monkeypatch.setattr(paired, "metric_bundle", _accuracy_bundle)
    labels = np.array([1, 2, 3, 0], dtype=np.int64)
    campp_probs = _probabilities([1, 0, 3, 1], confidence=0.55)
    ecapa_probs = _probabilities([1, 2, 3, 0], confidence=0.95)
    campp = _record(["a", "b", "c", "d"], labels, campp_probs)
    ecapa_order = np.array([2, 0, 3, 1])
    ecapa = _record(
        ["c", "a", "d", "b"],
        labels[ecapa_order],
        ecapa_probs[ecapa_order],
    )
    campp_metrics = _accuracy_bundle(labels, campp_probs.argmax(axis=1))
    result = paired.evaluate_preregistered_pair(
        campp, ecapa, _contract(campp_metrics)
    )

    assert result["metrics"]["campp_locked_control"] == campp_metrics
    assert result["fixed_50_50_error_transitions_vs_campp"]["baseline_errors"] == 2
    assert result["fixed_50_50_error_transitions_vs_campp"]["rescued_errors"] == 2
    assert result["fixed_50_50_error_transitions_vs_campp"]["introduced_errors"] == 0
    assert result["gate"]["passed"] is True


def test_pair_rejects_locked_campp_metric_mismatch(monkeypatch) -> None:
    monkeypatch.setattr(paired, "metric_bundle", _accuracy_bundle)
    labels = np.array([1, 2], dtype=np.int64)
    probabilities = _probabilities([1, 0])
    campp = _record(["a", "b"], labels, probabilities)
    ecapa = _record(["a", "b"], labels, probabilities)
    contract = _contract(_accuracy_bundle(labels, probabilities.argmax(axis=1)))
    contract["locked_comparator"]["raw_macro_f1"] += 0.01
    with pytest.raises(RuntimeError, match=r"Locked CAM\+\+ macro_f1 mismatch"):
        paired.evaluate_preregistered_pair(campp, ecapa, contract)
