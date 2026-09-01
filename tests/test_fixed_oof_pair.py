from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import scripts.dump_checkpoint_oof_single_fold as dump
import scripts.evaluate_fixed_oof_pair as paired


ROOT = Path(__file__).resolve().parents[1]


def test_paired_prereg_locks_equal_fusion_and_quarantines_60_40() -> None:
    contract = json.loads(
        (
            ROOT
            / "configs"
            / "analyses"
            / "no-proto-metric-only-paired-f0-prereg.json"
        ).read_text(encoding="utf-8")
    )
    assert contract["locked_primary_fusion"]["weights"] == [0.5, 0.5]
    assert contract["locked_primary_fusion"]["search_dimensions"] == 0
    assert contract["historical_diagnostic_only"][
        "selection_contaminated_for_fold0_gate"
    ] is True


def test_dump_contract_collapses_pseudo_labels_and_checks_probabilities() -> None:
    np.testing.assert_array_equal(
        dump.collapse_labels(np.array([0, 1, 446, 447, 1000])),
        np.array([0, 1, 446, 0, 0]),
    )
    good = np.full((2, 447), 1.0 / 447.0, dtype=np.float64)
    dump.validate_probability_matrix(good, 2)
    with pytest.raises(RuntimeError, match="shape"):
        dump.validate_probability_matrix(good[:, :-1], 2)
    bad = good.copy()
    bad[0, 0] = np.nan
    with pytest.raises(RuntimeError, match="NaN/Inf"):
        dump.validate_probability_matrix(bad, 2)


def test_dump_file_ids_are_pickle_free(tmp_path: Path) -> None:
    files = dump.pickle_free_string_array(["a.wav", Path("b.wav")])
    assert files.dtype.kind in {"U", "S"}
    output = tmp_path / "oof.npz"
    np.savez_compressed(output, files=files)
    with np.load(output, allow_pickle=False) as archive:
        assert archive["files"].tolist() == ["a.wav", "b.wav"]


def _record(files: list[str], labels: np.ndarray, predictions: np.ndarray) -> dict:
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


def test_pair_aligns_by_filename_and_uses_only_locked_weights(monkeypatch) -> None:
    labels = np.array([1, 2, 0, 3], dtype=np.int64)
    primary = _record(["a", "b", "c", "d"], labels, np.array([1, 0, 1, 3]))
    secondary_order = np.array([2, 0, 3, 1])
    secondary = _record(
        ["c", "a", "d", "b"],
        labels[secondary_order],
        np.array([0, 1, 0, 2]),
    )

    # Keep this unit test independent of the 447-class macro averaging detail.
    def fake_metrics(y_true, y_pred):
        accuracy = float(np.mean(y_true == y_pred))
        return {
            "macro_f1": accuracy,
            "accuracy": accuracy,
            "known_accuracy": accuracy,
            "ood_f1": accuracy,
        }

    monkeypatch.setattr(paired, "metric_bundle", fake_metrics)
    result = paired.evaluate_pair(primary, secondary)

    assert paired.PRIMARY_WEIGHTS == (0.5, 0.5)
    assert paired.HISTORICAL_DIAGNOSTIC_WEIGHTS == (0.6, 0.4)
    assert result["rows"] == 4
    assert "historical_60_40_diagnostic_only" in result["metrics"]


def test_pair_rejects_split_or_file_mismatch() -> None:
    left = _record(["a"], np.array([1]), np.array([1]))
    right = _record(["b"], np.array([1]), np.array([1]))
    with pytest.raises(RuntimeError, match="file sets differ"):
        paired.evaluate_pair(left, right)

    right = _record(["a"], np.array([1]), np.array([1]))
    right["split_seed"] = np.asarray([7])
    with pytest.raises(RuntimeError, match="split_seed mismatch"):
        paired.evaluate_pair(left, right)
