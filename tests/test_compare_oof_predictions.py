import numpy as np
import pytest

from scripts.compare_oof_predictions import compare_oof


def _write_oof(path, files, labels, probabilities, *, fold=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        files=np.asarray(files, dtype=str),
        labels=np.asarray(labels, dtype=np.int64),
        competition_probs=np.asarray(probabilities, dtype=np.float32),
        split_scheme=np.asarray(["kfold"]),
        split_fold=np.asarray([fold], dtype=np.int64),
        split_folds=np.asarray([3], dtype=np.int64),
        split_seed=np.asarray([42], dtype=np.int64),
    )


def test_identical_predictions_have_no_complementarity(tmp_path):
    files = ["a.wav", "b.wav", "c.wav"]
    labels = [0, 1, 2]
    probs = [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]
    candidate = tmp_path / "candidate" / "oof_predictions.npz"
    baseline = tmp_path / "baseline" / "oof_predictions.npz"
    _write_oof(candidate, files, labels, probs)
    _write_oof(baseline, files, labels, probs)

    report = compare_oof(candidate, baseline)

    complementarity = report["complementarity"]
    assert complementarity["prediction_disagreement_rate"] == 0
    assert complementarity["candidate_only_correct"] == 0
    assert complementarity["baseline_only_correct"] == 0
    assert complementarity["oracle"]["macro_f1_gain_over_best_standalone"] == 0
    assert all(
        row["macro_f1"] == pytest.approx(1.0)
        for row in report["fixed_blends_descriptive_only"]
    )


def test_complementary_errors_produce_positive_oracle_gain(tmp_path):
    files = ["a.wav", "b.wav", "c.wav", "d.wav"]
    labels = [0, 1, 2, 1]
    candidate_probs = [
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.8, 0.1],
    ]
    baseline_probs = [
        [0.1, 0.8, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
        [0.1, 0.1, 0.8],
    ]
    candidate = tmp_path / "candidate" / "oof_predictions.npz"
    baseline = tmp_path / "baseline" / "oof_predictions.npz"
    _write_oof(candidate, files, labels, candidate_probs)
    _write_oof(baseline, files, labels, baseline_probs)

    report = compare_oof(candidate, baseline)

    complementarity = report["complementarity"]
    assert complementarity["candidate_only_correct"] == 2
    assert complementarity["baseline_only_correct"] == 1
    assert complementarity["both_wrong"] == 0
    assert complementarity["oracle"]["overall_acc"] == 1.0
    assert complementarity["oracle"]["macro_f1_gain_over_best_standalone"] > 0


def test_filename_order_is_aligned_before_label_comparison(tmp_path):
    candidate = tmp_path / "candidate" / "oof_predictions.npz"
    baseline = tmp_path / "baseline" / "oof_predictions.npz"
    _write_oof(
        candidate,
        ["a.wav", "b.wav"],
        [0, 1],
        [[0.9, 0.1], [0.1, 0.9]],
    )
    _write_oof(
        baseline,
        ["b.wav", "a.wav"],
        [1, 0],
        [[0.1, 0.9], [0.9, 0.1]],
    )

    report = compare_oof(candidate, baseline)

    assert report["integrity"]["filename_order_aligned"] is True
    assert report["standalone"]["candidate"]["macro_f1"] == pytest.approx(1.0)
    assert report["standalone"]["baseline"]["macro_f1"] == pytest.approx(1.0)


def test_mismatched_file_sets_are_rejected(tmp_path):
    candidate = tmp_path / "candidate" / "oof_predictions.npz"
    baseline = tmp_path / "baseline" / "oof_predictions.npz"
    _write_oof(candidate, ["a.wav"], [0], [[0.9, 0.1]])
    _write_oof(baseline, ["b.wav"], [0], [[0.9, 0.1]])

    with pytest.raises(ValueError, match="file sets differ"):
        compare_oof(candidate, baseline)


def test_mismatched_split_is_rejected(tmp_path):
    candidate = tmp_path / "candidate" / "oof_predictions.npz"
    baseline = tmp_path / "baseline" / "oof_predictions.npz"
    _write_oof(candidate, ["a.wav"], [0], [[0.9, 0.1]], fold=0)
    _write_oof(baseline, ["a.wav"], [0], [[0.9, 0.1]], fold=1)

    with pytest.raises(ValueError, match="split_fold"):
        compare_oof(candidate, baseline)
