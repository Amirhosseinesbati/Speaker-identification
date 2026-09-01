from pathlib import Path

import numpy as np

from scripts.evaluate_fixed_duration_oof import load_reference, metric_bundle


def test_reference_oof_contract_and_direct_argmax_metrics(tmp_path: Path) -> None:
    reference = tmp_path / "oof.npz"
    np.savez_compressed(
        reference,
        files=np.array(["a.wav", "b.wav", "c.wav"]),
        labels=np.array([1, 2, 0], dtype=np.int64),
        split_scheme=np.array("kfold"),
        split_folds=np.array(3),
        split_fold=np.array(0),
        split_seed=np.array(42),
    )
    frame, split = load_reference(reference)
    assert frame["audio_file"].tolist() == ["a.wav", "b.wav", "c.wav"]
    assert split == {"scheme": "kfold", "folds": 3, "fold": 0, "seed": 42}

    probs = np.array(
        [
            [0.05, 0.90, 0.05],  # known speaker 1 correct
            [0.60, 0.10, 0.30],  # known speaker 2 -> unknown
            [0.20, 0.70, 0.10],  # unknown -> known
        ],
        dtype=np.float32,
    )
    metrics = metric_bundle(frame["label"].to_numpy(), probs, known_count=2)
    assert metrics["known_to_unknown"] == 1
    assert metrics["known_to_wrong_known"] == 0
    assert metrics["unknown_to_known"] == 1


def test_reference_oof_rejects_duplicate_files(tmp_path: Path) -> None:
    reference = tmp_path / "dup.npz"
    np.savez_compressed(
        reference,
        files=np.array(["a.wav", "a.wav"]),
        labels=np.array([1, 1], dtype=np.int64),
        split_scheme=np.array("kfold"),
        split_folds=np.array(3),
        split_fold=np.array(0),
        split_seed=np.array(42),
    )
    try:
        load_reference(reference)
    except ValueError as exc:
        assert "duplicate" in str(exc).lower()
    else:
        raise AssertionError("duplicate reference filenames must be rejected")
