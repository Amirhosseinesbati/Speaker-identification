from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

import scripts.analyze_oof_actual_duration as duration_audit


def _record(files: list[str], labels: np.ndarray) -> dict[str, np.ndarray]:
    probabilities = np.full((len(files), 447), 1e-8, dtype=np.float64)
    probabilities[np.arange(len(files)), labels] = 1.0
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return {
        "files": np.asarray(files),
        "labels": labels,
        "competition_probs": probabilities,
        "split_fold": np.asarray([0]),
        "split_folds": np.asarray([3]),
        "split_seed": np.asarray([42]),
    }


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0.1, "le_2s"),
        (2.0, "le_2s"),
        (2.01, "gt_2_le_4s"),
        (4.0, "gt_2_le_4s"),
        (6.0, "gt_4_le_6s"),
        (8.0, "gt_6_le_8s"),
        (12.0, "gt_8_le_12s"),
        (12.01, "gt_12s"),
    ],
)
def test_duration_bins_are_fixed_and_non_overlapping(
    seconds: float, expected: str
) -> None:
    assert duration_audit.duration_bin_name(seconds) == expected


def test_wav_duration_uses_header_frames(tmp_path: Path) -> None:
    path = tmp_path / "two_seconds.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 32000)
    assert duration_audit.wav_duration_seconds(path) == 2.0


def test_pair_alignment_is_by_filename_and_checks_split() -> None:
    labels = np.array([1, 0], dtype=np.int64)
    baseline = _record(["a.wav", "b.wav"], labels)
    candidate = _record(["b.wav", "a.wav"], labels[::-1])
    files, aligned_labels, _, _, split = duration_audit.align_pair(
        baseline, candidate
    )
    assert files.tolist() == ["a.wav", "b.wav"]
    assert aligned_labels.tolist() == [1, 0]
    assert split == {"split_fold": 0, "split_folds": 3, "split_seed": 42}

    candidate["split_seed"] = np.asarray([7])
    with pytest.raises(RuntimeError, match="split_seed mismatch"):
        duration_audit.align_pair(baseline, candidate)
