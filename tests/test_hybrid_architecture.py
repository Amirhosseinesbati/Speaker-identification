"""CPU-only tests for the hybrid pseudo-identity + OOD architecture."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio_windows import fit_short_audio, make_eval_windows
from src.data_pipeline import (
    apply_unknown_cluster_labels,
    create_class_mapping,
    ensure_target_columns,
    speaker_aware_kfold,
)
from src.train import TwoPartLoss, compute_ood_accuracy


def test_pseudo_identity_preserves_original_ood_target():
    frame = pd.DataFrame({
        "speaker_id": ["known-a", "unknown", "unknown"],
        "audio_file": ["a.wav", "u1.wav", "u2.wav"],
    })
    rewritten, stats = apply_unknown_cluster_labels(frame, {"u1.wav": 7})
    assert stats["n_rewritten"] == 1
    assert rewritten.loc[1, "speaker_id"] == "unknown_0007"
    assert rewritten.loc[1, "original_speaker_id"] == "unknown"
    assert rewritten.loc[1, "is_ood"] == 1
    assert rewritten.loc[0, "is_ood"] == 0
    mapping = create_class_mapping(rewritten)
    assert mapping["unknown"] == 0


def test_hybrid_bce_treats_pseudo_classes_as_ood():
    criterion = TwoPartLoss(
        use_ood=True, use_focal=False, competition_known_count=446,
        ood_weight=1.0, speaker_weight=0.0,
    )
    # known id=1 should be negative; pseudo-OOD id=447 should be positive.
    labels = torch.tensor([1, 447])
    good = torch.tensor([[-8.0], [8.0]])
    bad = -good
    speaker_logits = torch.zeros(2, 447)
    _, good_parts = criterion(good, speaker_logits, labels)
    _, bad_parts = criterion(bad, speaker_logits, labels)
    assert good_parts["loss_ood"] < 0.01
    assert bad_parts["loss_ood"] > 5.0
    assert compute_ood_accuracy(good, labels, 446) == 1.0


def test_pseudo_labels_do_not_change_kfold_membership():
    base = pd.DataFrame({
        "speaker_id": ["known-a"] * 3 + ["unknown"] * 6,
        "audio_file": [f"k{i}.wav" for i in range(3)] + [f"u{i}.wav" for i in range(6)],
    })
    base = ensure_target_columns(base)
    base["label"] = [1] * 3 + [0] * 6
    pseudo, _ = apply_unknown_cluster_labels(
        base, {f"u{i}.wav": i for i in range(6)})
    pseudo["label"] = [1] * 3 + list(range(447, 453))
    plain_splits = speaker_aware_kfold(base, folds=3, random_seed=42)
    pseudo_splits = speaker_aware_kfold(pseudo, folds=3, random_seed=42)
    assert [set(v.audio_file) for _, v in plain_splits] == [
        set(v.audio_file) for _, v in pseudo_splits
    ]


def test_metric_loss_keeps_pseudo_identity_supervision():
    criterion = TwoPartLoss(
        use_ood=False, use_focal=False, competition_known_count=446,
    )
    logits = torch.full((2, 447), -5.0)
    logits[0, 0] = 5.0       # global known label 1 -> head index 0
    logits[1, 446] = 5.0     # global pseudo label 447 -> head index 446
    total, parts = criterion(None, logits, torch.tensor([1, 447]))
    assert torch.isfinite(total)
    assert parts["loss_speaker"] < 0.03


def test_short_audio_tile_and_speech_window_shapes():
    short = torch.tensor([[1.0, -1.0]])
    tiled = fit_short_audio(short, 6, mode="tile_speech")
    torch.testing.assert_close(tiled, torch.tensor([[1.0, -1.0, 1.0, -1.0, 1.0, -1.0]]))

    waveform = torch.zeros(1, 32000)
    waveform[:, 12000:20000] = 0.5
    windows = make_eval_windows(
        waveform, target_length=8000, hop_ratio=0.25, max_windows=3,
        speech_aware=True,
    )
    assert len(windows) == 3
    assert all(w.shape == (1, 8000) for w in windows)
    assert sum(float(w.abs().mean()) for w in windows) > 0.5
