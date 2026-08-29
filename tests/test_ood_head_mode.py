"""
Tests for the config-gated OOD-head removal (cluster mode).

Covers the pure logic + a CPU-only tiny model so no pretrained encoder / GPU
is required:

  - ood_head_enabled / create_ood_head flag semantics (explicit flag wins,
    default ON for backward compat).
  - TwoHeadedSpeakerModel with ood_head=None: forward returns (None, spk),
    predict_proba / predict_proba_and_embed produce valid 447-way probs whose
    unknown mass comes from the cluster collapse.
  - TwoPartLoss with use_ood=False (speaker-only loss, ood_logits=None ok).
  - fused_probs_from_logits / compute_ood_accuracy / tune_ood_threshold with
    ood_logits=None.
  - forward_multi_window multi-window path with the OOD head absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.heads import (  # noqa: E402
    OODHead,
    ArcFaceHead,
    create_ood_head,
    ood_head_enabled,
)
from src.model import TwoHeadedSpeakerModel  # noqa: E402
from src.model_factory import resolve_speaker_head_layout  # noqa: E402
from src.pooling import IdentityPooling  # noqa: E402
from src.train import (  # noqa: E402
    TwoPartLoss,
    compute_ood_accuracy,
    forward_multi_window,
    forward_multi_window_evaluation,
    tune_ood_threshold,
)
from src.metrics import fused_probs_from_logits  # noqa: E402
from src.data_pipeline import (  # noqa: E402
    _sampling_rows_sha256,
    load_known_sampling_weights,
    make_balanced_batch_sampler,
)


class DummyEncoder(nn.Module):
    """Tiny encoder: (B, 1, T) → (B, 1, 8) + None lengths (identity pooling)."""

    output_dim = 8

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(1, 8)

    def forward(self, waveforms):
        x = waveforms.mean(dim=2, keepdim=True)  # (B, 1, 1)
        return self.proj(x), None


def _model(num_unknown_clusters: int, ood_head: bool = True) -> TwoHeadedSpeakerModel:
    enc = DummyEncoder()
    pool = IdentityPooling()
    spk = ArcFaceHead(input_dim=8, num_classes=446 + num_unknown_clusters,
                      embedding_dim=8, margin=0.3, scale=32.0)
    ood = OODHead(8, hidden_dim=4) if ood_head else None
    return TwoHeadedSpeakerModel(
        encoder=enc, pooling=pool, speaker_head=spk, ood_head=ood,
        num_known_speakers=446 + num_unknown_clusters,
        num_unknown_clusters=num_unknown_clusters,
    )


# ── flag semantics ──

def test_ood_head_enabled_flag_wins_and_default_on():
    assert ood_head_enabled({"model": {"ood_head": False, "num_unknown_clusters": 554}}) is False
    assert ood_head_enabled({"model": {"ood_head": True, "num_unknown_clusters": 554}}) is True
    # no explicit flag → default ON (backward compat with old checkpoints)
    assert ood_head_enabled({"model": {"num_unknown_clusters": 554}}) is True
    assert ood_head_enabled({"model": {}}) is True


def test_create_ood_head_none_when_disabled():
    cfg = {"model": {"ood_head": False, "num_unknown_clusters": 554,
                     "ood_head_config": {"hidden_dim": 4}}}
    assert create_ood_head(cfg, 8) is None
    cfg["model"]["ood_head"] = True
    assert isinstance(create_ood_head(cfg, 8), OODHead)


def test_known_scope_decouples_primary_head_from_metric_pseudo_classes():
    cfg = {"model": {"competition_num_known": 446,
                     "speaker_target_scope": "known"}}
    head_classes, output_pseudo, scope = resolve_speaker_head_layout(cfg, 1000)
    assert (head_classes, output_pseudo, scope) == (446, 0, "known")


def test_metric_scope_preserves_legacy_1000_way_head():
    cfg = {"model": {"competition_num_known": 446,
                     "speaker_target_scope": "metric"}}
    head_classes, output_pseudo, scope = resolve_speaker_head_layout(cfg, 1000)
    assert (head_classes, output_pseudo, scope) == (1000, 554, "metric")


# ── model forward / proba paths ──

def test_forward_returns_none_ood_when_head_disabled():
    m = _model(num_unknown_clusters=554, ood_head=False)
    m.eval()
    x = torch.randn(2, 1, 32)
    ood, spk = m(x, labels=None)
    assert ood is None
    assert spk.shape == (2, 1000)


def test_predict_proba_legacy_without_ood_has_zero_unknown():
    m = _model(num_unknown_clusters=0, ood_head=False)
    m.eval()
    p = m.predict_proba(torch.randn(2, 1, 32))
    assert p.shape == (2, 447)
    torch.testing.assert_close(p[:, 0], torch.zeros(2), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(p.sum(1), torch.ones(2), atol=1e-4, rtol=1e-4)


def test_predict_proba_cluster_without_ood_unknown_is_cluster_collapse():
    m = _model(num_unknown_clusters=554, ood_head=False)
    m.eval()
    x = torch.randn(2, 1, 32)
    p = m.predict_proba(x)
    assert p.shape == (2, 447)
    # unknown mass must equal the summed cluster softmax columns
    spk = m.head_speaker(m.pooling(m.encoder(x)[0]))
    exp_unknown = torch.softmax(spk, dim=1)[:, 446:].sum(dim=1)
    # predict_proba clamps to [1e-7, 1-1e-7] and renormalises → ~1e-4 tolerance
    torch.testing.assert_close(p[:, 0], exp_unknown, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(p.sum(1), torch.ones(2), atol=1e-4, rtol=1e-4)


def test_predict_proba_and_embed_without_ood():
    m = _model(num_unknown_clusters=554, ood_head=False)
    m.eval()
    w = torch.randn(3, 1, 32)  # (W, 1, T) single file, multi-window
    probs, emb = m.predict_proba_and_embed(w, temperature=1.0)
    assert probs.shape == (447,)
    assert emb.shape == (8,)
    torch.testing.assert_close(probs.sum(), torch.tensor(1.0), atol=1e-4, rtol=1e-4)


def test_forward_multi_window_without_ood():
    m = _model(num_unknown_clusters=0, ood_head=False)
    m.eval()
    x = torch.randn(2, 3, 1, 32)  # (B, W, 1, T)
    ood, spk = forward_multi_window(m, x, labels=None)
    assert ood is None
    assert spk.shape == (2, 446)


def test_multi_window_evaluation_averages_probabilities_per_window():
    m = _model(num_unknown_clusters=0, ood_head=True)
    m.eval()
    x = torch.randn(2, 3, 1, 32)

    mean_ood, mean_spk, mean_probs = forward_multi_window_evaluation(m, x)
    manual_ood, manual_spk, manual_probs = [], [], []
    for window_idx in range(x.shape[1]):
        ood, spk = m(x[:, window_idx], labels=None)
        manual_ood.append(ood)
        manual_spk.append(spk)
        manual_probs.append(fused_probs_from_logits(ood, spk))

    torch.testing.assert_close(mean_ood, torch.stack(manual_ood).mean(0))
    torch.testing.assert_close(mean_spk, torch.stack(manual_spk).mean(0))
    torch.testing.assert_close(mean_probs, torch.stack(manual_probs).mean(0))
    torch.testing.assert_close(
        mean_probs.sum(1), torch.ones(x.shape[0]), atol=1e-6, rtol=1e-6,
    )


def test_known_only_arcface_accepts_pseudo_metric_labels_safely():
    m = _model(num_unknown_clusters=0, ood_head=True)
    m.train()
    x = torch.randn(4, 1, 32)
    # Labels 447/1000 exist in the metric map but are outside the 446-way head.
    ood, spk, emb = m(
        x, labels=torch.tensor([1, 447, 1000, 0]), return_embedding=True,
    )
    assert ood.shape == (4, 1)
    assert spk.shape == (4, 446)
    assert emb.shape == (4, 8)


# ── loss ──

def test_two_part_loss_without_ood_is_speaker_only():
    crit = TwoPartLoss(use_ood=False, use_focal=False, speaker_weight=1.0)
    spk = torch.randn(4, 446)
    labels = torch.tensor([1, 0, 2, 3])
    total, d = crit(None, spk, labels)
    assert d["loss_ood"] == 0.0
    assert d["loss_total"] == pytest.approx(d["loss_speaker"], rel=1e-6)
    assert torch.isfinite(total)


def test_two_part_loss_with_ood_head_still_works():
    crit = TwoPartLoss(use_ood=True, use_focal=False)
    ood = torch.randn(4, 1)
    spk = torch.randn(4, 446)
    labels = torch.tensor([1, 0, 2, 3])
    total, d = crit(ood, spk, labels)
    assert d["loss_ood"] > 0.0
    assert torch.isfinite(total)


def test_known_scope_masks_pseudo_ids_from_speaker_ce_but_not_ood_bce():
    crit = TwoPartLoss(
        use_ood=True, use_focal=False, speaker_target_scope="known",
        competition_known_count=446,
    )
    ood = torch.randn(4, 1, requires_grad=True)
    spk = torch.randn(4, 446, requires_grad=True)
    labels = torch.tensor([1, 447, 2, 1000])
    total, parts = crit(ood, spk, labels)

    expected_speaker = torch.nn.functional.cross_entropy(
        spk[[0, 2]], torch.tensor([0, 1]),
    )
    assert parts["loss_speaker"] == pytest.approx(expected_speaker.item(), rel=1e-6)
    assert parts["loss_ood"] > 0.0
    total.backward()
    # Pseudo rows must have no primary-speaker CE gradient.
    torch.testing.assert_close(spk.grad[[1, 3]], torch.zeros_like(spk.grad[[1, 3]]))
    assert ood.grad is not None


def test_known_first_sampler_treats_pseudo_ids_as_ood():
    labels = torch.tensor(
        [1, 2, 3, 4, 447, 448, 999, 1000], dtype=torch.long,
    ).numpy()
    sampler = make_balanced_batch_sampler(
        labels, batch_size=4, ood_ratio=0.5, seed=7,
        competition_known_count=446,
    )
    sampled = labels[np.asarray(list(sampler))]
    for batch in sampled:
        assert ((batch == 0) | (batch > 446)).sum() == 2
        assert ((batch > 0) & (batch <= 446)).sum() == 2


def test_balanced_sampler_is_deterministic_per_epoch_and_changes_between_epochs():
    labels = np.asarray([0] * 8 + list(range(1, 17)), dtype=np.int64)
    sampler = make_balanced_batch_sampler(
        labels, batch_size=6, ood_ratio=1 / 3, seed=13,
    )

    epoch_zero_first = list(sampler)
    epoch_zero_second = list(sampler)
    assert epoch_zero_first == epoch_zero_second
    for batch_indices in epoch_zero_first:
        batch = labels[np.asarray(batch_indices)]
        assert (batch == 0).sum() == 2
        assert (batch != 0).sum() == 4

    sampler.set_epoch(1)
    epoch_one = list(sampler)
    assert epoch_one != epoch_zero_first


def test_weighted_known_sampler_preserves_ratio_and_ood_stream():
    labels = np.asarray([0] * 20 + list(range(1, 21)), dtype=np.int64)
    weights = np.ones_like(labels, dtype=np.float64)
    weights[20] = 12.0

    plain = make_balanced_batch_sampler(
        labels, batch_size=4, ood_ratio=0.5, seed=23,
    )
    weighted = make_balanced_batch_sampler(
        labels, batch_size=4, ood_ratio=0.5, seed=23,
        known_sampling_weights=weights,
    )

    plain_batches = list(plain)
    weighted_batches = list(weighted)
    for plain_batch, weighted_batch in zip(plain_batches, weighted_batches):
        plain_idx = np.asarray(plain_batch)
        weighted_idx = np.asarray(weighted_batch)
        assert (labels[plain_idx] == 0).sum() == 2
        assert (labels[weighted_idx] == 0).sum() == 2
        assert sorted(plain_idx[plain_idx < 20].tolist()) == sorted(
            weighted_idx[weighted_idx < 20].tolist()
        )


def test_weighted_known_sampler_is_deterministic_and_increases_exposure():
    labels = np.asarray([0] * 100 + list(range(1, 101)), dtype=np.int64)
    weights = np.ones_like(labels, dtype=np.float64)
    hard_index = 100
    weights[hard_index] = 20.0
    sampler = make_balanced_batch_sampler(
        labels, batch_size=10, ood_ratio=0.5, seed=31,
        known_sampling_weights=weights,
    )

    first = list(sampler)
    second = list(sampler)
    assert first == second

    known_draws = np.concatenate([
        np.asarray(batch)[np.asarray(batch) >= 100] for batch in first
    ])
    counts = np.bincount(known_draws - 100, minlength=100)
    assert counts[0] >= 5
    assert counts[0] > 4 * np.median(counts[1:])


def test_weighted_known_sampler_rejects_invalid_weights():
    labels = np.asarray([0, 0, 1, 2], dtype=np.int64)
    with pytest.raises(ValueError, match="match train_labels shape"):
        make_balanced_batch_sampler(
            labels, batch_size=4, known_sampling_weights=np.ones(3),
        )
    with pytest.raises(ValueError, match="strictly positive"):
        make_balanced_batch_sampler(
            labels, batch_size=4,
            known_sampling_weights=np.asarray([1.0, 1.0, 0.0, 1.0]),
        )


def test_known_sampling_artifact_is_split_locked(tmp_path):
    import json
    import pandas as pd

    train_df = pd.DataFrame({
        "audio_file": ["ood.wav", "a.wav", "b.wav"],
        "label": [0, 1, 2],
    })
    artifact = tmp_path / "weights.json"
    artifact.write_text(json.dumps({
        "schema_version": 1,
        "training_rows_sha256": _sampling_rows_sha256(train_df),
        "weights": {"a.wav": 2.0, "b.wav": 1.0},
    }), encoding="utf-8")
    config = {"data": {"known_sampling": {"weights_path": str(artifact)}}}

    weights = load_known_sampling_weights(config, train_df)
    np.testing.assert_array_equal(weights, np.asarray([1.0, 2.0, 1.0]))

    changed = train_df.copy()
    changed.loc[0, "audio_file"] = "different.wav"
    with pytest.raises(ValueError, match="current training split"):
        load_known_sampling_weights(config, changed)


def test_known_sampling_artifact_rejects_validation_or_missing_keys(tmp_path):
    import json
    import pandas as pd

    train_df = pd.DataFrame({
        "audio_file": ["ood.wav", "a.wav", "b.wav"],
        "label": [0, 1, 2],
    })
    artifact = tmp_path / "weights.json"
    artifact.write_text(json.dumps({
        "schema_version": 1,
        "training_rows_sha256": _sampling_rows_sha256(train_df),
        "weights": {"a.wav": 2.0, "validation.wav": 1.0},
    }), encoding="utf-8")
    config = {"data": {"known_sampling": {"weights_path": str(artifact)}}}

    with pytest.raises(ValueError, match="exactly match known training files"):
        load_known_sampling_weights(config, train_df)


# ── metrics / threshold helpers ──

def test_fused_probs_from_logits_accepts_none_ood():
    spk = torch.randn(2, 1000)
    p = fused_probs_from_logits(None, spk, num_unknown_clusters=554)
    assert p.shape == (2, 447)
    torch.testing.assert_close(p.sum(1), torch.ones(2), atol=1e-4, rtol=1e-4)
    exp_unknown = torch.softmax(spk, dim=1)[:, 446:].sum(dim=1)
    torch.testing.assert_close(p[:, 0], exp_unknown, atol=1e-5, rtol=1e-5)


def test_compute_ood_accuracy_none_returns_zero():
    assert compute_ood_accuracy(None, torch.tensor([0, 1, 2])) == 0.0


def test_tune_ood_threshold_none_returns_none():
    assert tune_ood_threshold(None, torch.tensor([0, 1, 2])) is None
