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
from src.pooling import IdentityPooling  # noqa: E402
from src.train import (  # noqa: E402
    TwoPartLoss,
    compute_ood_accuracy,
    forward_multi_window,
    tune_ood_threshold,
)
from src.metrics import fused_probs_from_logits  # noqa: E402


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