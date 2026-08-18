"""
Tests for the fine-tuning machinery: progressive unfreezing helpers
(``apply_encoder_finetune_mode`` / ``encoder_will_train`` / ``EMA.extend``) and
the ``PrototypicalLoss`` — all CPU-only, no GPU / no real encoder required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training_utils import EMA, apply_encoder_finetune_mode, encoder_will_train  # noqa: E402
from src.train import PrototypicalLoss  # noqa: E402


class DummyEncoder(nn.Module):
    """Mini encoder with a stem + 4 blocks (mirrors unfreeze_last_n_blocks)."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)
        self.blocks = nn.ModuleList([nn.Linear(10, 10) for _ in range(4)])

    def forward(self, x):
        return x

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True

    def unfreeze_last_n_blocks(self, n: int = 2):
        self.freeze()
        for m in self.blocks[-n:]:
            for p in m.parameters():
                p.requires_grad = True


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = DummyEncoder()
        self.head = nn.Linear(10, 10)


# ── encoder_will_train ──

def test_encoder_will_train():
    ft = {"model": {"encoder_type": "campp",
                    "encoder_config": {"campp": {"freeze_encoder": False}}}}
    assert encoder_will_train(ft) is True
    frozen = {"model": {"encoder_type": "campp",
                        "encoder_config": {"campp": {"freeze_encoder": True}}}}
    assert encoder_will_train(frozen) is False
    # WavLM keeps its transformer trainable → progressive N/A
    wavlm = {"model": {"encoder_type": "wavlm", "encoder_config": {
        "wavlm": {"freeze_feature_extractor": True}}}}
    assert encoder_will_train(wavlm) is False


# ── apply_encoder_finetune_mode ──

def test_apply_encoder_finetune_mode_partial():
    model = DummyModel()
    model.encoder.unfreeze()  # start trainable, then restore configured mode
    cfg = {"model": {"encoder_type": "campp", "encoder_config": {
        "campp": {"freeze_encoder": False, "unfreeze_last_n_blocks": 2}}}}
    apply_encoder_finetune_mode(model, cfg)
    assert model.encoder.fc.weight.requires_grad is False
    assert all(p.requires_grad for p in model.encoder.blocks[2].parameters())
    assert all(p.requires_grad for p in model.encoder.blocks[3].parameters())
    assert all(not p.requires_grad for p in model.encoder.blocks[0].parameters())
    assert all(not p.requires_grad for p in model.encoder.blocks[1].parameters())


def test_apply_encoder_finetune_mode_full():
    model = DummyModel()
    model.encoder.freeze()
    cfg = {"model": {"encoder_type": "campp", "encoder_config": {
        "campp": {"freeze_encoder": False, "unfreeze_last_n_blocks": 0}}}}
    apply_encoder_finetune_mode(model, cfg)
    assert all(p.requires_grad for p in model.encoder.parameters())


def test_apply_encoder_finetune_mode_frozen():
    model = DummyModel()
    model.encoder.unfreeze()
    cfg = {"model": {"encoder_type": "campp", "encoder_config": {
        "campp": {"freeze_encoder": True}}}}
    apply_encoder_finetune_mode(model, cfg)
    assert all(not p.requires_grad for p in model.encoder.parameters())


# ── EMA.extend (progressive unfreezing) ──

def test_ema_extend_adds_newly_trainable_params():
    model = DummyModel()
    model.encoder.freeze()  # EMA built while the encoder is frozen
    ema = EMA(model, decay=0.999)
    n_before = len(ema._names)
    model.encoder.unfreeze()
    ema.extend(model)
    n_after = len(ema._names)
    assert n_after > n_before
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert n_after == len(trainable)
    # existing head shadows are preserved (same length as trainable now)
    assert len(ema._shadow) == n_after


# ── PrototypicalLoss ──

def test_prototypical_loss_functional():
    pl = PrototypicalLoss(num_classes=10, embedding_dim=8,
                          scale=10.0, margin=0.1, decay=0.9)
    emb = F.normalize(torch.randn(8, 8), dim=1)
    labels = torch.tensor([0, 1, 1, 2, 0, 3, 4, 5])  # original ids (0 = unknown)
    loss = pl(emb, labels)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert pl.prototypes.shape == (10, 8)


def test_prototypical_loss_all_unknown_is_zero():
    pl = PrototypicalLoss(num_classes=10, embedding_dim=8)
    emb = F.normalize(torch.randn(4, 8), dim=1)
    labels = torch.zeros(4, dtype=torch.long)  # all unknown → masked out
    loss = pl(emb, labels)
    assert loss.item() == 0.0
