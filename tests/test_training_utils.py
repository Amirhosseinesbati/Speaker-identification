"""
Tests for the fine-tuning machinery: progressive unfreezing helpers
(``apply_encoder_finetune_mode`` / ``encoder_will_train`` / ``EMA.extend``) and
the ``PrototypicalLoss`` — all CPU-only, no GPU / no real encoder required.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training_utils import (  # noqa: E402
    EMA,
    apply_encoder_finetune_mode,
    build_scheduler,
    encoder_will_train,
    seed_everything,
)
from src.train import PrototypicalLoss, TwoPartLoss, train_epoch  # noqa: E402


class DummyEncoder(nn.Module):
    """Mini encoder with a stem + 4 blocks (mirrors unfreeze_last_n_blocks)."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)
        self.blocks = nn.ModuleList([nn.Linear(10, 10) for _ in range(4)])
        self.adapter_enabled = False

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

    def enable_se_bn_adapter(self):
        self.freeze()
        self.adapter_enabled = True


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
    se_bn = {"model": {"encoder_type": "ecapa", "encoder_config": {
        "ecapa": {"freeze_encoder": True, "adapter_mode": "se_bn"}}}}
    assert encoder_will_train(se_bn) is True


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


def test_apply_encoder_finetune_mode_ecapa_se_bn_adapter():
    model = DummyModel()
    model.encoder.unfreeze()
    cfg = {"model": {"encoder_type": "ecapa", "encoder_config": {
        "ecapa": {"freeze_encoder": True, "adapter_mode": "se_bn"}}}}
    apply_encoder_finetune_mode(model, cfg)
    assert model.encoder.adapter_enabled is True
    assert all(not p.requires_grad for p in model.encoder.parameters())


# ── EMA.extend (progressive unfreezing) ──

def test_ema_extend_adds_newly_trainable_params():
    model = DummyModel()
    model.encoder.freeze()  # EMA built while the encoder is frozen
    ema = EMA(model, decay=0.999)
    n_before = len(ema._shadow)
    model.encoder.unfreeze()
    ema.extend(model)
    n_after = len(ema._shadow)
    assert n_after > n_before
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert n_after == len(trainable)


def test_ema_update_after_extend_stays_aligned():
    """Regression: update()/state_dict() used positional zip, so after
    extend() the head shadows paired with encoder params (the 192-vs-256
    crash / silent checkpoint corruption under progressive unfreezing).
    Updates must be name-based and each shadow must track its own param."""
    model = DummyModel()
    model.encoder.freeze()
    ema = EMA(model, decay=0.999)
    head_name = "head.weight"
    head_shadow_before = ema._shadow[head_name].clone()

    model.encoder.unfreeze()
    ema.extend(model)
    assert len(ema._shadow) == len([p for p in model.parameters()
                                    if p.requires_grad])

    # Simulate an optimizer step: heads move one way, encoder another.
    with torch.no_grad():
        model.head.weight.add_(1.0)
        for p in model.encoder.parameters():
            p.mul_(2.0)

    ema.update(model)  # must not raise (used to crash on size mismatch)

    # The head shadow must have moved toward the HEAD weight only.
    expected = head_shadow_before * ema.decay + model.head.weight.float() * (
        1.0 - ema.decay
    )
    assert torch.allclose(ema._shadow[head_name], expected, atol=1e-6)

    # Every tracked shadow keeps the shape of its own parameter.
    for name, p in model.named_parameters():
        if name in ema._shadow:
            assert ema._shadow[name].shape == p.shape

    # state_dict substitutes each EMA weight under its own parameter name.
    sd = ema.state_dict(model)
    assert torch.allclose(sd[head_name], ema._shadow[head_name])
    assert torch.allclose(
        sd["encoder.blocks.3.weight"], ema._shadow["encoder.blocks.3.weight"]
    )


def test_ema_average_parameters_restores_raw_weights():
    model = DummyModel()
    ema = EMA(model, decay=0.5)
    raw = model.head.weight.detach().clone()
    with torch.no_grad():
        model.head.weight.add_(2.0)
    changed_raw = model.head.weight.detach().clone()
    ema.update(model)
    expected_ema = raw * 0.5 + changed_raw * 0.5

    with ema.average_parameters(model):
        assert torch.allclose(model.head.weight, expected_ema)
    assert torch.allclose(model.head.weight, changed_raw)


def test_seed_everything_repeats_python_numpy_and_torch():
    policy = seed_everything(2026, deterministic=True)
    first = (random.random(), np.random.rand(), torch.rand(3))
    seed_everything(2026, deterministic=True)
    second = (random.random(), np.random.rand(), torch.rand(3))

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])
    assert policy["seed"] == 2026
    assert policy["deterministic_algorithms"] is True


# ── build_scheduler (per-group cosine floor) ──

def _two_group_optimizer():
    """Encoder group (lr=1e-5) below the head's cosine floor (1.5e-5) — the
    exact configuration that used to invert the encoder's schedule."""
    encoder = nn.Linear(4, 4)
    head = nn.Linear(4, 4)
    return torch.optim.AdamW([
        {"params": list(encoder.parameters()), "lr": 1e-5},
        {"params": list(head.parameters()), "lr": 3e-4},
    ])


def test_scheduler_encoder_lr_never_rises_after_warmup():
    """Regression: eta_min was ``learning_rate * min_lr_ratio`` (one absolute
    floor for every group), so encoder_lr=1e-5 < 1.5e-5 annealed UPWARD. Each
    group must anneal down to its own ``base_lr * min_lr_ratio``."""
    optimizer = _two_group_optimizer()
    cfg = {"schedule": "cosine", "warmup_ratio": 0.1, "min_lr_ratio": 0.05,
           "learning_rate": 3e-4, "epochs": 200}
    scheduler = build_scheduler(optimizer, cfg, cfg["epochs"])
    warmup_epochs = int(round(cfg["epochs"] * cfg["warmup_ratio"]))  # 20

    enc_lrs = []
    head_lrs = []
    for _ in range(1, cfg["epochs"] + 1):
        scheduler.step()
        enc_lrs.append(optimizer.param_groups[0]["lr"])
        head_lrs.append(optimizer.param_groups[1]["lr"])

    # From the end of warmup on, the encoder LR must not rise (was inverted).
    enc_after_warmup = enc_lrs[warmup_epochs - 1]
    assert all(lr <= enc_after_warmup + 1e-12 for lr in enc_lrs[warmup_epochs - 1:])
    # Each group ends at its own floor: encoder 5e-7, head 1.5e-5.
    assert math.isclose(enc_lrs[-1], 1e-5 * 0.05, rel_tol=1e-6)
    assert math.isclose(head_lrs[-1], 3e-4 * 0.05, rel_tol=1e-6)
    # The head anneals monotonically after warmup (it ramps up during it).
    assert all(b <= a + 1e-12
               for a, b in zip(head_lrs[warmup_epochs - 1:],
                               head_lrs[warmup_epochs - 1:][1:]))


def test_scheduler_no_warmup_per_group_floor():
    optimizer = _two_group_optimizer()
    cfg = {"schedule": "cosine", "warmup_ratio": 0.0, "min_lr_ratio": 0.05,
           "learning_rate": 3e-4, "epochs": 100}
    scheduler = build_scheduler(optimizer, cfg, cfg["epochs"])
    for _ in range(1, cfg["epochs"] + 1):
        scheduler.step()
    assert math.isclose(optimizer.param_groups[0]["lr"], 1e-5 * 0.05, rel_tol=1e-6)
    assert math.isclose(optimizer.param_groups[1]["lr"], 3e-4 * 0.05, rel_tol=1e-6)


def test_scheduler_preserves_zero_lr_for_fixed_warm_started_heads():
    adapter = nn.Linear(4, 4)
    head = nn.Linear(4, 4)
    optimizer = torch.optim.AdamW([
        {"params": list(adapter.parameters()), "lr": 1e-5},
        {"params": list(head.parameters()), "lr": 0.0},
    ])
    cfg = {
        "schedule": "cosine",
        "warmup_ratio": 0.05,
        "min_lr_ratio": 0.05,
        "epochs": 45,
    }
    scheduler = build_scheduler(optimizer, cfg, cfg["epochs"])

    adapter_lrs = []
    head_lrs = []
    for _ in range(cfg["epochs"]):
        optimizer.step()
        scheduler.step()
        adapter_lrs.append(optimizer.param_groups[0]["lr"])
        head_lrs.append(optimizer.param_groups[1]["lr"])

    assert all(0.0 < lr <= 1e-5 for lr in adapter_lrs)
    assert all(lr == 0.0 for lr in head_lrs)
    assert math.isclose(adapter_lrs[-1], 1e-5 * 0.05, rel_tol=1e-6)


def test_exponential_scheduler_reaches_paper_endpoint_for_every_group():
    optimizer = _two_group_optimizer()
    cfg = {
        "schedule": "exponential",
        "warmup_ratio": 0.0,
        "min_lr_ratio": 0.25,
        "epochs": 10,
    }
    scheduler = build_scheduler(optimizer, cfg, cfg["epochs"])
    enc_lrs = []
    head_lrs = []
    for _ in range(cfg["epochs"]):
        optimizer.step()
        scheduler.step()
        enc_lrs.append(optimizer.param_groups[0]["lr"])
        head_lrs.append(optimizer.param_groups[1]["lr"])

    assert all(b < a for a, b in zip(enc_lrs, enc_lrs[1:]))
    assert all(b < a for a, b in zip(head_lrs, head_lrs[1:]))
    assert math.isclose(enc_lrs[-1], 1e-5 * 0.25, rel_tol=1e-6)
    assert math.isclose(head_lrs[-1], 3e-4 * 0.25, rel_tol=1e-6)


@pytest.mark.parametrize(
    "cfg",
    [
        {"schedule": "exponential", "warmup_ratio": 0.1, "min_lr_ratio": 0.25},
        {"schedule": "exponential", "warmup_ratio": 0.0, "min_lr_ratio": 0.0},
    ],
)
def test_exponential_scheduler_rejects_ambiguous_protocol(cfg: dict) -> None:
    with pytest.raises(ValueError):
        build_scheduler(_two_group_optimizer(), cfg, 10)


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


def test_prototypical_loss_backward():
    """Regression: forward scattered the AM-softmax margin IN-PLACE into the
    same tensor the margin was gathered from, so backward needed it at two
    versions at once (RuntimeError: ... modified by an inplace operation).
    The loss must be differentiable."""
    pl = PrototypicalLoss(num_classes=10, embedding_dim=8,
                          scale=10.0, margin=0.1, decay=0.9)
    x = torch.randn(8, 8, requires_grad=True)
    emb = F.normalize(x, dim=1)  # non-leaf; gradient lands on x
    labels = torch.tensor([1, 1, 2, 3, 4, 5, 6, 7])  # no unknown → all contribute
    loss = pl(emb, labels)
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_prototypical_loss_all_unknown_is_zero():
    pl = PrototypicalLoss(num_classes=10, embedding_dim=8)
    emb = F.normalize(torch.randn(4, 8), dim=1)
    labels = torch.zeros(4, dtype=torch.long)  # all unknown → masked out
    loss = pl(emb, labels)
    assert loss.item() == 0.0


def test_train_epoch_logs_backpropagated_proto_objective():
    class TinySpeakerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(4, 3)
            self.head_ood = nn.Linear(3, 1)
            self.head_speaker = nn.Linear(3, 2)

        def forward(self, waveforms, labels=None, return_embedding=False):
            del labels
            embedding = F.normalize(
                self.projection(waveforms.flatten(1)), dim=1
            )
            outputs = (self.head_ood(embedding), self.head_speaker(embedding))
            return (*outputs, embedding) if return_embedding else outputs

    class ConstantProtoLoss(nn.Module):
        def forward(self, embeddings, labels):
            del labels
            return embeddings.sum() * 0.0 + 2.0

    model = TinySpeakerModel()
    criterion = TwoPartLoss(
        use_focal=False,
        ood_weight=0.1425,
        speaker_weight=0.8075,
        competition_known_count=2,
        speaker_target_scope="known",
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    batch = (torch.randn(2, 1, 4), torch.tensor([1, 2]))

    metrics = train_epoch(
        model,
        [batch],
        optimizer,
        criterion,
        torch.amp.GradScaler("cpu", enabled=False),
        torch.device("cpu"),
        autocast_fn=lambda: torch.autocast("cpu", enabled=False),
        proto_criterion=ConstantProtoLoss(),
        proto_weight=0.05,
    )

    assert metrics["loss_proto"] == pytest.approx(2.0)
    assert metrics["loss_proto_weighted"] == pytest.approx(0.1)
    assert metrics["loss"] == pytest.approx(metrics["loss_primary"] + 0.1)
    assert metrics["loss_primary_normalized"] == pytest.approx(
        metrics["loss_primary"] / 0.95
    )
