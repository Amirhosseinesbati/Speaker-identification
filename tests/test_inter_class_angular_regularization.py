from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.experiment_config import load_profile
from src.train import (
    TwoPartLoss,
    exclusive_inter_class_angular_loss,
    resolve_inter_class_regularizer,
    train_epoch,
)


P5_TREATMENT = (
    "p5-campp-known446-ood-crossfile-consistency-c01-long120-oof-f0"
)
P6_TREATMENT = (
    "p6-campp-known446-ood-crossfile-consistency-interclass-e01-long120-oof-f0"
)
P6_CONTROL = (
    "p6-campp-known446-ood-crossfile-consistency-interclass-control-long120-oof-f0"
)
P6_RAW_SHA256 = {
    P6_CONTROL: "2ea8b7a7c9b63f9efc970df7d410f26317b439cfbbd66b53ffd0a9a1545a33b0",
    P6_TREATMENT: "d30c5631b8fd8499a4f2655f7dc41c5e3d5f6b0194ec4cfdcdf40628a5a2dbdc",
}


def _normalise_identity(config: dict) -> dict:
    config = deepcopy(config)
    config.pop("experiment", None)
    config["logging"]["checkpoint_dir"] = "<profile-checkpoints>"
    config["logging"]["log_dir"] = "<profile-logs>"
    config["hardware"]["profiles"]["vastai_3090_campp"][
        "description"
    ] = "<profile-description>"
    return config


def test_exclusive_energy_is_zero_for_nonpositive_off_diagonals() -> None:
    assert exclusive_inter_class_angular_loss(torch.eye(3)).item() == pytest.approx(0.0)
    antipodal = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    assert exclusive_inter_class_angular_loss(antipodal).item() == pytest.approx(0.0)


def test_exclusive_energy_matches_positive_cosine_formula_and_backpropagates() -> None:
    weights = torch.tensor(
        [[1.0, 0.0], [1.0, 1.0], [-1.0, 0.0]], requires_grad=True
    )
    loss = exclusive_inter_class_angular_loss(weights)
    expected = 2.0 * (2.0 ** -0.5) ** 2 / 3.0
    assert loss.item() == pytest.approx(expected)
    loss.backward()
    assert weights.grad is not None
    assert torch.isfinite(weights.grad).all()
    assert weights.grad.abs().sum().item() > 0.0


def test_exclusive_energy_rejects_subcenter_tensor() -> None:
    with pytest.raises(ValueError, match="2-D ArcFace"):
        exclusive_inter_class_angular_loss(torch.randn(3, 2, 4))


def test_exclusive_energy_stays_float32_inside_autocast() -> None:
    weights = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = exclusive_inter_class_angular_loss(weights)
    assert loss.dtype == torch.float32
    assert loss.item() == pytest.approx(0.5)


def test_two_part_loss_uses_exact_convex_speaker_mixture() -> None:
    criterion = TwoPartLoss(
        use_focal=False,
        ood_weight=0.15,
        speaker_weight=0.85,
        competition_known_count=2,
        speaker_target_scope="known",
    )
    ood_logits = torch.tensor([[0.2], [-0.3]])
    speaker_logits = torch.tensor([[1.1, -0.2], [-0.4, 0.9]])
    labels = torch.tensor([1, 2])
    regularizer = torch.tensor(0.5)

    total, metrics = criterion(
        ood_logits,
        speaker_logits,
        labels,
        speaker_regularizer=regularizer,
        speaker_regularizer_weight=0.01,
    )
    raw_speaker = F.cross_entropy(speaker_logits, labels - 1)
    ood_target = torch.zeros_like(ood_logits)
    raw_ood = F.binary_cross_entropy_with_logits(ood_logits, ood_target)
    effective = 0.99 * raw_speaker + 0.01 * regularizer
    expected = 0.15 * raw_ood + 0.85 * effective

    assert total.item() == pytest.approx(expected.item())
    assert metrics["loss_speaker"] == pytest.approx(raw_speaker.item())
    assert metrics["loss_speaker_effective"] == pytest.approx(effective.item())
    assert metrics["loss_inter_class"] == pytest.approx(0.5)
    assert metrics["loss_inter_class_weighted"] == pytest.approx(0.85 * 0.01 * 0.5)


class _TinyAngularModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(4, 3)
        self.head_ood = nn.Linear(3, 1)
        self.head_speaker = nn.Linear(3, 2, bias=False)

    def forward(self, waveforms, labels=None, return_embedding=False):
        del labels
        embedding = F.normalize(self.projection(waveforms.flatten(1)), dim=1)
        outputs = (self.head_ood(embedding), self.head_speaker(embedding))
        return (*outputs, embedding) if return_embedding else outputs


def test_train_epoch_logs_backpropagated_inter_class_objective() -> None:
    torch.manual_seed(19)
    model = _TinyAngularModel()
    with torch.no_grad():
        model.head_speaker.weight.copy_(
            torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        )
    criterion = TwoPartLoss(
        use_focal=False,
        ood_weight=0.15,
        speaker_weight=0.85,
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
        inter_class_weight=0.01,
    )

    assert metrics["loss_inter_class"] > 0.0
    assert metrics["loss_inter_class_weighted"] == pytest.approx(
        0.85 * 0.01 * metrics["loss_inter_class"]
    )
    assert metrics["loss"] == pytest.approx(metrics["loss_primary"])
    assert metrics["loss_speaker_effective"] != pytest.approx(
        metrics["loss_speaker"]
    )


def test_config_parser_keeps_disabled_default_at_zero() -> None:
    weight, inter_type = resolve_inter_class_regularizer({"loss": {}})
    assert weight == 0.0
    assert inter_type == "exclusive_angular_energy"


def test_p6_control_reproduces_p5_treatment_configuration() -> None:
    p5 = _normalise_identity(load_profile(P5_TREATMENT))
    p6_control = _normalise_identity(load_profile(P6_CONTROL))

    assert p6_control == p5


def test_p6_pair_differs_only_by_inter_class_enabled() -> None:
    p6_control = _normalise_identity(load_profile(P6_CONTROL))
    p6_treatment = _normalise_identity(load_profile(P6_TREATMENT))

    assert p6_control["training"]["loss"]["speaker"]["inter_class"] == {
        "enabled": False,
        "type": "exclusive_angular_energy",
        "weight": 0.01,
    }
    assert p6_treatment["training"]["loss"]["speaker"]["inter_class"] == {
        "enabled": True,
        "type": "exclusive_angular_energy",
        "weight": 0.01,
    }
    p6_treatment["training"]["loss"]["speaker"]["inter_class"][
        "enabled"
    ] = False
    assert p6_treatment == p6_control


def test_p6_raw_config_hash_is_preregistered() -> None:
    root = Path(__file__).resolve().parents[1]
    for profile, expected in P6_RAW_SHA256.items():
        path = root / "configs" / "experiments" / f"{profile}.yaml"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
