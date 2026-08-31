"""CPU-only regression tests for the preregistered P7 OOD-head JSD pair."""

from __future__ import annotations

import math
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment_config import load_profile  # noqa: E402
from src.train import (  # noqa: E402
    TwoPartLoss,
    resolve_ood_jsd_consistency,
    target_clean_aug_bernoulli_jsd,
    train_epoch,
)


CONTROL = "p7-campp-known446-ood-cleanaug-oodjsd-control-lmft-oof-f0"
TREATMENT = "p7-campp-known446-ood-cleanaug-oodjsd-w12-lmft-oof-f0"


def _normalise_identity(config: dict) -> dict:
    config = deepcopy(config)
    config.pop("experiment", None)
    config["logging"]["checkpoint_dir"] = "<profile-checkpoints>"
    config["logging"]["log_dir"] = "<profile-logs>"
    config["hardware"]["profiles"]["vastai_3090_campp"][
        "description"
    ] = "<profile-description>"
    return config


def test_p7_pair_changes_only_the_jsd_multiplier() -> None:
    control = _normalise_identity(load_profile(CONTROL))
    treatment = _normalise_identity(load_profile(TREATMENT))

    for config, expected_weight in ((control, 0.0), (treatment, 12.0)):
        training = config["training"]
        assert training["ood_head_only"] is True
        assert training["epochs"] == 60
        assert training["early_stopping_start_epoch"] == 20
        assert training["early_stopping_patience"] == 12
        assert training["learning_rate"] == pytest.approx(5e-5)
        assert training["warm_start_checkpoint"].endswith(
            "p0-campp-known446-ood-control-oof-f0/campp_best_raw.pt"
        )
        assert config["model"]["encoder_config"]["campp"] == {
            "freeze_encoder": True,
            "local_path": "weights/campp",
            "model_id": "iic/speech_campplus_sv_en_voxceleb_16k",
            "pooling_type": "identity",
            "revision": "v1.0.2",
            "unfreeze_last_n_blocks": 0,
        }
        assert training["loss"]["ood"]["clean_aug_jsd"] == {
            "enabled": True,
            "type": "target_clean_aug_bernoulli_jsd",
            "weight": expected_weight,
        }

    treatment["training"]["loss"]["ood"]["clean_aug_jsd"]["weight"] = 0.0
    assert treatment == control


@pytest.mark.parametrize("weight", [0.0, 12.0])
def test_resolver_keeps_enabled_control_distinct_from_disabled(weight: float) -> None:
    train_cfg = {
        "loss": {
            "ood": {
                "clean_aug_jsd": {
                    "enabled": True,
                    "type": "target_clean_aug_bernoulli_jsd",
                    "weight": weight,
                }
            }
        }
    }
    assert resolve_ood_jsd_consistency(train_cfg) == (
        True,
        weight,
        "target_clean_aug_bernoulli_jsd",
    )

    train_cfg["loss"]["ood"]["clean_aug_jsd"]["enabled"] = False
    assert resolve_ood_jsd_consistency(train_cfg)[0:2] == (False, 0.0)


@pytest.mark.parametrize("weight", [-1.0, float("nan"), float("inf")])
def test_resolver_rejects_invalid_weights(weight: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        resolve_ood_jsd_consistency({
            "loss": {"ood": {"clean_aug_jsd": {
                "enabled": True,
                "weight": weight,
            }}}
        })


def test_bernoulli_jsd_is_finite_and_stops_clean_gradient() -> None:
    augmented = torch.tensor([[-0.8], [0.4], [1.2]], requires_grad=True)
    clean = torch.tensor([[-0.6], [0.3], [1.0]], requires_grad=True)
    labels = torch.tensor([1, 0, 447])

    loss = target_clean_aug_bernoulli_jsd(
        augmented, clean, labels, competition_known_count=446,
    )
    assert loss.item() > 0.0
    assert math.isfinite(loss.item())
    loss.backward()
    assert augmented.grad is not None
    assert torch.isfinite(augmented.grad).all()
    assert clean.grad is None


def test_pseudo_identity_is_a_binary_unknown_target() -> None:
    logits = torch.tensor([[-20.0], [20.0]])
    labels = torch.tensor([1, 447])
    correct = target_clean_aug_bernoulli_jsd(
        logits, logits, labels, competition_known_count=446,
    )
    swapped = target_clean_aug_bernoulli_jsd(
        -logits, -logits, labels, competition_known_count=446,
    )
    assert correct.item() < 1e-6
    assert swapped.item() > 0.6


class TinyOODOnlyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(4, 3)
        self.head_speaker = nn.Linear(3, 2)
        self.head_ood = nn.Linear(3, 1)

    def forward(self, waveforms, labels=None, return_embedding=False):
        del labels
        embedding = torch.tanh(self.encoder(waveforms.flatten(1)))
        outputs = (self.head_ood(embedding), self.head_speaker(embedding))
        return (*outputs, embedding) if return_embedding else outputs


def test_train_epoch_updates_only_ood_head_and_logs_exact_weight() -> None:
    torch.manual_seed(13)
    model = TinyOODOnlyModel()
    model.encoder.requires_grad_(False)
    model.head_speaker.requires_grad_(False)
    criterion = TwoPartLoss(
        use_focal=False,
        ood_weight=0.15,
        speaker_weight=0.85,
        competition_known_count=2,
        speaker_target_scope="known",
    )
    optimizer = torch.optim.SGD(model.head_ood.parameters(), lr=0.05)
    frozen_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }
    ood_before = {
        name: parameter.detach().clone()
        for name, parameter in model.head_ood.named_parameters()
    }
    clean = torch.tensor(
        [[[[0.1, -0.2, 0.3, 0.4]]], [[[0.4, 0.2, -0.1, 0.3]]]]
    )
    augmented = clean + torch.tensor(
        [[[[0.2, -0.1, 0.0, 0.1]]], [[[-0.1, 0.2, 0.1, -0.2]]]]
    )
    batch = (
        {"augmented": augmented, "clean": clean},
        torch.tensor([1, 3]),
    )

    metrics = train_epoch(
        model,
        [batch],
        optimizer,
        criterion,
        torch.amp.GradScaler("cpu", enabled=False),
        torch.device("cpu"),
        autocast_fn=lambda: torch.autocast("cpu", enabled=False),
        ood_jsd_enabled=True,
        ood_jsd_weight=12.0,
        ood_head_only=True,
    )

    assert metrics["loss_ood_jsd"] > 0.0
    assert metrics["loss_ood_jsd_weighted"] == pytest.approx(
        0.15 * 12.0 * metrics["loss_ood_jsd"]
    )
    assert metrics["loss"] == pytest.approx(
        metrics["loss_primary"] + metrics["loss_ood_jsd_weighted"]
    )
    for name, before in frozen_before.items():
        assert torch.equal(dict(model.named_parameters())[name], before)
    assert any(
        not torch.equal(dict(model.head_ood.named_parameters())[name], before)
        for name, before in ood_before.items()
    )
    assert model.encoder.training is False
    assert model.head_speaker.training is False
    assert model.head_ood.training is False
