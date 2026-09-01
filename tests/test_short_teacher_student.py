from __future__ import annotations

import copy
from contextlib import nullcontext

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.heads import ArcFaceHead
from src.experiment_config import load_profile
from src.short_teacher_student import (
    ShortTeacherStudent,
    build_short_teacher_student,
    differentiable_fused_probs,
    teacher_student_losses,
)
from src.metrics import fused_probs_from_logits
from src.train import TwoPartLoss, train_epoch


def _config() -> dict:
    return {
        "audio": {
            "sample_rate": 10,
            "duration_seconds": 8.0,
            "num_train_windows": 1,
        },
        "model": {
            "speaker_head_type": "arcface",
            "speaker_target_scope": "known",
            "ood_head": True,
        },
        "training": {
            "warm_start_checkpoint": "checkpoints/control.pt",
            "loss": {
                "proto": {"enabled": False},
                "consistency": {"enabled": False},
                "short_teacher_student": {
                    "enabled": True,
                    "student_duration_seconds": 2.0,
                    "posterior_weight": 1.0,
                    "embedding_weight": 1.0,
                },
            },
        },
    }


def test_contract_rejects_confounded_objectives() -> None:
    contract = build_short_teacher_student(_config())
    assert contract is not None
    assert contract.student_duration_seconds == 2.0

    config = copy.deepcopy(_config())
    config["training"]["adaptive_margin"] = {"enabled": True}
    with pytest.raises(ValueError, match="distinct hypothesis"):
        build_short_teacher_student(config)

    config = copy.deepcopy(_config())
    config["training"]["loss"]["consistency"]["enabled"] = True
    with pytest.raises(ValueError, match="generic consistency"):
        build_short_teacher_student(config)


def test_p11_preregistration_is_locked_to_fold0_raw_policy() -> None:
    config = load_profile("p11-campp-known446-ood-longshort-ts-oof-f0")
    contract = build_short_teacher_student(config)
    assert contract is not None
    assert contract.student_duration_seconds == 2.0
    assert contract.posterior_weight == 1.0
    assert contract.embedding_weight == 1.0
    assert config["data"]["split"] == {
        "scheme": "kfold",
        "folds": 3,
        "fold": 0,
        "seed": 42,
    }
    assert config["model"]["speaker_target_scope"] == "known"
    assert config["model"]["speaker_head_config"]["arcface"] == {
        "embedding_dim": 192,
        "margin": 0.4,
        "scale": 30.0,
    }
    assert config["training"]["ema_enabled"] is False
    assert config["training"]["early_stopping_start_epoch"] == 10
    assert config["training"]["early_stopping_patience"] == 8
    assert config["hardware"]["profiles"]["vastai_3090_campp"][
        "batch_size"
    ] == 48
    assert config["experiment"]["operational_preflight"][
        "positive_teacher_losses_verified"
    ] is True
    assert config["experiment"]["preregistered_gate"] == {
        "min_macro_f1_gain": 0.002,
        "max_known_accuracy_drop": 0.001,
        "max_ood_f1_drop": 0.001,
        "require_raw_probability_average": True,
        "full_file_evaluation_is_primary": True,
        "fixed_duration_diagnostics": {
            "durations_seconds": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "compare_same_crop_baseline_and_treatment": True,
            "tune_or_select_on_diagnostics": False,
        },
        "expansion_if_passed": "separately_preregister_same_recipe_folds_1_and_2",
        "later_folds_automatic": False,
        "leaderboard_tuning": False,
    }

def test_short_crop_never_reads_right_padding_and_is_reproducible() -> None:
    contract = ShortTeacherStudent(2.0, 1.0, 1.0, sample_rate=10)
    waveforms = torch.zeros(2, 1, 80)
    waveforms[0, 0, :50] = torch.arange(1, 51)
    waveforms[0, 0, 50:] = 9999
    waveforms[1, 0, :10] = torch.arange(1, 11)
    waveforms[1, 0, 10:] = 9999
    durations = torch.tensor([5.0, 1.0])
    kwargs = dict(training_seed=42, epoch=3, step=7, window_index=0)
    left = contract.crop_student_view(waveforms, durations, **kwargs)
    right = contract.crop_student_view(waveforms, durations, **kwargs)
    torch.testing.assert_close(left, right)
    assert tuple(left.shape) == (2, 1, 20)
    assert not torch.any(left == 9999)
    torch.testing.assert_close(
        left[1, 0, :10], torch.arange(1, 11, dtype=torch.float32)
    )
    torch.testing.assert_close(left[1, 0, 10:], torch.zeros(10))


def test_differentiable_fusion_exactly_matches_locked_eval_policy() -> None:
    torch.manual_seed(11)
    ood = torch.randn(7, 1, requires_grad=True)
    speaker = torch.randn(7, 446, requires_grad=True)
    train_probs = differentiable_fused_probs(ood, speaker)
    eval_probs = fused_probs_from_logits(ood, speaker)
    torch.testing.assert_close(train_probs, eval_probs)
    train_probs.square().sum().backward()
    assert ood.grad is not None
    assert speaker.grad is not None


class _Student(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head_speaker = ArcFaceHead(4, 2, embedding_dim=3, scale=10.0)
        self.num_unknown_clusters = 0


def test_teacher_student_loss_backpropagates_only_student() -> None:
    student = _Student()
    student_embedding = F.normalize(
        torch.randn(4, 3, requires_grad=True), dim=1
    )
    student_ood = torch.randn(4, 1, requires_grad=True)
    teacher_embedding = F.normalize(torch.randn(4, 3), dim=1)
    teacher_ood = torch.randn(4, 1)
    teacher_speaker = torch.randn(4, 2)
    losses = teacher_student_losses(
        student_model=student,
        student_ood_logits=student_ood,
        student_embedding=student_embedding,
        teacher_ood_logits=teacher_ood,
        teacher_speaker_logits=teacher_speaker,
        teacher_embedding=teacher_embedding,
        contract=ShortTeacherStudent(2.0, 1.0, 1.0),
    )
    assert losses["posterior_kl"].item() >= 0
    assert -1 <= losses["embedding_cosine"].item() <= 1
    losses["total"].backward()
    assert student_ood.grad is not None
    assert student.head_speaker.weight.grad is not None
    assert teacher_ood.grad is None


class _TinyTwoHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature = nn.Linear(1, 4)
        self.head_ood = nn.Linear(4, 1)
        self.head_speaker = ArcFaceHead(4, 2, embedding_dim=3, scale=10.0)
        self.num_unknown_clusters = 0

    def forward(
        self,
        waveforms: torch.Tensor,
        labels: torch.Tensor | None = None,
        return_embedding: bool = False,
    ):
        pooled = self.feature(waveforms.mean(dim=-1))
        ood_logits = self.head_ood(pooled)
        remapped = None
        if labels is not None:
            remapped = torch.zeros_like(labels)
            known = (labels > 0) & (labels <= 2)
            remapped[known] = labels[known] - 1
        speaker_logits = self.head_speaker(pooled, labels=remapped)
        if return_embedding:
            embedding = F.normalize(
                self.head_speaker.embedding_proj(pooled), dim=1
            )
            return ood_logits, speaker_logits, embedding
        return ood_logits, speaker_logits


def test_train_epoch_executes_real_long_short_objective() -> None:
    torch.manual_seed(7)
    student = _TinyTwoHead()
    teacher = copy.deepcopy(student).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    teacher_before = {
        name: value.detach().clone() for name, value in teacher.state_dict().items()
    }
    student_before = student.head_ood.weight.detach().clone()
    waveforms = torch.randn(4, 1, 80)
    labels = torch.tensor([0, 1, 2, 1])
    durations = torch.tensor([8.0, 8.0, 5.0, 1.0])
    criterion = TwoPartLoss(
        use_focal=False,
        ood_weight=0.15,
        speaker_weight=0.85,
        use_ood=True,
        competition_known_count=2,
        speaker_target_scope="known",
    )
    optimizer = torch.optim.SGD(student.parameters(), lr=1e-2)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    metrics = train_epoch(
        student,
        [(waveforms, labels, durations)],
        optimizer,
        criterion,
        scaler,
        torch.device("cpu"),
        autocast_fn=nullcontext,
        teacher_model=teacher,
        short_teacher_student=ShortTeacherStudent(
            2.0, 1.0, 1.0, sample_rate=10
        ),
        training_seed=42,
        epoch=1,
    )
    assert metrics["loss_teacher_posterior_weighted"] > 0
    assert metrics["loss_teacher_embedding_weighted"] > 0
    assert not torch.equal(student_before, student.head_ood.weight)
    for name, value in teacher.state_dict().items():
        torch.testing.assert_close(value, teacher_before[name])
