"""CPU-only regression tests for paired clean/aug embedding consistency."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline import SpeakerDataset  # noqa: E402
from src.train import TwoPartLoss, train_epoch  # noqa: E402


def _dataset(*, paired: bool, mixup_alpha: float = 0.0) -> SpeakerDataset:
    dataset = SpeakerDataset(
        pd.DataFrame([{"audio_file": "unused.wav", "label": 1}]),
        audio_dir=".",
        sample_rate=10,
        duration_seconds=1.0,
        augment=True,
        mixup_alpha=mixup_alpha,
        num_train_windows=2,
        augmentation={},
        return_clean_aug_pair=paired,
    )
    dataset._load_audio = lambda _path: torch.arange(20, dtype=torch.float32).view(1, -1)
    dataset.augmentor = lambda waveform: waveform + 10.0
    return dataset


def test_paired_dataset_uses_identical_crop_before_augmentation():
    torch.manual_seed(7)
    views, label = _dataset(paired=True)[0]

    assert set(views) == {"augmented", "clean"}
    assert views["augmented"].shape == (2, 1, 10)
    assert views["clean"].shape == views["augmented"].shape
    assert torch.equal(
        views["augmented"] - views["clean"],
        torch.full_like(views["clean"], 10.0),
    )
    assert label.item() == 1


def test_default_dataset_return_is_unchanged_tensor():
    torch.manual_seed(7)
    windows, _ = _dataset(paired=False)[0]
    assert isinstance(windows, torch.Tensor)
    assert windows.shape == (2, 1, 10)


def test_paired_dataset_rejects_mixup():
    with pytest.raises(ValueError, match="incompatible with mixup"):
        _dataset(paired=True, mixup_alpha=0.2)


class TinyPairedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 3)
        self.head_ood = nn.Linear(3, 1)
        self.head_speaker = nn.Linear(3, 2)

    def embed(self, waveforms):
        return F.normalize(self.projection(waveforms.flatten(1)), dim=1)

    def forward(self, waveforms, labels=None, return_embedding=False):
        del labels
        embedding = self.embed(waveforms)
        outputs = (self.head_ood(embedding), self.head_speaker(embedding))
        return (*outputs, embedding) if return_embedding else outputs


def test_train_epoch_backpropagates_logged_consistency_objective():
    torch.manual_seed(11)
    model = TinyPairedModel()
    criterion = TwoPartLoss(
        use_focal=False,
        ood_weight=0.15,
        speaker_weight=0.85,
        competition_known_count=2,
        speaker_target_scope="known",
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    clean = torch.randn(2, 1, 4, requires_grad=True)
    augmented = clean.detach().clone() + torch.tensor(
        [[[-0.3, 0.2, 0.1, 0.4]], [[0.2, -0.4, 0.3, -0.1]]]
    )
    batch = (
        {"augmented": augmented, "clean": clean},
        torch.tensor([1, 2]),
    )

    metrics = train_epoch(
        model,
        [batch],
        optimizer,
        criterion,
        torch.amp.GradScaler("cpu", enabled=False),
        torch.device("cpu"),
        autocast_fn=lambda: torch.autocast("cpu", enabled=False),
        consistency_weight=0.1,
    )

    assert metrics["loss_consistency"] > 0.0
    assert metrics["loss_consistency_weighted"] == pytest.approx(
        metrics["loss_consistency"] * 0.1
    )
    assert metrics["loss"] == pytest.approx(
        metrics["loss_primary"] + metrics["loss_consistency_weighted"]
    )
    assert -1.0 <= metrics["pair_cosine"] <= 1.0
    assert metrics["embedding_std_augmented"] > 0.0
    assert metrics["embedding_std_clean"] > 0.0
    assert clean.grad is None


def test_train_epoch_rejects_consistency_without_paired_views():
    model = TinyPairedModel()
    criterion = TwoPartLoss(
        use_focal=False,
        competition_known_count=2,
        speaker_target_scope="known",
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    batch = (torch.randn(2, 1, 4), torch.tensor([1, 2]))

    with pytest.raises(ValueError, match="requires paired"):
        train_epoch(
            model,
            [batch],
            optimizer,
            criterion,
            torch.amp.GradScaler("cpu", enabled=False),
            torch.device("cpu"),
            autocast_fn=lambda: torch.autocast("cpu", enabled=False),
            consistency_weight=0.1,
        )
