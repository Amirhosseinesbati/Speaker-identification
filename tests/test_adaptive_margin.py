from copy import deepcopy
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch
import yaml

from src.adaptive_margin import (
    DurationAdaptiveMargin,
    build_duration_adaptive_margin,
)
from src.heads import ArcFaceHead
from src.train import TwoPartLoss, train_epoch


ROOT = Path(__file__).resolve().parents[1]
PAPER_PROFILE = (
    ROOT
    / "configs/experiments/p10-campp-known446-ood-dalmft-paper-oof-f0.yaml"
)


def _full_config() -> dict:
    return {
        "audio": {
            "sample_rate": 16_000,
            "duration_seconds": 8.0,
        },
        "model": {
            "encoder_type": "campp",
            "speaker_head_type": "arcface",
            "speaker_target_scope": "known",
            "encoder_config": {
                "campp": {
                    "freeze_encoder": False,
                    "unfreeze_last_n_blocks": 0,
                }
            },
        },
        "training": {
            "warm_start_checkpoint": "checkpoints/control/campp_best_raw.pt",
            "loss": {"proto": {"enabled": False}},
            "adaptive_margin": {
                "enabled": True,
                "strategy": "duration_linear",
                "disable_augmentation": True,
                "min_duration_seconds": 1.0,
                "max_duration_seconds": 6.0,
                "margin_anchor_min_duration_seconds": 2.0,
                "margin_anchor_max_duration_seconds": 6.0,
                "min_margin": 0.2,
                "max_margin": 0.5,
                "seed_offset": 17,
            },
        },
    }


def test_duration_margin_mapping_has_paper_endpoints() -> None:
    contract = build_duration_adaptive_margin(_full_config())
    assert contract is not None

    durations = torch.tensor([0.4, 1.0, 2.0, 4.0, 6.0, 9.0])
    margins = contract.margin_for_duration(durations)

    assert torch.allclose(
        margins,
        torch.tensor([0.2, 0.2, 0.2, 0.35, 0.5, 0.5]),
        atol=1e-7,
    )


def test_sampling_and_cropping_are_deterministic_without_global_rng() -> None:
    contract = DurationAdaptiveMargin(
        min_duration_seconds=1.0,
        max_duration_seconds=6.0,
        margin_anchor_min_duration_seconds=2.0,
        margin_anchor_max_duration_seconds=6.0,
        min_margin=0.2,
        max_margin=0.5,
        sample_rate=10,
        seed_offset=23,
    )
    torch.manual_seed(101)
    expected_next_global_draw = torch.rand(1)
    torch.manual_seed(101)

    duration_a = contract.sample_duration_seconds(
        training_seed=42, epoch=3, step=8, window_index=1
    )
    duration_b = contract.sample_duration_seconds(
        training_seed=42, epoch=3, step=8, window_index=1
    )
    waveforms = torch.arange(2 * 1 * 80, dtype=torch.float32).reshape(2, 1, 80)
    crop_a = contract.crop_batch(
        waveforms,
        duration_a,
        training_seed=42,
        epoch=3,
        step=8,
        window_index=1,
    )
    crop_b = contract.crop_batch(
        waveforms,
        duration_b,
        training_seed=42,
        epoch=3,
        step=8,
        window_index=1,
    )

    assert duration_a == duration_b
    assert 1.0 <= duration_a <= 6.0
    assert torch.equal(crop_a, crop_b)
    assert crop_a.shape[-1] == round(duration_a * contract.sample_rate)
    assert torch.equal(torch.rand(1), expected_next_global_draw)


def test_crop_never_uses_right_padding_beyond_source_duration() -> None:
    contract = DurationAdaptiveMargin(
        min_duration_seconds=1.0,
        max_duration_seconds=6.0,
        margin_anchor_min_duration_seconds=2.0,
        margin_anchor_max_duration_seconds=6.0,
        min_margin=0.2,
        max_margin=0.5,
        sample_rate=10,
        seed_offset=29,
    )
    waveforms = torch.ones(2, 1, 80)
    # Deliberately place non-zero sentinel values in the area SpeakerDataset
    # declares to be padding.  A correct D-ALMFT crop must not consume them.
    source_durations = torch.tensor([0.5, 8.0])
    cropped = contract.crop_batch(
        waveforms,
        4.0,
        source_durations_seconds=source_durations,
        training_seed=42,
        epoch=1,
        step=0,
        window_index=0,
    )
    assert cropped.shape == (2, 1, 40)
    assert torch.equal(cropped[0, 0, :5], torch.ones(5))
    assert torch.equal(cropped[0, 0, 5:], torch.zeros(35))
    assert torch.equal(cropped[1], torch.ones_like(cropped[1]))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda cfg: cfg["training"].update(warm_start_checkpoint=None),
            "secondary stage",
        ),
        (
            lambda cfg: cfg["training"]["adaptive_margin"].update(
                disable_augmentation=False
            ),
            "disable_augmentation",
        ),
        (
            lambda cfg: cfg["model"]["encoder_config"]["campp"].update(
                unfreeze_last_n_blocks=1
            ),
            "full-encoder",
        ),
        (
            lambda cfg: cfg["training"]["loss"]["proto"].update(enabled=True),
            "prototype loss",
        ),
        (
            lambda cfg: cfg["training"]["adaptive_margin"].update(
                margin_anchor_min_duration_seconds=0.5
            ),
            "sample_min_duration",
        ),
    ],
)
def test_contract_rejects_historical_lmft_shortcuts(mutation, message: str) -> None:
    config = deepcopy(_full_config())
    mutation(config)
    with pytest.raises(ValueError, match=message):
        build_duration_adaptive_margin(config)


def test_arcface_batch_margin_matches_fixed_margin() -> None:
    torch.manual_seed(5)
    head = ArcFaceHead(
        input_dim=6,
        num_classes=3,
        embedding_dim=4,
        margin=0.35,
        scale=17.0,
    ).eval()
    embeddings = torch.randn(4, 6)
    labels = torch.tensor([0, 1, 2, 0])

    fixed = head(embeddings, labels=labels)
    adaptive = head(
        embeddings,
        labels=labels,
        margins=torch.full((4,), 0.35),
    )

    assert torch.allclose(adaptive, fixed, atol=2e-6, rtol=1e-6)


def test_arcface_per_sample_margin_changes_only_target_logit() -> None:
    torch.manual_seed(7)
    head = ArcFaceHead(
        input_dim=5,
        num_classes=3,
        embedding_dim=4,
        margin=0.2,
        scale=12.0,
    ).eval()
    embeddings = torch.randn(2, 5)
    labels = torch.tensor([0, 2])

    zero = head(embeddings, labels=labels, margins=torch.zeros(2))
    adaptive = head(embeddings, labels=labels, margins=torch.tensor([0.2, 0.5]))

    for row, target in enumerate(labels.tolist()):
        non_target = [idx for idx in range(3) if idx != target]
        assert torch.allclose(adaptive[row, non_target], zero[row, non_target])
        assert adaptive[row, target] < zero[row, target]


def test_arcface_rejects_invalid_margin_shape() -> None:
    head = ArcFaceHead(input_dim=4, num_classes=2, embedding_dim=3).eval()
    with pytest.raises(ValueError, match="scalar or shaped"):
        head(
            torch.randn(2, 4),
            labels=torch.tensor([0, 1]),
            margins=torch.ones(2, 1),
        )


def test_preregistered_profile_matches_published_dalmft_core() -> None:
    config = yaml.safe_load(PAPER_PROFILE.read_text(encoding="utf-8"))
    contract = build_duration_adaptive_margin(config)
    assert contract is not None

    assert config["data"]["split"] == {
        "scheme": "kfold",
        "folds": 3,
        "fold": 0,
        "seed": 42,
    }
    assert config["audio"]["num_train_windows"] == 1
    assert config["audio"]["speech_aware_crop_probability"] == 0.0
    assert (contract.min_duration_seconds, contract.max_duration_seconds) == (
        1.0,
        6.0,
    )
    assert (
        contract.margin_anchor_min_duration_seconds,
        contract.margin_anchor_max_duration_seconds,
    ) == (2.0, 6.0)
    assert (contract.min_margin, contract.max_margin) == (0.2, 0.5)
    training = config["training"]
    assert training["epochs"] == 10
    assert training["schedule"] == "exponential"
    assert training["warmup_ratio"] == 0.0
    assert training["min_lr_ratio"] == 0.25
    assert training["learning_rate"] == training["encoder_lr"] == 1e-4
    assert training["ema_enabled"] is False
    assert config["model"]["encoder_config"]["campp"] == {
        "freeze_encoder": False,
        "unfreeze_last_n_blocks": 0,
    }


def test_preregistered_profile_locks_competition_guardrails() -> None:
    config = yaml.safe_load(PAPER_PROFILE.read_text(encoding="utf-8"))
    experiment = config["experiment"]
    gate = experiment["preregistered_gate"]

    assert experiment["preregistered_baseline"]["checkpoint_sha256"] == (
        "f50f67f549b913b57111043b43daca1ff8bcbbf49bebe5dccab91ade8b19ae0d"
    )
    assert gate["min_macro_f1_gain"] == 0.002
    assert gate["max_known_accuracy_drop"] == 0.001
    assert gate["max_ood_f1_drop"] == 0.001
    assert gate["require_raw_probability_average"] is True
    assert gate["later_folds_automatic"] is False
    assert gate["leaderboard_tuning"] is False


class _MarginRecordingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.1))
        self.head_ood = torch.nn.Linear(1, 1)
        self.seen_margins = []
        self.seen_lengths = []

    def forward(self, waveforms, labels=None, speaker_margins=None):
        assert labels is not None
        assert speaker_margins is not None
        self.seen_margins.append(speaker_margins.detach().cpu().clone())
        self.seen_lengths.append(int(waveforms.shape[-1]))
        feature = waveforms.mean(dim=(1, 2), keepdim=False).unsqueeze(1)
        ood_logits = self.head_ood(feature)
        signed = feature * self.scale
        speaker_logits = torch.cat([signed, -signed], dim=1)
        return ood_logits, speaker_logits


def test_train_epoch_applies_sampled_crop_and_per_sample_margins() -> None:
    contract = DurationAdaptiveMargin(
        min_duration_seconds=1.0,
        max_duration_seconds=6.0,
        margin_anchor_min_duration_seconds=2.0,
        margin_anchor_max_duration_seconds=6.0,
        min_margin=0.2,
        max_margin=0.5,
        sample_rate=10,
        seed_offset=31,
    )
    waveforms = torch.linspace(-1.0, 1.0, 2 * 80).reshape(2, 1, 1, 80)
    labels = torch.tensor([1, 2])
    source_durations = torch.tensor([0.5, 5.0])
    dataloader = [(waveforms, labels, source_durations)]
    model = _MarginRecordingModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = TwoPartLoss(
        use_focal=False,
        ood_weight=0.15,
        speaker_weight=0.85,
        use_ood=True,
        competition_known_count=2,
        speaker_target_scope="known",
    )
    scaler = torch.amp.GradScaler("cpu", enabled=False)

    metrics = train_epoch(
        model,
        dataloader,
        optimizer,
        criterion,
        scaler,
        torch.device("cpu"),
        autocast_fn=nullcontext,
        adaptive_margin=contract,
        training_seed=42,
        epoch=4,
    )

    sampled = contract.sample_duration_seconds(
        training_seed=42,
        epoch=4,
        step=0,
        window_index=0,
    )
    effective = torch.minimum(source_durations, torch.full_like(source_durations, sampled))
    expected_margins = contract.margin_for_duration(effective)
    assert model.seen_lengths == [round(sampled * contract.sample_rate)]
    assert len(model.seen_margins) == 1
    assert torch.allclose(model.seen_margins[0], expected_margins)
    assert metrics["adaptive_duration_seconds"] == pytest.approx(
        float(effective.mean()), abs=1e-7
    )
    assert metrics["adaptive_margin"] == pytest.approx(
        float(expected_margins.mean()), abs=1e-7
    )
