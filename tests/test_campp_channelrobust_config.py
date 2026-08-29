from copy import deepcopy
from pathlib import Path

from src.experiment_config import load_profile


ROOT = Path(__file__).resolve().parents[1]
CONTROL = "p0-campp-known446-ood-control-oof-f0"
CANDIDATE = "p3-campp-known446-ood-channelrobust-oof-f0"
CONTINUATION = "p3-campp-known446-ood-channelrobust-continuation-oof-f0"


def test_channelrobust_changes_only_augmentation_and_output_identity() -> None:
    control = deepcopy(load_profile(CONTROL))
    candidate = deepcopy(load_profile(CANDIDATE))

    assert candidate["data"]["split"] == control["data"]["split"]
    assert candidate["training"]["seed"] == control["training"]["seed"] == 42
    assert candidate["model"] == control["model"]
    assert candidate["training"] == control["training"]
    assert candidate["audio"] == control["audio"]
    assert candidate["augmentation"] != control["augmentation"]

    expected = {
        "noise_p": 0.6,
        "music_p": 0.1,
        "rir_p": 0.6,
        "pitch_p": 0.0,
        "stretch_p": 0.1,
    }
    augmentation = candidate["augmentation"]
    assert {
        "noise_p": augmentation["domain"]["musan"]["noise_p"],
        "music_p": augmentation["domain"]["musan"]["music_p"],
        "rir_p": augmentation["domain"]["rirs_reverb"]["p"],
        "pitch_p": augmentation["waveform"]["pitch_shift"]["p"],
        "stretch_p": augmentation["waveform"]["time_stretch"]["p"],
    } == expected

    for config in (control, candidate):
        config.pop("experiment", None)
        config["logging"]["checkpoint_dir"] = "<profile-checkpoints>"
        config["logging"]["log_dir"] = "<profile-logs>"
    candidate["augmentation"] = control["augmentation"]
    assert candidate == control


def test_channelrobust_continuation_changes_only_resume_identity_and_patience() -> None:
    source = deepcopy(load_profile(CANDIDATE))
    continuation = deepcopy(load_profile(CONTINUATION))

    assert continuation["training"]["resume_checkpoint"].endswith(
        "p3-campp-known446-ood-channelrobust-oof-f0/campp_best_raw.pt"
    )
    assert continuation["training"]["resume_history_path"].endswith(
        "p3-campp-known446-ood-channelrobust-oof-f0/campp_latest.pt"
    )
    assert continuation["training"]["early_stopping_patience"] == 12
    assert continuation["training"]["epochs"] == source["training"]["epochs"] == 200

    for config in (source, continuation):
        config.pop("experiment", None)
        config["logging"]["checkpoint_dir"] = "<profile-checkpoints>"
        config["logging"]["log_dir"] = "<profile-logs>"
    continuation["training"].pop("resume_checkpoint")
    continuation["training"].pop("resume_history_path")
    continuation["training"]["early_stopping_patience"] = source["training"][
        "early_stopping_patience"
    ]
    assert continuation == source
