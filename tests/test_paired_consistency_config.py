from copy import deepcopy

from src.experiment_config import load_profile
from src.pipelines.steps import _training_milestone_epochs


CONTROL = "p4-campp-known446-ood-channelrobust-paired-control-oof-f0"
CANDIDATE = "p4-campp-known446-ood-channelrobust-consistency-c01-oof-f0"
LONG_CONTROL = "p4-campp-known446-ood-channelrobust-paired-control-long80-oof-f0"
LONG_CANDIDATE = "p4-campp-known446-ood-channelrobust-consistency-c01-long80-oof-f0"


def _normalise_identity(config: dict) -> dict:
    config = deepcopy(config)
    config.pop("experiment", None)
    config["logging"]["checkpoint_dir"] = "<profile-checkpoints>"
    config["logging"]["log_dir"] = "<profile-logs>"
    config["hardware"]["profiles"]["vastai_3090_campp"][
        "description"
    ] = "<profile-description>"
    return config


def test_paired_consistency_is_a_single_objective_change() -> None:
    control = _normalise_identity(load_profile(CONTROL))
    candidate = _normalise_identity(load_profile(CANDIDATE))

    assert control["training"]["early_stopping_patience"] == 0
    assert candidate["training"]["early_stopping_patience"] == 0
    assert control["training"]["epochs"] == candidate["training"]["epochs"] == 40
    assert control["training"]["warm_start_checkpoint"] == candidate["training"][
        "warm_start_checkpoint"
    ]
    assert control["training"]["loss"]["consistency"] == {
        "enabled": False,
        "type": "cosine",
        "weight": 0.1,
    }
    assert candidate["training"]["loss"]["consistency"] == {
        "enabled": True,
        "type": "cosine",
        "weight": 0.1,
    }

    candidate["training"]["loss"]["consistency"]["enabled"] = False
    assert candidate == control


def test_long80_pair_is_a_single_objective_change() -> None:
    control = _normalise_identity(load_profile(LONG_CONTROL))
    candidate = _normalise_identity(load_profile(LONG_CANDIDATE))

    for config in (control, candidate):
        assert config["training"]["epochs"] == 80
        assert config["training"]["milestone_epochs"] == [40]
        assert config["training"]["early_stopping_patience"] == 0
        assert _training_milestone_epochs(config["training"]) == {40}

    candidate["training"]["loss"]["consistency"]["enabled"] = False
    assert candidate == control


def test_long80_changes_only_horizon_and_identity_from_40_epoch_recipe() -> None:
    short = _normalise_identity(load_profile(CONTROL))
    long = _normalise_identity(load_profile(LONG_CONTROL))

    assert short["training"]["epochs"] == 40
    assert "milestone_epochs" not in short["training"]
    assert long["training"].pop("epochs") == 80
    assert long["training"].pop("milestone_epochs") == [40]
    short["training"].pop("epochs")
    assert long == short


def test_milestone_validation_rejects_terminal_or_nonpositive_epochs() -> None:
    assert _training_milestone_epochs({"epochs": 80, "milestone_epochs": [40]}) == {40}

    for invalid in ([0], [-1], [80], [81]):
        try:
            _training_milestone_epochs({"epochs": 80, "milestone_epochs": invalid})
        except ValueError as exc:
            assert "strictly below" in str(exc)
        else:
            raise AssertionError(f"invalid milestone accepted: {invalid}")
