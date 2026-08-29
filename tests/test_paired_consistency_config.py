from copy import deepcopy

from src.experiment_config import load_profile


CONTROL = "p4-campp-known446-ood-channelrobust-paired-control-oof-f0"
CANDIDATE = "p4-campp-known446-ood-channelrobust-consistency-c01-oof-f0"


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
