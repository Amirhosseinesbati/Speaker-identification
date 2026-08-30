from copy import deepcopy
import hashlib
from pathlib import Path

from src.experiment_config import load_profile
from src.pipelines.steps import _training_milestone_epochs
from scripts.audit_campp_paired_consistency_lme20 import (
    assert_paired_single_objective_contract,
)


CONTROL = "p5-campp-known446-ood-crossfile-paired-control-long120-oof-f0"
TREATMENT = "p5-campp-known446-ood-crossfile-consistency-c01-long120-oof-f0"
SAME_CROP_CONTROL = (
    "p4-campp-known446-ood-channelrobust-paired-control-long120-oof-f0"
)
RAW_SHA256 = {
    CONTROL: "ceae8376e4bf6963063295e2e7d0a44a64aa492988fde9caa091989ea464726e",
    TREATMENT: "5243b42eebf82d5f2fb75588ec0040072137a44ee9d811ee50cacff6ac98d5ec",
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


def test_cross_file_pair_differs_only_by_consistency_enabled() -> None:
    control = _normalise_identity(load_profile(CONTROL))
    treatment = _normalise_identity(load_profile(TREATMENT))

    for config in (control, treatment):
        assert config["data"]["known_sampling"] == {"pair_files": True}
        assert config["audio"]["ood_batch_ratio"] == 0.5
        assert config["training"]["epochs"] == 120
        assert config["training"]["milestone_epochs"] == [40, 80]
        assert config["training"]["early_stopping_patience"] == 0
        assert config["training"]["selection_variant"] == "raw"
        assert _training_milestone_epochs(config["training"]) == {40, 80}

    assert control["training"]["loss"]["consistency"] == {
        "enabled": False,
        "type": "cosine",
        "weight": 0.1,
        "pairing": "cross_file_batch",
    }
    assert treatment["training"]["loss"]["consistency"] == {
        "enabled": True,
        "type": "cosine",
        "weight": 0.1,
        "pairing": "cross_file_batch",
    }

    treatment["training"]["loss"]["consistency"]["enabled"] = False
    assert treatment == control


def test_cross_file_control_changes_only_sampler_from_same_crop_control() -> None:
    same_crop = _normalise_identity(load_profile(SAME_CROP_CONTROL))
    cross_file = _normalise_identity(load_profile(CONTROL))

    assert cross_file["data"].pop("known_sampling") == {"pair_files": True}
    assert "known_sampling" not in same_crop["data"]
    assert cross_file["audio"].pop("ood_batch_ratio") == 0.5
    assert same_crop["audio"].pop("ood_batch_ratio") == 0.35
    assert cross_file["training"]["loss"]["consistency"].pop("pairing") == (
        "cross_file_batch"
    )
    assert "pairing" not in same_crop["training"]["loss"]["consistency"]
    assert cross_file == same_crop


def test_cross_file_raw_configs_match_preregistered_hashes() -> None:
    root = Path(__file__).resolve().parents[1]
    for profile, expected in RAW_SHA256.items():
        path = root / "configs" / "experiments" / f"{profile}.yaml"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected


def test_terminal_audit_accepts_the_locked_cross_file_contract() -> None:
    assert_paired_single_objective_contract(
        load_profile(CONTROL),
        load_profile(TREATMENT),
        expected_epochs=120,
        expected_milestones=(40, 80),
        expected_pairing="cross_file_batch",
        expected_ood_batch_ratio=0.5,
    )
