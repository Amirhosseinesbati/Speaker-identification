from scripts.verify_oof_experiments import FAMILIES, _family_invariant
from src.experiment_config import load_profile


def test_reproduction_families_are_pinned_and_inter_class_is_disabled():
    for family in FAMILIES.values():
        resolved = [load_profile(profile) for profile in family["profiles"]]
        assert {_family_invariant(config) for config in resolved} == {
            family["invariant_sha256"]
        }
        assert all(
            config["training"]["loss"]["speaker"]["inter_class"]["enabled"]
            is False
            for config in resolved
        )
        if family is FAMILIES["no_proto"]:
            assert all(
                config["training"]["early_stopping_start_epoch"]
                == config["training"]["freeze_epochs"] + 1
                == 21
                for config in resolved
            )
