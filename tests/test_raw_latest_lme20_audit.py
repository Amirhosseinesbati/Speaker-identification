from pathlib import Path

from scripts.audit_raw_latest_lme20 import (
    SNAPSHOT_VARIANT,
    SNAPSHOT_WEIGHT,
    latest_checkpoint_path,
)


def test_latest_snapshot_contract_is_fixed() -> None:
    assert SNAPSHOT_VARIANT == "latest"
    assert SNAPSHOT_WEIGHT == 0.5


def test_latest_checkpoint_path_is_fold_specific() -> None:
    root = Path("checkpoints")
    assert latest_checkpoint_path(root, 2) == (
        root
        / "p0-campp-known446-ood-control-oof-f2"
        / "campp_latest.pt"
    )
