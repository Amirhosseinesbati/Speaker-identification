import numpy as np

from scripts.audit_campp_channelrobust_lme20 import (
    acceptance_gate,
    align_oof,
)


def test_align_oof_uses_control_order_and_checks_labels() -> None:
    reference = {
        "files": np.array(["a", "b", "c"]),
        "labels": np.array([1, 0, 2]),
    }
    candidate = {
        "files": np.array(["c", "a", "b"]),
        "labels": np.array([2, 1, 0]),
        "competition_probs": np.eye(3, 447, dtype=np.float32),
        "embeddings": np.arange(12, dtype=np.float32).reshape(3, 4),
        "split_fold": np.array([0]),
    }

    aligned = align_oof(reference, candidate)

    np.testing.assert_array_equal(aligned["files"], reference["files"])
    np.testing.assert_array_equal(aligned["labels"], reference["labels"])
    np.testing.assert_array_equal(
        aligned["embeddings"], candidate["embeddings"][[1, 2, 0]]
    )


def test_acceptance_gate_requires_every_preregistered_condition() -> None:
    standalone = {
        "macro_f1": -0.005,
        "accuracy": 0.0,
        "known_accuracy": 0.0,
        "ood_f1": 0.0,
    }
    fusion = {
        "macro_f1": 0.0021,
        "accuracy": 0.001,
        "known_accuracy": -0.0009,
        "ood_f1": -0.0009,
    }
    assert acceptance_gate(standalone, fusion, 0.20)["passed"]

    for key, value in (
        ("macro_f1", 0.0019),
        ("known_accuracy", -0.0011),
        ("ood_f1", -0.0011),
    ):
        failed = dict(fusion)
        failed[key] = value
        assert not acceptance_gate(standalone, failed, 0.20)["passed"]
    assert not acceptance_gate(standalone, fusion, 0.199)["passed"]
    too_weak = dict(standalone)
    too_weak["macro_f1"] = -0.0101
    assert not acceptance_gate(too_weak, fusion, 0.20)["passed"]

