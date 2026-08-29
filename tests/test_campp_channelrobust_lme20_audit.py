import json

import numpy as np
import pytest
import torch

from scripts.audit_campp_channelrobust_lme20 import (
    acceptance_gate,
    align_oof,
    assert_augmentation_only_contract,
    validate_raw_bundle_binding,
)
from scripts.analyze_control_oof_centroid_crossfit import sha256_file


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


def test_align_oof_rejects_label_drift_after_filename_alignment() -> None:
    reference = {
        "files": np.array(["a", "b"]),
        "labels": np.array([1, 0]),
    }
    candidate = {
        "files": np.array(["b", "a"]),
        "labels": np.array([0, 2]),
        "competition_probs": np.zeros((2, 447), dtype=np.float32),
        "embeddings": np.zeros((2, 4), dtype=np.float32),
    }

    with pytest.raises(RuntimeError, match="labels differ"):
        align_oof(reference, candidate)


@pytest.mark.parametrize("key", ["competition_probs", "embeddings"])
def test_align_oof_rejects_nonfinite_candidate_evidence(key: str) -> None:
    reference = {
        "files": np.array(["a", "b"]),
        "labels": np.array([1, 0]),
    }
    candidate = {
        "files": np.array(["a", "b"]),
        "labels": np.array([1, 0]),
        "competition_probs": np.zeros((2, 447), dtype=np.float32),
        "embeddings": np.zeros((2, 4), dtype=np.float32),
    }
    candidate[key][0, 0] = np.nan

    with pytest.raises(RuntimeError, match="non-finite"):
        align_oof(reference, candidate)


@pytest.mark.parametrize(
    ("probabilities", "message"),
    [
        (np.array([[1.1, -0.1], [0.0, 1.0]]), "outside"),
        (np.array([[0.4, 0.4], [0.0, 1.0]]), "sum to one"),
    ],
)
def test_align_oof_rejects_invalid_probability_simplex(
    probabilities: np.ndarray, message: str
) -> None:
    reference = {
        "files": np.array(["a", "b"]),
        "labels": np.array([1, 0]),
    }
    padded = np.zeros((2, 447), dtype=np.float32)
    padded[:, :2] = probabilities
    candidate = {
        "files": np.array(["a", "b"]),
        "labels": np.array([1, 0]),
        "competition_probs": padded,
        "embeddings": np.zeros((2, 4), dtype=np.float32),
    }
    with pytest.raises(RuntimeError, match=message):
        align_oof(reference, candidate)


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


def test_augmentation_only_contract_rejects_any_second_treatment() -> None:
    control = {
        "model": {"encoder_type": "campp"},
        "data": {"split": {"fold": 0, "folds": 3, "seed": 42}},
        "augmentation": {"noise_p": 0.2},
        "training": {"learning_rate": 3e-4},
        "logging": {"checkpoint_dir": "control"},
        "experiment": {"purpose": "control"},
    }
    candidate = {
        **control,
        "augmentation": {"noise_p": 0.6},
        "logging": {"checkpoint_dir": "candidate"},
        "experiment": {"purpose": "augmentation ablation"},
    }
    assert_augmentation_only_contract(control, candidate)

    changed_lr = {
        **candidate,
        "training": {"learning_rate": 1e-4},
    }
    with pytest.raises(RuntimeError, match="outside augmentation: training"):
        assert_augmentation_only_contract(control, changed_lr)


def _write_bound_bundle(tmp_path, *, diverge_raw: bool = False):
    candidate_dir = tmp_path / "candidate"
    bundle_dir = candidate_dir / "campp_best_bundle"
    bundle_dir.mkdir(parents=True)
    selected_path = candidate_dir / "campp_best.pt"
    raw_path = candidate_dir / "campp_best_raw.pt"
    oof_path = bundle_dir / "oof_predictions.npz"
    state = {"weight": torch.tensor([1.0, 2.0])}
    payload = {
        "epoch": 7,
        "val_macro_f1": 0.93,
        "weight_variant": "raw",
        "model_state_dict": state,
    }
    torch.save(payload, selected_path)
    raw_payload = dict(payload)
    raw_payload["model_state_dict"] = {
        "weight": torch.tensor([1.0, 3.0] if diverge_raw else [1.0, 2.0])
    }
    torch.save(raw_payload, raw_path)
    np.savez(oof_path, files=np.asarray(["a.wav"]))
    manifest = {
        "checkpoint": str(selected_path),
        "checkpoint_sha256": sha256_file(selected_path),
        "oof_predictions_sha256": sha256_file(oof_path),
    }
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return candidate_dir, raw_path, oof_path


def test_raw_bundle_binding_verifies_manifest_and_selected_weights(tmp_path) -> None:
    candidate_dir, raw_path, oof_path = _write_bound_bundle(tmp_path)
    result = validate_raw_bundle_binding(candidate_dir, raw_path, oof_path)
    assert result["selected_epoch"] == 7
    assert result["weight_variant"] == "raw"

    with oof_path.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(RuntimeError, match="OOF manifest SHA mismatch"):
        validate_raw_bundle_binding(candidate_dir, raw_path, oof_path)


def test_raw_bundle_binding_rejects_selected_raw_model_drift(tmp_path) -> None:
    candidate_dir, raw_path, oof_path = _write_bound_bundle(
        tmp_path, diverge_raw=True
    )
    with pytest.raises(RuntimeError, match="model states differ"):
        validate_raw_bundle_binding(candidate_dir, raw_path, oof_path)
