from copy import deepcopy

import numpy as np
import pytest
import torch

from scripts.audit_campp_paired_consistency_lme20 import (
    MATCHED_CONFIG_SHA256,
    MATCHED_CONTROL_PROFILE,
    ROOT,
    TREATMENT_CONFIG_SHA256,
    TREATMENT_PROFILE,
    acceptance_gate,
    assert_paired_single_objective_contract,
    embedding_spread,
    matched_extension_diagnostic,
    milestone_diagnostic,
    sha256_file,
    terminal_curve_diagnostic,
)
from src.experiment_config import load_profile


def test_long80_contract_changes_only_consistency_enabled() -> None:
    matched = load_profile(MATCHED_CONTROL_PROFILE)
    treatment = load_profile(TREATMENT_PROFILE)
    assert_paired_single_objective_contract(matched, treatment)

    changed = deepcopy(treatment)
    changed["training"]["encoder_lr"] *= 2
    with pytest.raises(RuntimeError, match="outside consistency"):
        assert_paired_single_objective_contract(matched, changed)


def test_long80_raw_config_files_match_preregistered_hashes() -> None:
    expected = {
        MATCHED_CONTROL_PROFILE: MATCHED_CONFIG_SHA256,
        TREATMENT_PROFILE: TREATMENT_CONFIG_SHA256,
    }
    for profile, digest in expected.items():
        path = ROOT / "configs" / "experiments" / f"{profile}.yaml"
        assert sha256_file(path) == digest


def test_embedding_spread_is_finite_and_detects_collapse() -> None:
    artifact = {
        "train_embeddings": np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32
        )
    }
    assert embedding_spread(artifact) > 0.0

    with pytest.raises(RuntimeError, match="collapsed"):
        embedding_spread({"train_embeddings": np.ones((3, 2), np.float32)})


def test_acceptance_gate_requires_every_preregistered_guardrail() -> None:
    passing = acceptance_gate(
        matched_delta={"macro_f1": 0.002},
        fusion_delta={
            "macro_f1": 0.002,
            "known_accuracy": -0.001,
            "ood_f1": -0.001,
        },
        rescue_rate=0.20,
        spread_ratio=0.95,
    )
    assert passing["passed"] is True

    cases = (
        {"matched_delta": {"macro_f1": 0.0019}},
        {"fusion_delta": {
            "macro_f1": 0.0019,
            "known_accuracy": -0.001,
            "ood_f1": -0.001,
        }},
        {"fusion_delta": {
            "macro_f1": 0.002,
            "known_accuracy": -0.0011,
            "ood_f1": -0.001,
        }},
        {"fusion_delta": {
            "macro_f1": 0.002,
            "known_accuracy": -0.001,
            "ood_f1": -0.0011,
        }},
        {"rescue_rate": 0.199},
        {"spread_ratio": 0.949},
    )
    defaults = {
        "matched_delta": {"macro_f1": 0.002},
        "fusion_delta": {
            "macro_f1": 0.002,
            "known_accuracy": -0.001,
            "ood_f1": -0.001,
        },
        "rescue_rate": 0.20,
        "spread_ratio": 0.95,
    }
    for override in cases:
        arguments = {**defaults, **override}
        assert acceptance_gate(**arguments)["passed"] is False


def test_milestone_diagnostic_binds_epoch_profile_and_finite_metrics(tmp_path) -> None:
    path = tmp_path / "campp_milestone_epoch040_raw.pt"
    torch.save({
        "epoch": 40,
        "config": {"logging": {"checkpoint_dir": f"checkpoints/{TREATMENT_PROFILE}"}},
        "training_history": [{
            "epoch": 40,
            "val_macro_f1": 0.94,
            "val_logit_avg_macro_f1": 0.93,
            "val_known_acc": 0.95,
            "val_ood_f1": 0.96,
            "val_ema_macro_f1": 0.94,
            "train_loss": 1.0,
            "train_loss_consistency": 0.2,
            "train_loss_consistency_weighted": 0.02,
            "train_pair_cosine": 0.8,
            "train_embedding_std_augmented": 0.1,
            "train_embedding_std_clean": 0.1,
        }],
    }, path)
    result = milestone_diagnostic(path, TREATMENT_PROFILE)
    assert result["epoch"] == 40
    assert result["metrics"]["train_loss_consistency_weighted"] == pytest.approx(0.02)

    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["training_history"][-1]["val_macro_f1"] = float("nan")
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="non-finite"):
        milestone_diagnostic(path, TREATMENT_PROFILE)


def _terminal_history(*, treatment: bool) -> list[dict]:
    rows = []
    for epoch in range(1, 81):
        macro = 0.94
        if treatment:
            macro = 0.930 + max(0, epoch - 60) * 0.00012
        rows.append({
            "epoch": epoch,
            "val_macro_f1": macro,
            "val_known_acc": 0.95,
            "val_ood_f1": 0.96,
        })
    return rows


def test_terminal_curve_and_matched_extension_gate_are_predeclared(tmp_path) -> None:
    curves = {}
    for profile, treatment in (
        (MATCHED_CONTROL_PROFILE, False),
        (TREATMENT_PROFILE, True),
    ):
        path = tmp_path / f"{profile}.pt"
        torch.save({
            "epoch": 80,
            "config": {"logging": {"checkpoint_dir": f"checkpoints/{profile}"}},
            "training_history": _terminal_history(treatment=treatment),
        }, path)
        curves[profile] = terminal_curve_diagnostic(path, profile)

    diagnostic = matched_extension_diagnostic(
        curves[MATCHED_CONTROL_PROFILE],
        curves[TREATMENT_PROFILE],
        spread_ratio=0.98,
    )
    assert diagnostic["eligible_for_separate_matched_extension"] is True
    assert diagnostic["checks"]["treatment_best_in_final_window"] is True
    assert diagnostic["relative_gap_gain"] >= 0.0005

    collapsed = matched_extension_diagnostic(
        curves[MATCHED_CONTROL_PROFILE],
        curves[TREATMENT_PROFILE],
        spread_ratio=0.90,
    )
    assert collapsed["eligible_for_separate_matched_extension"] is False


def test_terminal_curve_rejects_incomplete_history(tmp_path) -> None:
    path = tmp_path / "latest.pt"
    torch.save({
        "epoch": 80,
        "config": {"logging": {"checkpoint_dir": f"checkpoints/{TREATMENT_PROFILE}"}},
        "training_history": _terminal_history(treatment=True)[:-1],
    }, path)
    with pytest.raises(RuntimeError, match="not contiguous"):
        terminal_curve_diagnostic(path, TREATMENT_PROFILE)
