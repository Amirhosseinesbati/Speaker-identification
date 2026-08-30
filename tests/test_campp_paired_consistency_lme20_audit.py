from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from scripts.audit_campp_paired_consistency_lme20 import (
    HORIZON_SPECS,
    LONG120_MATCHED_CONFIG_SHA256,
    LONG120_MATCHED_CONTROL_PROFILE,
    LONG120_TREATMENT_CONFIG_SHA256,
    LONG120_TREATMENT_PROFILE,
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
    paired_randomization_diagnostic,
    sha256_file,
    terminal_curve_diagnostic,
)
from scripts.audit_campp_milestone import main as milestone_main
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


def test_long120_contract_and_raw_config_hashes_are_preregistered() -> None:
    spec = HORIZON_SPECS[120]
    assert spec["matched_profile"] == LONG120_MATCHED_CONTROL_PROFILE
    assert spec["treatment_profile"] == LONG120_TREATMENT_PROFILE
    expected = {
        LONG120_MATCHED_CONTROL_PROFILE: LONG120_MATCHED_CONFIG_SHA256,
        LONG120_TREATMENT_PROFILE: LONG120_TREATMENT_CONFIG_SHA256,
    }
    for profile, digest in expected.items():
        path = ROOT / "configs" / "experiments" / f"{profile}.yaml"
        assert sha256_file(path) == digest

    matched = load_profile(LONG120_MATCHED_CONTROL_PROFILE)
    treatment = load_profile(LONG120_TREATMENT_PROFILE)
    assert_paired_single_objective_contract(
        matched,
        treatment,
        expected_epochs=120,
        expected_milestones=(40, 80),
    )


def test_embedding_spread_is_finite_and_detects_collapse() -> None:
    artifact = {
        "train_embeddings": np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32
        )
    }
    assert embedding_spread(artifact) > 0.0

    with pytest.raises(RuntimeError, match="collapsed"):
        embedding_spread({"train_embeddings": np.ones((3, 2), np.float32)})


def test_paired_randomization_is_deterministic_and_tracks_primary_delta() -> None:
    labels = np.tile(np.arange(7, dtype=np.int64), 8)
    baseline = (labels + 1) % 7
    candidate = labels.copy()
    first = paired_randomization_diagnostic(
        labels, baseline, candidate, replicates=512, seed=123
    )
    second = paired_randomization_diagnostic(
        labels, baseline, candidate, replicates=512, seed=123
    )

    assert first == second
    assert first["observed_delta"] == pytest.approx(7 / 447)
    assert first["candidate_only_correct"] == len(labels)
    assert first["baseline_only_correct"] == 0
    assert first["one_sided_improvement_p_value"] < 0.01
    assert first["decision_role"] == "descriptive_only_cannot_override_locked_gate"


def test_paired_randomization_rejects_unaligned_inputs() -> None:
    with pytest.raises(ValueError, match="aligned"):
        paired_randomization_diagnostic(
            np.asarray([0, 1]),
            np.asarray([0]),
            np.asarray([0, 1]),
            replicates=10,
        )


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
    history = []
    for epoch in range(1, 41):
        history.append({
            "epoch": epoch,
            "val_macro_f1": 0.90 + epoch * 0.001,
            "val_logit_avg_macro_f1": 0.89 + epoch * 0.001,
            "val_known_acc": 0.91 + epoch * 0.001,
            "val_ood_f1": 0.92 + epoch * 0.001,
            "val_ema_macro_f1": 0.90 + epoch * 0.001,
            "train_loss": 2.0 - epoch * 0.01,
            "val_loss": 1.5 - epoch * 0.005,
            "train_loss_consistency": 0.2,
            "train_loss_consistency_weighted": 0.02,
            "train_pair_cosine": 0.8,
            "train_embedding_std_augmented": 0.1,
            "train_embedding_std_clean": 0.1,
        })
    torch.save({
        "epoch": 40,
        "config": {"logging": {"checkpoint_dir": f"checkpoints/{TREATMENT_PROFILE}"}},
        "training_history": history,
    }, path)
    result = milestone_diagnostic(path, TREATMENT_PROFILE)
    assert result["epoch"] == 40
    assert result["metrics"]["train_loss_consistency_weighted"] == pytest.approx(0.02)
    assert result["history_length"] == 40
    assert result["trajectory"]["best_raw_epoch"] == 40
    assert result["trajectory"]["tail_minus_previous"]["val_macro_f1"] > 0
    assert result["trajectory"]["slopes_last_20"]["val_macro_f1"] > 0

    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["training_history"][-1]["val_macro_f1"] = float("nan")
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="non-finite"):
        milestone_diagnostic(path, TREATMENT_PROFILE)


def test_milestone_diagnostic_rejects_history_gap(tmp_path) -> None:
    path = tmp_path / "campp_milestone_epoch040_raw.pt"
    history = _terminal_history(treatment=False)[:40]
    history.pop(19)
    torch.save({
        "epoch": 40,
        "config": {"logging": {"checkpoint_dir": f"checkpoints/{TREATMENT_PROFILE}"}},
        "training_history": history,
    }, path)
    with pytest.raises(RuntimeError, match="not contiguous"):
        milestone_diagnostic(path, TREATMENT_PROFILE)


def test_milestone_cli_writes_diagnostic_only_receipt(tmp_path) -> None:
    checkpoint = tmp_path / "campp_milestone_epoch040_raw.pt"
    history = _terminal_history(treatment=False)[:40]
    for row in history:
        row.update({
            "val_logit_avg_macro_f1": row["val_macro_f1"] - 0.001,
            "val_ema_macro_f1": row["val_macro_f1"] - 0.002,
            "train_loss": 1.0,
            "val_loss": 1.1,
            "train_loss_consistency": 0.0,
            "train_loss_consistency_weighted": 0.0,
            "train_pair_cosine": 0.0,
            "train_embedding_std_augmented": 0.0,
            "train_embedding_std_clean": 0.0,
        })
    torch.save({
        "epoch": 40,
        "config": {"logging": {"checkpoint_dir": f"checkpoints/{TREATMENT_PROFILE}"}},
        "training_history": history,
    }, checkpoint)
    output = tmp_path / "milestone_receipt.json"
    assert milestone_main([
        "--checkpoint", str(checkpoint),
        "--profile", TREATMENT_PROFILE,
        "--epoch", "40",
        "--output", str(output),
    ]) == 0
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["decision_role"].startswith("diagnostic_only")
    assert receipt["diagnostic"]["history_length"] == 40
    assert receipt["diagnostic"]["sha256"]


def _terminal_history(*, treatment: bool, horizon: int = 80) -> list[dict]:
    rows = []
    for epoch in range(1, horizon + 1):
        macro = 0.94
        if treatment:
            macro = 0.930 + max(0, epoch - (horizon - 20)) * 0.00012
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


def test_long120_terminal_curve_uses_epochs_101_to_120(tmp_path) -> None:
    curves = {}
    for profile, treatment in (
        (LONG120_MATCHED_CONTROL_PROFILE, False),
        (LONG120_TREATMENT_PROFILE, True),
    ):
        path = tmp_path / f"{profile}.pt"
        torch.save({
            "epoch": 120,
            "config": {"logging": {"checkpoint_dir": f"checkpoints/{profile}"}},
            "training_history": _terminal_history(
                treatment=treatment, horizon=120,
            ),
        }, path)
        curves[profile] = terminal_curve_diagnostic(
            path, profile, expected_epoch=120,
        )

    treatment_curve = curves[LONG120_TREATMENT_PROFILE]
    assert treatment_curve["previous_window"] == [101, 110]
    assert treatment_curve["tail_window"] == [111, 120]
    assert treatment_curve["best_raw_epoch"] == 120
    diagnostic = matched_extension_diagnostic(
        curves[LONG120_MATCHED_CONTROL_PROFILE],
        treatment_curve,
        spread_ratio=0.98,
    )
    assert diagnostic["eligible_for_separate_matched_extension"] is True


def test_terminal_curve_rejects_incomplete_history(tmp_path) -> None:
    path = tmp_path / "latest.pt"
    torch.save({
        "epoch": 80,
        "config": {"logging": {"checkpoint_dir": f"checkpoints/{TREATMENT_PROFILE}"}},
        "training_history": _terminal_history(treatment=True)[:-1],
    }, path)
    with pytest.raises(RuntimeError, match="not contiguous"):
        terminal_curve_diagnostic(path, TREATMENT_PROFILE)
