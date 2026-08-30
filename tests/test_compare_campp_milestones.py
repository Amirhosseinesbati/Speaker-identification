from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.compare_campp_milestones import (
    DIAGNOSTIC_ROLE,
    _load,
    compare_reports,
)


def report(epoch: int, raw: float, *, contiguous: bool = True) -> dict:
    metrics = {
        "val_macro_f1": raw,
        "val_logit_avg_macro_f1": raw - 0.001,
        "val_ema_macro_f1": raw - 0.002,
        "val_known_acc": raw + 0.01,
        "val_ood_f1": raw + 0.02,
        "train_loss": 1.0,
        "val_loss": 1.1,
    }
    trajectory = {
        "previous_window": [21, 30],
        "tail_window": [31, 40],
        "tail_means": metrics,
        "slopes_last_20": {key: 0.001 for key in metrics},
        "best_raw_epoch": epoch - 2,
        "best_raw_macro_f1": raw + 0.003,
    }
    return {
        "decision_role": DIAGNOSTIC_ROLE,
        "diagnostic": {
            "path": f"checkpoint-{raw}.pt",
            "sha256": str(raw),
            "epoch": epoch,
            "history_contiguous": contiguous,
            "metrics": metrics,
            "trajectory": trajectory,
        },
    }


def test_compare_reports_computes_paired_deltas() -> None:
    comparison = compare_reports(report(40, 0.94), report(40, 0.93))
    assert comparison["epoch"] == 40
    assert comparison["same_epoch_treatment_minus_control"][
        "val_macro_f1"
    ] == pytest.approx(0.01)
    assert comparison["tail_mean_treatment_minus_control"][
        "val_known_acc"
    ] == pytest.approx(0.01)
    assert comparison["slope_last20_treatment_minus_control"][
        "val_macro_f1"
    ] == pytest.approx(0.0)
    assert comparison["best_raw_within_horizon"][
        "macro_f1_delta"
    ] == pytest.approx(0.01)
    assert "diagnostic" in comparison["decision_role"]


def test_compare_reports_rejects_mismatched_epochs() -> None:
    with pytest.raises(ValueError, match="same positive epoch"):
        compare_reports(report(40, 0.94), report(80, 0.93))


def test_load_rejects_noncontiguous_history(tmp_path: Path) -> None:
    path = tmp_path / "milestone.json"
    path.write_text(json.dumps(report(40, 0.94, contiguous=False)))
    with pytest.raises(ValueError, match="not contiguous"):
        _load(path)
