"""Compare matched CAM++ milestone audits without making a decision.

The input reports must describe the same epoch, have contiguous histories and
carry the diagnostic-only role emitted by ``audit_campp_milestone.py``.  This
tool performs deterministic arithmetic only; it cannot select a checkpoint,
stop a run, pass a terminal gate or authorize another Fold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DIAGNOSTIC_ROLE = "diagnostic_only_cannot_select_checkpoint_or_authorize_run"
METRIC_KEYS = (
    "val_macro_f1",
    "val_logit_avg_macro_f1",
    "val_ema_macro_f1",
    "val_known_acc",
    "val_ood_f1",
    "train_loss",
    "val_loss",
    "train_loss_consistency",
    "train_pair_cosine",
    "train_embedding_std_clean",
    "train_embedding_std_augmented",
)
TRAJECTORY_KEYS = (
    "val_macro_f1",
    "val_logit_avg_macro_f1",
    "val_ema_macro_f1",
    "val_known_acc",
    "val_ood_f1",
    "train_loss",
    "val_loss",
)


def _load(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("decision_role") != DIAGNOSTIC_ROLE:
        raise ValueError(f"Not a diagnostic-only milestone report: {path}")
    diagnostic = report.get("diagnostic")
    if not isinstance(diagnostic, dict):
        raise ValueError(f"Missing milestone diagnostic: {path}")
    if diagnostic.get("history_contiguous") is not True:
        raise ValueError(f"Milestone history is not contiguous: {path}")
    return report


def _numeric_delta(
    treatment: dict[str, Any], control: dict[str, Any], keys: tuple[str, ...]
) -> dict[str, float]:
    delta: dict[str, float] = {}
    for key in keys:
        if key not in treatment or key not in control:
            continue
        delta[key] = float(treatment[key]) - float(control[key])
    return delta


def compare_reports(
    treatment_report: dict[str, Any], control_report: dict[str, Any]
) -> dict[str, Any]:
    treatment = treatment_report["diagnostic"]
    control = control_report["diagnostic"]
    treatment_epoch = int(treatment.get("epoch", -1))
    control_epoch = int(control.get("epoch", -1))
    if treatment_epoch <= 0 or treatment_epoch != control_epoch:
        raise ValueError(
            "Treatment and control milestones must have the same positive epoch"
        )

    treatment_trajectory = treatment.get("trajectory") or {}
    control_trajectory = control.get("trajectory") or {}
    if treatment_trajectory.get("tail_window") != control_trajectory.get(
        "tail_window"
    ):
        raise ValueError("Treatment and control tail windows do not match")
    if treatment_trajectory.get("previous_window") != control_trajectory.get(
        "previous_window"
    ):
        raise ValueError("Treatment and control previous windows do not match")

    return {
        "decision_role": (
            "paired_milestone_diagnostic_only_cannot_stop_or_authorize_run"
        ),
        "epoch": treatment_epoch,
        "treatment": {
            "path": treatment.get("path"),
            "sha256": treatment.get("sha256"),
        },
        "control": {
            "path": control.get("path"),
            "sha256": control.get("sha256"),
        },
        "same_epoch_treatment_minus_control": _numeric_delta(
            treatment.get("metrics") or {},
            control.get("metrics") or {},
            METRIC_KEYS,
        ),
        "tail_mean_treatment_minus_control": _numeric_delta(
            treatment_trajectory.get("tail_means") or {},
            control_trajectory.get("tail_means") or {},
            TRAJECTORY_KEYS,
        ),
        "slope_last20_treatment_minus_control": _numeric_delta(
            treatment_trajectory.get("slopes_last_20") or {},
            control_trajectory.get("slopes_last_20") or {},
            TRAJECTORY_KEYS,
        ),
        "best_raw_within_horizon": {
            "treatment_epoch": int(treatment_trajectory["best_raw_epoch"]),
            "treatment_macro_f1": float(
                treatment_trajectory["best_raw_macro_f1"]
            ),
            "control_epoch": int(control_trajectory["best_raw_epoch"]),
            "control_macro_f1": float(control_trajectory["best_raw_macro_f1"]),
            "macro_f1_delta": float(
                treatment_trajectory["best_raw_macro_f1"]
            )
            - float(control_trajectory["best_raw_macro_f1"]),
        },
        "warning": (
            "Milestone evidence is descriptive. Only the preregistered terminal "
            "paired LME20 gate may accept or reject P5."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    comparison = compare_reports(_load(args.treatment), _load(args.control))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "epoch": comparison["epoch"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
