"""Project paired-consistency wall time from matched worker measurements.

The projection deliberately combines two kinds of evidence collected on the
same worker:

* complete-epoch timings parsed from the terminal matched-control log; and
* fixed-batch throughput from control and treatment GPU probes.

Validation and checkpoint overhead are charged at the observed control rate.
Only the training component is scaled by the measured treatment/control
throughput ratio.  The result is an operational launch gate, not a scientific
result and not permission to change the preregistered recipe.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_STAGE_COMPLETE = re.compile(
    r"\b(?P<stage>Train|Val):\s+100%\|[^\n]*?"
    r"\[(?P<elapsed>(?:\d+:){1,2}\d+)<"
)
_EPOCH_COMPLETE = re.compile(r"\bEpoch\s+(?P<epoch>\d+)/\d+\s+[—-]")


def _duration_seconds(value: str) -> int:
    parts = [int(part) for part in value.split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return 60 * minutes + seconds
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return 3600 * hours + 60 * minutes + seconds
    raise ValueError(f"Unsupported duration: {value!r}")


def parse_complete_epoch_stages(log_text: str) -> list[dict[str, float]]:
    """Return Train/Raw-Val/EMA-Val durations for complete epochs only."""
    pending: list[tuple[str, int]] = []
    epochs: list[dict[str, float]] = []
    for line in log_text.replace("\r", "\n").splitlines():
        stage_match = _STAGE_COMPLETE.search(line)
        if stage_match:
            pending.append((
                stage_match.group("stage"),
                _duration_seconds(stage_match.group("elapsed")),
            ))
            continue
        epoch_match = _EPOCH_COMPLETE.search(line)
        if not epoch_match:
            continue
        if len(pending) < 3 or [item[0] for item in pending[-3:]] != [
            "Train", "Val", "Val",
        ]:
            raise ValueError(
                "Complete epoch lacks the expected Train/Val/Val timing sequence: "
                f"epoch={epoch_match.group('epoch')} pending={pending[-5:]}"
            )
        stages = pending[-3:]
        epochs.append({
            "epoch": int(epoch_match.group("epoch")),
            "train_seconds": float(stages[0][1]),
            "raw_val_seconds": float(stages[1][1]),
            "ema_val_seconds": float(stages[2][1]),
        })
        pending.clear()
    if not epochs:
        raise ValueError("No complete Train/Val/Val epochs found in control log")
    expected = list(range(1, len(epochs) + 1))
    actual = [int(row["epoch"]) for row in epochs]
    if actual != expected:
        raise ValueError(f"Control epoch sequence is not contiguous: {actual}")
    return epochs


def probe_files_per_second(report: dict[str, Any], batch_size: int) -> float:
    matches = [
        row for row in report.get("results", [])
        if row.get("status") == "ok"
        and int(row.get("batch_size", -1)) == int(batch_size)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one successful batch-{batch_size} probe row, got "
            f"{len(matches)}"
        )
    value = float(matches[0]["files_per_second"])
    if value <= 0:
        raise ValueError("Probe files_per_second must be positive")
    return value


def project_paired_runtime(
    *,
    epoch_stages: list[dict[str, float]],
    control_wall_seconds: float,
    control_files_per_second: float,
    treatment_files_per_second: float,
    treatment_epochs: int,
    timeout_hours: float,
    required_headroom_fraction: float,
    dph_total: float | None = None,
    max_incremental_cost_usd: float | None = None,
) -> dict[str, Any]:
    """Conservatively project treatment wall time and evaluate launch gates."""
    if control_wall_seconds <= 0 or treatment_epochs <= 0 or timeout_hours <= 0:
        raise ValueError("Wall time, epoch count and timeout must be positive")
    if control_files_per_second <= 0 or treatment_files_per_second <= 0:
        raise ValueError("Probe throughputs must be positive")
    if not 0 <= required_headroom_fraction < 1:
        raise ValueError("required_headroom_fraction must be in [0, 1)")
    if (dph_total is None) != (max_incremental_cost_usd is None):
        raise ValueError(
            "dph_total and max_incremental_cost_usd must be provided together"
        )

    completed = len(epoch_stages)
    median_train = statistics.median(
        float(row["train_seconds"]) for row in epoch_stages
    )
    median_validation = statistics.median(
        float(row["raw_val_seconds"]) + float(row["ema_val_seconds"])
        for row in epoch_stages
    )
    observed_epoch_wall = float(control_wall_seconds) / completed
    # Account for setup, checkpointing, aggregation and other non-train work by
    # never using less than either observed residual or measured validation.
    nontrain_seconds = max(
        median_validation,
        observed_epoch_wall - median_train,
    )
    train_time_multiplier = (
        float(control_files_per_second) / float(treatment_files_per_second)
    )
    projected_epoch_seconds = (
        nontrain_seconds + median_train * train_time_multiplier
    )
    projected_total_seconds = projected_epoch_seconds * int(treatment_epochs)
    timeout_seconds = float(timeout_hours) * 3600.0
    headroom_fraction = 1.0 - projected_total_seconds / timeout_seconds
    time_gate_pass = headroom_fraction >= required_headroom_fraction

    projected_cost_usd = None
    cost_gate_pass = True
    if dph_total is not None and max_incremental_cost_usd is not None:
        projected_cost_usd = (
            projected_total_seconds / 3600.0 * float(dph_total)
        )
        cost_gate_pass = projected_cost_usd <= float(max_incremental_cost_usd)

    return {
        "control_completed_epochs": completed,
        "control_observed_epoch_wall_seconds": observed_epoch_wall,
        "control_median_train_seconds": median_train,
        "control_median_validation_seconds": median_validation,
        "conservative_nontrain_seconds_per_epoch": nontrain_seconds,
        "control_probe_files_per_second": control_files_per_second,
        "treatment_probe_files_per_second": treatment_files_per_second,
        "treatment_train_time_multiplier": train_time_multiplier,
        "treatment_epochs": int(treatment_epochs),
        "projected_treatment_epoch_seconds": projected_epoch_seconds,
        "projected_treatment_total_seconds": projected_total_seconds,
        "projected_treatment_hours": projected_total_seconds / 3600.0,
        "timeout_hours": float(timeout_hours),
        "required_headroom_fraction": float(required_headroom_fraction),
        "projected_headroom_fraction": headroom_fraction,
        "time_gate_pass": time_gate_pass,
        "dph_total": dph_total,
        "projected_incremental_cost_usd": projected_cost_usd,
        "max_incremental_cost_usd": max_incremental_cost_usd,
        "cost_gate_pass": cost_gate_pass,
        "launch_runtime_gate_pass": time_gate_pass and cost_gate_pass,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-log", type=Path, required=True)
    parser.add_argument("--control-probe-report", type=Path, required=True)
    parser.add_argument("--treatment-probe-report", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--control-wall-seconds", type=float, required=True)
    parser.add_argument("--treatment-epochs", type=int, default=120)
    parser.add_argument("--timeout-hours", type=float, default=12.0)
    parser.add_argument("--required-headroom-fraction", type=float, default=0.20)
    parser.add_argument("--dph-total", type=float, default=None)
    parser.add_argument("--max-incremental-cost-usd", type=float, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    stages = parse_complete_epoch_stages(
        args.control_log.read_text(encoding="utf-8", errors="replace")
    )
    control_probe = json.loads(
        args.control_probe_report.read_text(encoding="utf-8")
    )
    treatment_probe = json.loads(
        args.treatment_probe_report.read_text(encoding="utf-8")
    )
    projection = project_paired_runtime(
        epoch_stages=stages,
        control_wall_seconds=args.control_wall_seconds,
        control_files_per_second=probe_files_per_second(
            control_probe, args.batch_size,
        ),
        treatment_files_per_second=probe_files_per_second(
            treatment_probe, args.batch_size,
        ),
        treatment_epochs=args.treatment_epochs,
        timeout_hours=args.timeout_hours,
        required_headroom_fraction=args.required_headroom_fraction,
        dph_total=args.dph_total,
        max_incremental_cost_usd=args.max_incremental_cost_usd,
    )
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "control_log": str(args.control_log),
        "control_probe_report": str(args.control_probe_report),
        "treatment_probe_report": str(args.treatment_probe_report),
        "batch_size": args.batch_size,
        **projection,
        "warning": (
            "Operational launch gate only. It does not authorize a recipe, "
            "checkpoint, threshold, blend or leaderboard decision."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if projection["launch_runtime_gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
