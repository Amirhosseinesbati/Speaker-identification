"""Evaluate the locked Channel-Robust stateful-continuation gate.

This script does not launch training.  It converts the preregistered terminal
rules into a deterministic JSON decision after the source supervisor stops.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch


ALLOWED_TERMINATIONS = {"timeout", "patience"}
MIN_RAW_BY_EPOCH = 60
MIN_RAW_F1 = 0.93
FINAL_WINDOW = 8
EMA_WINDOW = 10
MIN_EMA_SLOPE = 0.0002
LOSS_WINDOW = 5
MIN_TRAIN_LOSS_DROP = 0.05
MAX_VAL_LOSS_RISE = 0.03
MIN_TREND_TESTS = 2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_payload(path: Path) -> Any:
    if path.suffix.lower() in {".pt", ".pth", ".ckpt"}:
        return torch.load(path, map_location="cpu", weights_only=False)
    return json.loads(path.read_text(encoding="utf-8"))


def load_history(path: Path) -> list[dict]:
    payload = _load_payload(path)
    if isinstance(payload, dict):
        payload = payload.get("training_history", payload.get("history"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("History source must contain a non-empty history list")
    if not all(isinstance(row, dict) and "epoch" in row for row in payload):
        raise ValueError("Every history row must be an object with an epoch")
    history = sorted(
        (dict(row) for row in payload), key=lambda row: int(row["epoch"]),
    )
    epochs = [int(row["epoch"]) for row in history]
    if epochs != list(range(1, epochs[-1] + 1)):
        raise ValueError("History epochs must be unique and contiguous from 1")
    return history


def _finite_series(history: list[dict], key: str) -> list[float]:
    values: list[float] = []
    for row in history:
        value = row.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"History key {key} contains a missing/non-finite value")
        values.append(float(value))
    return values


def _ols_slope(values: list[float]) -> float:
    x_mean = (len(values) - 1) / 2.0
    y_mean = sum(values) / len(values)
    numerator = sum(
        (index - x_mean) * (value - y_mean)
        for index, value in enumerate(values)
    )
    denominator = sum((index - x_mean) ** 2 for index in range(len(values)))
    return numerator / denominator


def evaluate_continuation(
    history: list[dict],
    *,
    selected_raw_epoch: int,
    termination: str,
    provenance_ok: bool,
    safety_ok: bool,
) -> dict:
    """Return the preregistered mechanical decision and all intermediates."""
    raw = _finite_series(history, "val_macro_f1")
    ema = _finite_series(history, "val_ema_macro_f1")
    train_loss = _finite_series(history, "train_loss")
    val_loss = _finite_series(history, "val_loss")
    final_epoch = int(history[-1]["epoch"])

    if selected_raw_epoch < 1 or selected_raw_epoch > final_epoch:
        raise ValueError("Selected Raw epoch is outside the complete history")
    best_epoch = int(max(history, key=lambda row: float(row["val_macro_f1"]))["epoch"])
    if selected_raw_epoch != best_epoch:
        raise ValueError(
            f"Selected Raw epoch {selected_raw_epoch} is not history best {best_epoch}"
        )

    checkpoint_recent = final_epoch - selected_raw_epoch < FINAL_WINDOW
    ema_slope = (
        _ols_slope(ema[-EMA_WINDOW:]) if len(ema) >= EMA_WINDOW else None
    )
    ema_rising = ema_slope is not None and ema_slope >= MIN_EMA_SLOPE

    train_drop = None
    val_rise = None
    loss_adapting = False
    if len(history) >= LOSS_WINDOW * 2:
        previous_train = train_loss[-LOSS_WINDOW * 2:-LOSS_WINDOW]
        final_train = train_loss[-LOSS_WINDOW:]
        previous_val = val_loss[-LOSS_WINDOW * 2:-LOSS_WINDOW]
        final_val = val_loss[-LOSS_WINDOW:]
        train_drop = sum(previous_train) / LOSS_WINDOW - sum(final_train) / LOSS_WINDOW
        val_rise = sum(final_val) / LOSS_WINDOW - sum(previous_val) / LOSS_WINDOW
        loss_adapting = (
            train_drop >= MIN_TRAIN_LOSS_DROP and val_rise <= MAX_VAL_LOSS_RISE
        )

    pre60 = [
        float(row["val_macro_f1"])
        for row in history if int(row["epoch"]) <= MIN_RAW_BY_EPOCH
    ]
    best_raw_by_60 = max(pre60) if pre60 else None
    raw_gate = best_raw_by_60 is not None and best_raw_by_60 >= MIN_RAW_F1
    tests = {
        "selected_raw_within_final_8": checkpoint_recent,
        "ema_final10_slope_at_least_0_0002": ema_rising,
        "loss_adaptation_guardrail": loss_adapting,
    }
    passed_tests = sum(bool(value) for value in tests.values())
    terminal_ok = str(termination).lower().strip() in ALLOWED_TERMINATIONS
    eligible = bool(
        terminal_ok
        and provenance_ok
        and safety_ok
        and raw_gate
        and passed_tests >= MIN_TREND_TESTS
    )

    return {
        "schema_version": 1,
        "eligible": eligible,
        "termination": str(termination).lower().strip(),
        "termination_eligible": terminal_ok,
        "provenance_ok": bool(provenance_ok),
        "safety_ok": bool(safety_ok),
        "final_epoch": final_epoch,
        "selected_raw_epoch": int(selected_raw_epoch),
        "selected_raw_f1": raw[selected_raw_epoch - 1],
        "best_raw_by_epoch_60": best_raw_by_60,
        "raw_0_93_by_epoch_60": raw_gate,
        "trend_tests": tests,
        "trend_tests_passed": passed_tests,
        "trend_tests_required": MIN_TREND_TESTS,
        "ema_final10_ols_slope": ema_slope,
        "train_loss_previous5_minus_final5": train_drop,
        "val_loss_final5_minus_previous5": val_rise,
        "contract": {
            "allowed_terminations": sorted(ALLOWED_TERMINATIONS),
            "selected_checkpoint_final_window": FINAL_WINDOW,
            "ema_window": EMA_WINDOW,
            "min_ema_slope": MIN_EMA_SLOPE,
            "loss_window": LOSS_WINDOW,
            "min_train_loss_drop": MIN_TRAIN_LOSS_DROP,
            "max_val_loss_rise": MAX_VAL_LOSS_RISE,
            "min_raw_by_epoch": MIN_RAW_BY_EPOCH,
            "min_raw_f1": MIN_RAW_F1,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--selected-raw-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--termination", choices=["timeout", "patience", "complete", "failure"],
        required=True,
    )
    parser.add_argument("--provenance-ok", action="store_true")
    parser.add_argument("--safety-ok", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    history_path = args.history.expanduser().resolve()
    checkpoint_path = args.selected_raw_checkpoint.expanduser().resolve()
    if not history_path.is_file():
        raise FileNotFoundError(history_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = _load_payload(checkpoint_path)
    if not isinstance(checkpoint, dict):
        raise ValueError("Selected Raw checkpoint must be a checkpoint object")
    selected_epoch = int(checkpoint.get("epoch", 0))
    weight_variant = str(checkpoint.get("weight_variant", "raw")).lower().strip()
    if weight_variant != "raw":
        raise ValueError(
            f"Selected continuation checkpoint must be Raw, got {weight_variant!r}"
        )
    result = evaluate_continuation(
        load_history(history_path),
        selected_raw_epoch=selected_epoch,
        termination=args.termination,
        provenance_ok=args.provenance_ok,
        safety_ok=args.safety_ok,
    )
    checkpoint_f1 = checkpoint.get("val_macro_f1")
    if (not isinstance(checkpoint_f1, (int, float)) or
            not math.isclose(
                float(checkpoint_f1), float(result["selected_raw_f1"]),
                rel_tol=0.0, abs_tol=1e-9,
            )):
        raise ValueError(
            "Selected checkpoint Macro-F1 does not match its history epoch"
        )
    result.update({
        "history_path": str(history_path),
        "history_sha256": _sha256(history_path),
        "selected_raw_checkpoint": str(checkpoint_path),
        "selected_raw_checkpoint_sha256": _sha256(checkpoint_path),
    })
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["eligible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
