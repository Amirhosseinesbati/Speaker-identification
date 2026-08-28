"""Compare candidate and baseline training histories without touching a run.

The output is JSON so campaign heartbeats can use the same evidence for
Telegram interpretation, scientific gates, and the final run report.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable


DEFAULT_METRICS = (
    "val_macro_f1",
    "val_logit_avg_macro_f1",
    "val_known_acc",
    "val_ood_f1",
    "val_ema_macro_f1",
    "val_ema_logit_avg_macro_f1",
    "val_ema_known_acc",
    "val_ema_ood_f1",
    "train_loss",
    "val_loss",
    "lr",
)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _load_payload(path: Path) -> Any:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def _primary_loss_scale(payload: Any) -> float:
    if not isinstance(payload, dict):
        return 1.0
    config = payload.get("config", {}) or {}
    training = config.get("training", {}) or {}
    loss = training.get("loss", {}) or {}
    speaker = loss.get("speaker", {}) or {}
    ood = loss.get("ood", {}) or {}
    speaker_weight = float(
        speaker.get("weight", training.get("speaker_loss_weight", 1.0))
    )
    ood_weight = float(ood.get("weight", training.get("ood_loss_weight", 1.0)))
    scale = speaker_weight + ood_weight
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Primary speaker/OOD loss weights must have a positive sum")
    return scale


def load_history_context(path: Path) -> tuple[list[dict[str, Any]], float]:
    payload = _load_payload(path)
    loss_scale = _primary_loss_scale(payload)
    history_payload = payload
    if isinstance(history_payload, dict):
        history_payload = history_payload.get(
            "training_history", history_payload.get("history")
        )
    if not isinstance(history_payload, list) or not history_payload:
        raise ValueError(f"No non-empty training history in {path}")
    if not all(
        isinstance(row, dict) and _finite(row.get("epoch"))
        for row in history_payload
    ):
        raise ValueError(f"Malformed epoch rows in {path}")
    return (
        sorted(history_payload, key=lambda row: int(row["epoch"])),
        loss_scale,
    )


def load_history(path: Path) -> list[dict[str, Any]]:
    return load_history_context(path)[0]


def _rows_by_epoch(history: Iterable[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(row["epoch"]): row for row in history}


def _best(history: Iterable[dict[str, Any]], key: str, *, maximize: bool) -> dict | None:
    rows = [row for row in history if _finite(row.get(key))]
    if not rows:
        return None
    selected = (max if maximize else min)(rows, key=lambda row: float(row[key]))
    return {"epoch": int(selected["epoch"]), "value": float(selected[key])}


def _summary(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def compare_histories(
    candidate: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    *,
    tail_window: int = 10,
    candidate_primary_loss_scale: float = 1.0,
    baseline_primary_loss_scale: float = 1.0,
) -> dict[str, Any]:
    if candidate_primary_loss_scale <= 0 or baseline_primary_loss_scale <= 0:
        raise ValueError("Primary loss scales must be positive")
    candidate_by_epoch = _rows_by_epoch(candidate)
    baseline_by_epoch = _rows_by_epoch(baseline)
    common_epochs = sorted(set(candidate_by_epoch) & set(baseline_by_epoch))
    if not common_epochs:
        raise ValueError("Candidate and baseline have no common epochs")
    latest_epoch = common_epochs[-1]
    horizon_baseline = [row for row in baseline if int(row["epoch"]) <= latest_epoch]

    metric_keys = sorted(
        {
            key
            for row in candidate + baseline
            for key in row
            if key != "epoch" and _finite(row.get(key))
        }
        & set(DEFAULT_METRICS)
    )
    latest: dict[str, Any] = {}
    tail: dict[str, Any] = {}
    tail_epochs = common_epochs[-tail_window:]
    for key in metric_keys:
        candidate_value = candidate_by_epoch[latest_epoch].get(key)
        baseline_value = baseline_by_epoch[latest_epoch].get(key)
        if _finite(candidate_value) and _finite(baseline_value):
            latest[key] = {
                "candidate": float(candidate_value),
                "baseline": float(baseline_value),
                "delta": float(candidate_value) - float(baseline_value),
            }
        deltas = [
            float(candidate_by_epoch[epoch][key])
            - float(baseline_by_epoch[epoch][key])
            for epoch in tail_epochs
            if _finite(candidate_by_epoch[epoch].get(key))
            and _finite(baseline_by_epoch[epoch].get(key))
        ]
        if deltas:
            tail[key] = {
                **(_summary(deltas) or {}),
                "positive_fraction": sum(delta > 0 for delta in deltas) / len(deltas),
            }

    maxima = {key for key in metric_keys if key not in {"train_loss", "val_loss"}}
    best = {
        key: {
            "candidate": _best(candidate, key, maximize=key in maxima),
            "baseline_same_horizon": _best(
                horizon_baseline, key, maximize=key in maxima
            ),
            "baseline_full": _best(baseline, key, maximize=key in maxima),
        }
        for key in metric_keys
    }

    candidate_last = candidate_by_epoch[latest_epoch]
    baseline_last = baseline_by_epoch[latest_epoch]

    def normalized_primary_loss(
        row: dict[str, Any], key: str, scale: float
    ) -> float | None:
        normalized_key = f"{key}_primary_normalized"
        if _finite(row.get(normalized_key)):
            return float(row[normalized_key])
        if key == "train_loss" and _finite(row.get("train_loss_primary")):
            return float(row["train_loss_primary"]) / scale
        if _finite(row.get(key)):
            return float(row[key]) / scale
        return None

    normalized_latest = {}
    normalized_tail = {}
    for key in ("train_loss", "val_loss"):
        candidate_value = normalized_primary_loss(
            candidate_last, key, candidate_primary_loss_scale
        )
        baseline_value = normalized_primary_loss(
            baseline_last, key, baseline_primary_loss_scale
        )
        if candidate_value is not None and baseline_value is not None:
            normalized_latest[key] = {
                "candidate": candidate_value,
                "baseline": baseline_value,
                "delta": candidate_value - baseline_value,
            }
        deltas = []
        for epoch in tail_epochs:
            candidate_value = normalized_primary_loss(
                candidate_by_epoch[epoch], key, candidate_primary_loss_scale
            )
            baseline_value = normalized_primary_loss(
                baseline_by_epoch[epoch], key, baseline_primary_loss_scale
            )
            if candidate_value is not None and baseline_value is not None:
                deltas.append(candidate_value - baseline_value)
        if deltas:
            normalized_tail[key] = _summary(deltas)

    def gap(row: dict[str, Any], left: str, right: str) -> float | None:
        if not _finite(row.get(left)) or not _finite(row.get(right)):
            return None
        return float(row[left]) - float(row[right])

    return {
        "candidate_epochs": len(candidate),
        "baseline_epochs": len(baseline),
        "latest_common_epoch": latest_epoch,
        "tail_window": len(tail_epochs),
        "latest": latest,
        "tail_delta": tail,
        "best": best,
        "primary_loss_scale_correction": {
            "candidate_weight_sum": candidate_primary_loss_scale,
            "baseline_weight_sum": baseline_primary_loss_scale,
            "latest": normalized_latest,
            "tail_delta": normalized_tail,
            "interpretation": (
                "Use these normalized primary losses for cross-recipe comparison. "
                "Raw loss may omit or differently weight auxiliary objectives."
            ),
        },
        "diagnostic_gaps": {
            "candidate_probability_minus_logit": gap(
                candidate_last, "val_macro_f1", "val_logit_avg_macro_f1"
            ),
            "baseline_probability_minus_logit": gap(
                baseline_last, "val_macro_f1", "val_logit_avg_macro_f1"
            ),
            "candidate_raw_minus_ema": gap(
                candidate_last, "val_macro_f1", "val_ema_macro_f1"
            ),
            "baseline_raw_minus_ema": gap(
                baseline_last, "val_macro_f1", "val_ema_macro_f1"
            ),
            "candidate_val_loss_above_min": (
                float(candidate_last["val_loss"])
                - min(float(row["val_loss"]) for row in candidate if _finite(row.get("val_loss")))
                if _finite(candidate_last.get("val_loss"))
                else None
            ),
            "baseline_val_loss_above_min": (
                float(baseline_last["val_loss"])
                - min(
                    float(row["val_loss"])
                    for row in horizon_baseline
                    if _finite(row.get("val_loss"))
                )
                if _finite(baseline_last.get("val_loss"))
                else None
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--tail-window", type=int, default=10)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.tail_window <= 0:
        raise ValueError("tail-window must be positive")
    candidate_history, candidate_loss_scale = load_history_context(args.candidate)
    baseline_history, baseline_loss_scale = load_history_context(args.baseline)
    result = compare_histories(
        candidate_history,
        baseline_history,
        tail_window=args.tail_window,
        candidate_primary_loss_scale=candidate_loss_scale,
        baseline_primary_loss_scale=baseline_loss_scale,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
