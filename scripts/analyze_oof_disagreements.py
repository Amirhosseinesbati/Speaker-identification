"""Diagnose same-fold OOF disagreements without selecting a fusion rule.

The report is deliberately descriptive.  It exposes which files one model
rescues, observable confidence signals, audio duration, and optional unknown
cluster membership.  Fixed gates are predeclared diagnostics only: choosing a
gate from their same-fold scores would be selection-biased and must be
confirmed on another fold before it can influence a submission.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import wave
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from scripts.compare_oof_predictions import (
    SPLIT_KEYS,
    _metrics_from_predictions,
    _scalar,
    _sha256,
    compare_oof,
    load_oof,
)


DEFAULT_THRESHOLDS = (0.0, 0.05, 0.10, 0.20)


def _align_records(
    candidate: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    candidate_files = candidate["files"].astype(str).tolist()
    baseline_files = baseline["files"].astype(str).tolist()
    if set(candidate_files) != set(baseline_files):
        raise ValueError("Candidate and baseline OOF file sets differ")
    baseline_index = {name: index for index, name in enumerate(baseline_files)}
    order = np.asarray([baseline_index[name] for name in candidate_files])
    labels = candidate["labels"]
    baseline_labels = baseline["labels"][order]
    if not np.array_equal(labels, baseline_labels):
        raise ValueError("Candidate and baseline labels differ after alignment")
    for key in SPLIT_KEYS:
        if _scalar(candidate, key) != _scalar(baseline, key):
            raise ValueError(f"Candidate and baseline {key} metadata differ")
    return (
        candidate_files,
        labels,
        candidate["competition_probs"],
        baseline["competition_probs"][order],
        order,
    )


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-12, 1.0)
    return -(clipped * np.log(clipped)).sum(axis=1) / math.log(probabilities.shape[1])


def _known_margin(probabilities: np.ndarray) -> np.ndarray:
    known = probabilities[:, 1:]
    if known.shape[1] == 1:
        return known[:, 0]
    top_two = np.partition(known, -2, axis=1)[:, -2:]
    return top_two[:, 1] - top_two[:, 0]


def _duration_seconds(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            return float(handle.getnframes() / rate) if rate else None
    except (FileNotFoundError, wave.Error, EOFError):
        return None


def _quantiles(values: Iterable[float | None]) -> dict[str, float | int | None]:
    numeric = np.asarray(
        [float(value) for value in values if value is not None and math.isfinite(value)],
        dtype=np.float64,
    )
    if not len(numeric):
        return {"count": 0, "mean": None, "p10": None, "p50": None, "p90": None}
    p10, p50, p90 = np.quantile(numeric, [0.1, 0.5, 0.9])
    return {
        "count": int(len(numeric)),
        "mean": float(numeric.mean()),
        "p10": float(p10),
        "p50": float(p50),
        "p90": float(p90),
    }


def _group_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "candidate_ood_probability",
        "baseline_ood_probability",
        "ood_probability_delta",
        "candidate_unknown_margin",
        "baseline_unknown_margin",
        "unknown_margin_delta",
        "candidate_entropy",
        "baseline_entropy",
        "probability_l1",
        "embedding_cosine",
        "duration_seconds",
    )
    return {
        "samples": len(rows),
        "unknown_samples": sum(bool(row["is_unknown"]) for row in rows),
        "features": {field: _quantiles(row.get(field) for row in rows) for field in fields},
    }


def _auc_diagnostic(
    rows: Sequence[dict[str, Any]], field: str
) -> dict[str, float | int | str | None]:
    from sklearn.metrics import roc_auc_score

    single_correct = [
        row
        for row in rows
        if row["outcome"] in {"candidate_only_correct", "baseline_only_correct"}
        and row.get(field) is not None
    ]
    labels = np.asarray(
        [row["outcome"] == "candidate_only_correct" for row in single_correct],
        dtype=np.int64,
    )
    scores = np.asarray([row[field] for row in single_correct], dtype=np.float64)
    if len(single_correct) < 2 or len(np.unique(labels)) < 2:
        return {
            "samples": len(single_correct),
            "auc": None,
            "separability": None,
            "candidate_direction": None,
        }
    auc = float(roc_auc_score(labels, scores))
    return {
        "samples": len(single_correct),
        "auc": auc,
        "separability": max(auc, 1.0 - auc),
        "candidate_direction": "higher" if auc >= 0.5 else "lower",
    }


def _fixed_gate_diagnostics(
    rows: Sequence[dict[str, Any]],
    labels: np.ndarray,
    candidate_predictions: np.ndarray,
    baseline_predictions: np.ndarray,
    *,
    num_classes: int,
    thresholds: Sequence[float],
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for score_name in ("ood_probability_delta", "unknown_margin_delta"):
        scores = np.asarray([row[score_name] for row in rows], dtype=np.float64)
        candidate_predicts_unknown = candidate_predictions == 0
        baseline_predicts_known = baseline_predictions != 0
        for threshold in thresholds:
            use_candidate = (
                candidate_predicts_unknown
                & baseline_predicts_known
                & (scores >= float(threshold))
            )
            predictions = np.where(use_candidate, candidate_predictions, baseline_predictions)
            diagnostics.append(
                {
                    "score": score_name,
                    "threshold": float(threshold),
                    "candidate_selected": int(use_candidate.sum()),
                    **_metrics_from_predictions(predictions, labels, num_classes),
                }
            )
    return diagnostics


def analyze_disagreements(
    candidate_path: Path,
    baseline_path: Path,
    *,
    audio_dir: Path | None = None,
    unknown_cluster_path: Path | None = None,
    fixed_thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    base_report = compare_oof(candidate_path, baseline_path)
    candidate = load_oof(candidate_path)
    baseline = load_oof(baseline_path)
    files, labels, candidate_probs, baseline_probs, baseline_order = _align_records(
        candidate, baseline
    )
    if candidate_probs.shape != baseline_probs.shape:
        raise ValueError("Candidate and baseline competition probability shapes differ")

    candidate_predictions = candidate_probs.argmax(axis=1)
    baseline_predictions = baseline_probs.argmax(axis=1)
    y_true = np.where(labels > candidate_probs.shape[1] - 1, 0, labels)
    candidate_correct = candidate_predictions == y_true
    baseline_correct = baseline_predictions == y_true

    candidate_entropy = _entropy(candidate_probs)
    baseline_entropy = _entropy(baseline_probs)
    candidate_unknown_margin = candidate_probs[:, 0] - candidate_probs[:, 1:].max(axis=1)
    baseline_unknown_margin = baseline_probs[:, 0] - baseline_probs[:, 1:].max(axis=1)
    candidate_known_margin = _known_margin(candidate_probs)
    baseline_known_margin = _known_margin(baseline_probs)

    embedding_cosine: np.ndarray | None = None
    if "embeddings" in candidate and "embeddings" in baseline:
        candidate_embeddings = candidate["embeddings"]
        baseline_embeddings = baseline["embeddings"][baseline_order]
        numerator = (candidate_embeddings * baseline_embeddings).sum(axis=1)
        denominator = np.linalg.norm(candidate_embeddings, axis=1) * np.linalg.norm(
            baseline_embeddings, axis=1
        )
        embedding_cosine = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator, dtype=np.float64),
            where=denominator > 0,
        )

    clusters: dict[str, Any] = {}
    if unknown_cluster_path and unknown_cluster_path.is_file():
        clusters = json.loads(unknown_cluster_path.read_text(encoding="utf-8"))

    rows: list[dict[str, Any]] = []
    for index, filename in enumerate(files):
        if candidate_correct[index] and baseline_correct[index]:
            outcome = "both_correct"
        elif candidate_correct[index]:
            outcome = "candidate_only_correct"
        elif baseline_correct[index]:
            outcome = "baseline_only_correct"
        else:
            outcome = "both_wrong"
        duration = _duration_seconds(audio_dir / filename) if audio_dir else None
        rows.append(
            {
                "file": filename,
                "label": int(y_true[index]),
                "is_unknown": bool(y_true[index] == 0),
                "unknown_cluster": clusters.get(filename),
                "outcome": outcome,
                "candidate_prediction": int(candidate_predictions[index]),
                "baseline_prediction": int(baseline_predictions[index]),
                "candidate_ood_probability": float(candidate_probs[index, 0]),
                "baseline_ood_probability": float(baseline_probs[index, 0]),
                "ood_probability_delta": float(
                    candidate_probs[index, 0] - baseline_probs[index, 0]
                ),
                "candidate_unknown_margin": float(candidate_unknown_margin[index]),
                "baseline_unknown_margin": float(baseline_unknown_margin[index]),
                "unknown_margin_delta": float(
                    candidate_unknown_margin[index] - baseline_unknown_margin[index]
                ),
                "candidate_known_margin": float(candidate_known_margin[index]),
                "baseline_known_margin": float(baseline_known_margin[index]),
                "candidate_entropy": float(candidate_entropy[index]),
                "baseline_entropy": float(baseline_entropy[index]),
                "probability_l1": float(
                    np.abs(candidate_probs[index] - baseline_probs[index]).sum()
                ),
                "embedding_cosine": (
                    float(embedding_cosine[index]) if embedding_cosine is not None else None
                ),
                "duration_seconds": duration,
            }
        )

    outcomes = sorted({row["outcome"] for row in rows})
    outcome_summaries = {
        outcome: _group_summary([row for row in rows if row["outcome"] == outcome])
        for outcome in outcomes
    }
    diagnostic_fields = (
        "ood_probability_delta",
        "unknown_margin_delta",
        "candidate_entropy",
        "baseline_entropy",
        "probability_l1",
        "embedding_cosine",
        "duration_seconds",
    )

    cluster_summary: dict[str, dict[str, int]] = {}
    for row in rows:
        if not row["is_unknown"] or row["unknown_cluster"] is None:
            continue
        key = str(row["unknown_cluster"])
        counts = cluster_summary.setdefault(
            key,
            {
                "samples": 0,
                "candidate_only_correct": 0,
                "baseline_only_correct": 0,
                "both_wrong": 0,
                "both_correct": 0,
            },
        )
        counts["samples"] += 1
        counts[row["outcome"]] += 1

    report = {
        "integrity": {
            **base_report["integrity"],
            "candidate_sha256": _sha256(candidate_path),
            "baseline_sha256": _sha256(baseline_path),
            "audio_dir": str(audio_dir) if audio_dir else None,
            "unknown_cluster_path": (
                str(unknown_cluster_path) if unknown_cluster_path else None
            ),
        },
        "standalone": base_report["standalone"],
        "complementarity": base_report["complementarity"],
        "outcome_summaries": outcome_summaries,
        "single_correct_auc_diagnostics": {
            field: _auc_diagnostic(rows, field) for field in diagnostic_fields
        },
        "unknown_cluster_summary": dict(
            sorted(
                cluster_summary.items(),
                key=lambda item: (
                    -item[1]["candidate_only_correct"],
                    -item[1]["baseline_only_correct"],
                    -item[1]["samples"],
                    item[0],
                ),
            )
        ),
        "fixed_gates_descriptive_only": _fixed_gate_diagnostics(
            rows,
            labels,
            candidate_predictions,
            baseline_predictions,
            num_classes=candidate_probs.shape[1],
            thresholds=fixed_thresholds,
        ),
        "selection_warning": (
            "These diagnostics are same-fold evidence. Do not select a gate or "
            "threshold from this report for submission; predeclare it and confirm "
            "on another fold first."
        ),
    }
    disagreement_rows = [row for row in rows if row["outcome"] != "both_correct"]
    return report, disagreement_rows


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path)
    parser.add_argument("--unknown-cluster-path", type=Path)
    parser.add_argument(
        "--fixed-threshold",
        type=float,
        action="append",
        dest="fixed_thresholds",
        help="Predeclared descriptive threshold; repeat for several values",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-csv", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report, rows = analyze_disagreements(
        args.candidate,
        args.baseline,
        audio_dir=args.audio_dir,
        unknown_cluster_path=args.unknown_cluster_path,
        fixed_thresholds=(
            args.fixed_thresholds
            if args.fixed_thresholds is not None
            else DEFAULT_THRESHOLDS
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(args.case_csv, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
