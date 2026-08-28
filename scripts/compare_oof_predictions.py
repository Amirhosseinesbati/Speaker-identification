"""Measure same-fold OOF error complementarity without tuning a fusion.

The candidate and baseline must cover the same validation files, labels, class
space and split.  Fixed blends are reported only as descriptive diagnostics;
choosing a weight from these scores on the same fold would be selection-biased.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import evaluate_competition_probs, macro_f1_score


REQUIRED_ARRAYS = {
    "files",
    "labels",
    "competition_probs",
    "split_scheme",
    "split_fold",
    "split_folds",
    "split_seed",
}
SPLIT_KEYS = ("split_scheme", "split_fold", "split_folds", "split_seed")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scalar(record: dict[str, np.ndarray], key: str) -> Any:
    value = record[key]
    if value.size != 1:
        raise ValueError(f"{key} must contain exactly one value")
    return value.reshape(-1)[0].item()


def load_oof(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        missing = sorted(REQUIRED_ARRAYS - set(payload.files))
        if missing:
            raise ValueError(f"{path} is missing OOF arrays: {', '.join(missing)}")
        record = {key: np.asarray(payload[key]).copy() for key in payload.files}

    files = record["files"]
    labels = record["labels"]
    probs = record["competition_probs"]
    if files.ndim != 1 or labels.ndim != 1:
        raise ValueError(f"{path} files and labels must be one-dimensional")
    if probs.ndim != 2 or probs.shape[1] < 2:
        raise ValueError(f"{path} competition_probs must be a non-empty matrix")
    if len(files) != len(labels) or len(files) != len(probs):
        raise ValueError(f"{path} has inconsistent OOF sample counts")
    if len(set(files.tolist())) != len(files):
        raise ValueError(f"{path} contains duplicate OOF filenames")
    if not np.isfinite(probs).all() or (probs < 0).any():
        raise ValueError(f"{path} contains invalid competition probabilities")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-4, rtol=1e-4):
        raise ValueError(f"{path} competition probability rows do not sum to one")
    return record


def _load_class_map(bundle: Path) -> dict | None:
    path = bundle / "class_map.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _prediction_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    return evaluate_competition_probs(
        torch.from_numpy(probabilities), torch.from_numpy(labels)
    )


def _metrics_from_predictions(
    predictions: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score

    y_true = np.where(labels > num_classes - 1, 0, labels)
    known = y_true != 0
    return {
        "macro_f1": macro_f1_score(y_true, predictions, num_classes=num_classes),
        "ood_f1": float(
            f1_score(
                (y_true == 0).astype(int),
                (predictions == 0).astype(int),
                zero_division=0,
            )
        ),
        "known_acc": (
            float(accuracy_score(y_true[known], predictions[known]))
            if known.any()
            else 0.0
        ),
        "overall_acc": float(accuracy_score(y_true, predictions)),
    }


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _subgroup_diagnostics(
    candidate_correct: np.ndarray,
    baseline_correct: np.ndarray,
    mask: np.ndarray,
) -> dict[str, int | float]:
    candidate_correct = candidate_correct[mask]
    baseline_correct = baseline_correct[mask]
    samples = int(mask.sum())
    candidate_only = int((candidate_correct & ~baseline_correct).sum())
    baseline_only = int((~candidate_correct & baseline_correct).sum())
    both_wrong = int((~candidate_correct & ~baseline_correct).sum())
    return {
        "samples": samples,
        "candidate_only_correct": candidate_only,
        "baseline_only_correct": baseline_only,
        "both_wrong": both_wrong,
        "disagreement_rate": _safe_ratio(
            int((candidate_correct != baseline_correct).sum()), samples
        ),
        "oracle_correct_rate": _safe_ratio(samples - both_wrong, samples),
    }


def compare_oof(
    candidate_path: Path,
    baseline_path: Path,
    *,
    fixed_candidate_weights: Sequence[float] = (0.25, 0.5, 0.75),
) -> dict[str, Any]:
    candidate = load_oof(candidate_path)
    baseline = load_oof(baseline_path)

    candidate_files = candidate["files"].tolist()
    baseline_files = baseline["files"].tolist()
    if set(candidate_files) != set(baseline_files):
        missing_candidate = len(set(baseline_files) - set(candidate_files))
        missing_baseline = len(set(candidate_files) - set(baseline_files))
        raise ValueError(
            "Candidate and baseline OOF file sets differ: "
            f"candidate_missing={missing_candidate}, baseline_missing={missing_baseline}"
        )

    baseline_index = {name: index for index, name in enumerate(baseline_files)}
    order = np.asarray([baseline_index[name] for name in candidate_files])
    order_aligned = not np.array_equal(order, np.arange(len(order)))
    baseline_labels = baseline["labels"][order]
    baseline_probs = baseline["competition_probs"][order]
    candidate_labels = candidate["labels"]
    candidate_probs = candidate["competition_probs"]

    if not np.array_equal(candidate_labels, baseline_labels):
        raise ValueError("Candidate and baseline labels differ after filename alignment")
    if candidate_probs.shape != baseline_probs.shape:
        raise ValueError("Candidate and baseline competition probability shapes differ")
    for key in SPLIT_KEYS:
        if _scalar(candidate, key) != _scalar(baseline, key):
            raise ValueError(f"Candidate and baseline {key} metadata differ")

    candidate_class_map = _load_class_map(candidate_path.parent)
    baseline_class_map = _load_class_map(baseline_path.parent)
    if (candidate_class_map is None) != (baseline_class_map is None):
        raise ValueError("Only one OOF bundle contains class_map.json")
    if candidate_class_map is not None and candidate_class_map != baseline_class_map:
        raise ValueError("Candidate and baseline class maps differ")

    weights = [float(weight) for weight in fixed_candidate_weights]
    if not weights or any(not math.isfinite(weight) or not 0 <= weight <= 1 for weight in weights):
        raise ValueError("Fixed candidate weights must be finite values in [0, 1]")

    num_classes = int(candidate_probs.shape[1])
    y_true = np.where(candidate_labels > num_classes - 1, 0, candidate_labels)
    candidate_predictions = candidate_probs.argmax(axis=1)
    baseline_predictions = baseline_probs.argmax(axis=1)
    candidate_correct = candidate_predictions == y_true
    baseline_correct = baseline_predictions == y_true
    candidate_errors = ~candidate_correct
    baseline_errors = ~baseline_correct
    both_wrong = candidate_errors & baseline_errors
    either_wrong = candidate_errors | baseline_errors
    candidate_only = candidate_correct & baseline_errors
    baseline_only = candidate_errors & baseline_correct

    oracle_predictions = np.where(
        candidate_correct, candidate_predictions, baseline_predictions
    )
    oracle_metrics = _metrics_from_predictions(
        oracle_predictions, candidate_labels, num_classes
    )
    candidate_metrics = _prediction_metrics(candidate_probs, candidate_labels)
    baseline_metrics = _prediction_metrics(baseline_probs, candidate_labels)
    standalone_best = max(candidate_metrics["macro_f1"], baseline_metrics["macro_f1"])

    fixed_blends = []
    for weight in weights:
        blend = weight * candidate_probs + (1.0 - weight) * baseline_probs
        fixed_blends.append(
            {
                "candidate_weight": weight,
                "baseline_weight": 1.0 - weight,
                **_prediction_metrics(blend, candidate_labels),
            }
        )

    samples = len(candidate_files)
    candidate_error_count = int(candidate_errors.sum())
    baseline_error_count = int(baseline_errors.sum())
    both_wrong_count = int(both_wrong.sum())
    candidate_only_count = int(candidate_only.sum())
    baseline_only_count = int(baseline_only.sum())
    known_mask = y_true != 0

    return {
        "integrity": {
            "candidate": str(candidate_path),
            "candidate_sha256": _sha256(candidate_path),
            "baseline": str(baseline_path),
            "baseline_sha256": _sha256(baseline_path),
            "samples": samples,
            "competition_classes": num_classes,
            "split": {key: _scalar(candidate, key) for key in SPLIT_KEYS},
            "filename_order_aligned": order_aligned,
            "class_map_verified": candidate_class_map is not None,
        },
        "standalone": {
            "candidate": candidate_metrics,
            "baseline": baseline_metrics,
        },
        "complementarity": {
            "prediction_disagreement_count": int(
                (candidate_predictions != baseline_predictions).sum()
            ),
            "prediction_disagreement_rate": _safe_ratio(
                int((candidate_predictions != baseline_predictions).sum()), samples
            ),
            "candidate_errors": candidate_error_count,
            "baseline_errors": baseline_error_count,
            "both_wrong": both_wrong_count,
            "candidate_only_correct": candidate_only_count,
            "baseline_only_correct": baseline_only_count,
            "error_jaccard": _safe_ratio(both_wrong_count, int(either_wrong.sum())),
            "shared_error_fraction_of_smaller_error_set": _safe_ratio(
                both_wrong_count, min(candidate_error_count, baseline_error_count)
            ),
            "candidate_recovers_baseline_errors": _safe_ratio(
                candidate_only_count, baseline_error_count
            ),
            "baseline_recovers_candidate_errors": _safe_ratio(
                baseline_only_count, candidate_error_count
            ),
            "oracle": {
                **oracle_metrics,
                "macro_f1_gain_over_best_standalone": (
                    oracle_metrics["macro_f1"] - standalone_best
                ),
            },
            "known": _subgroup_diagnostics(
                candidate_correct, baseline_correct, known_mask
            ),
            "unknown": _subgroup_diagnostics(
                candidate_correct, baseline_correct, ~known_mask
            ),
        },
        "fixed_blends_descriptive_only": fixed_blends,
        "selection_warning": (
            "Do not select or tune a fusion weight from same-fold scores. "
            "Use predeclared weights or nested/cross-fold selection before submission."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument(
        "--candidate-weight",
        type=float,
        action="append",
        dest="candidate_weights",
        help="Fixed descriptive candidate weight; repeat for several values",
    )
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = compare_oof(
        args.candidate,
        args.baseline,
        fixed_candidate_weights=(
            args.candidate_weights
            if args.candidate_weights is not None
            else (0.25, 0.5, 0.75)
        ),
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
