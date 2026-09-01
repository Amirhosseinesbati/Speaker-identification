"""Evaluate an immutable equal-weight pair of aligned OOF probability files.

The only gate-bearing candidate is fixed 50/50 probability averaging.  The
historical 60/40 pair is emitted for forensic comparison but is explicitly
ineligible for selection because its weight was chosen on Fold 0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_control_oof_centroid_crossfit import (  # noqa: E402
    NUM_CLASSES,
    metric_bundle,
    metric_delta,
)


PRIMARY_WEIGHTS = (0.5, 0.5)
HISTORICAL_DIAGNOSTIC_WEIGHTS = (0.6, 0.4)
GATE = {
    "equal_fusion_macro_f1_min": 0.96,
    "macro_f1_gain_vs_better_single_min": 0.002,
    "known_accuracy_delta_vs_better_single_min": -0.001,
    "ood_f1_delta_vs_better_single_min": -0.001,
    "rescue_rate_vs_better_single_min": 0.25,
    "rescued_errors_must_exceed_introduced_errors": True,
    "historical_training_overlap_rows_max": 0,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_oof(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        record = {key: archive[key].copy() for key in archive.files}
    required = {"files", "labels", "competition_probs", "split_fold", "split_folds", "split_seed"}
    missing = required - set(record)
    if missing:
        raise RuntimeError(f"OOF {path} lacks keys: {sorted(missing)}")
    files = record["files"].astype(str)
    probabilities = record["competition_probs"].astype(np.float64)
    if len(set(files.tolist())) != len(files):
        raise RuntimeError(f"OOF {path} contains duplicate files")
    if probabilities.shape != (len(files), NUM_CLASSES):
        raise RuntimeError(f"OOF {path} has probability shape {probabilities.shape}")
    if not np.isfinite(probabilities).all() or np.any(probabilities < -1e-7):
        raise RuntimeError(f"OOF {path} contains invalid probabilities")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=2e-5):
        raise RuntimeError(f"OOF {path} probability rows do not sum to one")
    return record


def collapse_labels(labels: np.ndarray) -> np.ndarray:
    values = np.asarray(labels, dtype=np.int64)
    return np.where(values >= NUM_CLASSES, 0, values).astype(np.int64)


def transition_bundle(
    labels: np.ndarray,
    baseline_predictions: np.ndarray,
    candidate_predictions: np.ndarray,
) -> dict:
    baseline_correct = baseline_predictions == labels
    candidate_correct = candidate_predictions == labels
    rescued = ~baseline_correct & candidate_correct
    introduced = baseline_correct & ~candidate_correct
    baseline_errors = int(np.sum(~baseline_correct))
    return {
        "rescued_errors": int(np.sum(rescued)),
        "introduced_errors": int(np.sum(introduced)),
        "baseline_errors": baseline_errors,
        "rescue_rate": float(np.sum(rescued) / max(baseline_errors, 1)),
    }


def class_coverage(labels: np.ndarray) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    known = labels > 0
    known_labels, known_counts = np.unique(labels[known], return_counts=True)
    return {
        "rows": int(len(labels)),
        "known_rows": int(np.sum(known)),
        "unknown_rows": int(np.sum(~known)),
        "unique_known_classes": int(len(known_labels)),
        "known_support_min": int(known_counts.min()) if len(known_counts) else 0,
        "known_support_max": int(known_counts.max()) if len(known_counts) else 0,
    }


def subset_evaluation(
    labels: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    mask: np.ndarray,
) -> dict:
    indices = np.flatnonzero(mask)
    subset_labels = labels[indices]
    pred1 = p1[indices].argmax(axis=1).astype(np.int64)
    pred2 = p2[indices].argmax(axis=1).astype(np.int64)
    equal_pred = (0.5 * p1[indices] + 0.5 * p2[indices]).argmax(axis=1)
    metric1 = metric_bundle(subset_labels, pred1)
    metric2 = metric_bundle(subset_labels, pred2)
    if metric1["macro_f1"] >= metric2["macro_f1"]:
        better = pred1
    else:
        better = pred2
    return {
        "coverage": class_coverage(subset_labels),
        "metrics": {
            "primary": metric1,
            "secondary": metric2,
            "equal_50_50": metric_bundle(subset_labels, equal_pred),
        },
        "equal_error_transitions": transition_bundle(
            subset_labels, better, equal_pred
        ),
    }


def evaluate_pair(primary: dict, secondary: dict) -> dict:
    primary_files = primary["files"].astype(str)
    secondary_files = secondary["files"].astype(str)
    if set(primary_files.tolist()) != set(secondary_files.tolist()):
        raise RuntimeError("Paired OOF file sets differ")
    index = {name: position for position, name in enumerate(secondary_files)}
    order = np.asarray([index[name] for name in primary_files], dtype=np.int64)
    primary_labels = collapse_labels(primary["labels"])
    secondary_labels = collapse_labels(secondary["labels"])[order]
    if not np.array_equal(primary_labels, secondary_labels):
        raise RuntimeError("Paired OOF labels differ after file alignment")
    for key, expected in (("split_fold", 0), ("split_folds", 3), ("split_seed", 42)):
        left = int(np.asarray(primary[key]).reshape(-1)[0])
        right = int(np.asarray(secondary[key]).reshape(-1)[0])
        if left != right or left != expected:
            raise RuntimeError(
                f"Paired OOF {key} mismatch: expected={expected}, "
                f"primary={left}, secondary={right}"
            )

    p1 = primary["competition_probs"].astype(np.float64)
    p2 = secondary["competition_probs"].astype(np.float64)[order]
    pred1 = p1.argmax(axis=1).astype(np.int64)
    pred2 = p2.argmax(axis=1).astype(np.int64)
    metric1 = metric_bundle(primary_labels, pred1)
    metric2 = metric_bundle(primary_labels, pred2)
    if metric1["macro_f1"] >= metric2["macro_f1"]:
        better_name, better_metrics, better_predictions = "primary", metric1, pred1
    else:
        better_name, better_metrics, better_predictions = "secondary", metric2, pred2

    equal_probs = PRIMARY_WEIGHTS[0] * p1 + PRIMARY_WEIGHTS[1] * p2
    historical_probs = (
        HISTORICAL_DIAGNOSTIC_WEIGHTS[0] * p1
        + HISTORICAL_DIAGNOSTIC_WEIGHTS[1] * p2
    )
    equal_predictions = equal_probs.argmax(axis=1).astype(np.int64)
    historical_predictions = historical_probs.argmax(axis=1).astype(np.int64)
    equal_metrics = metric_bundle(primary_labels, equal_predictions)
    historical_metrics = metric_bundle(primary_labels, historical_predictions)
    transitions = transition_bundle(
        primary_labels, better_predictions, equal_predictions
    )
    delta = metric_delta(equal_metrics, better_metrics)
    historical_overlap = None
    if "historical_split" in secondary:
        historical_split = secondary["historical_split"].astype(str)[order]
        unexpected = sorted(set(historical_split.tolist()) - {"train", "val"})
        if unexpected:
            raise RuntimeError(
                f"Historical split contains unexpected values: {unexpected}"
            )
        train_mask = historical_split == "train"
        val_mask = historical_split == "val"
        historical_overlap = {
            "training_overlap_rows": int(np.sum(train_mask)),
            "training_overlap_fraction": float(np.mean(train_mask)),
            "held_out_rows": int(np.sum(val_mask)),
            "held_out_fraction": float(np.mean(val_mask)),
            "full_coverage": class_coverage(primary_labels),
            "training_overlap_subset": subset_evaluation(
                primary_labels, p1, p2, train_mask
            ),
            "held_out_subset": subset_evaluation(
                primary_labels, p1, p2, val_mask
            ),
            "warning": (
                "Full-fold metrics are invalid for selection whenever any row "
                "was used to train the historical model. Held-out subset metrics "
                "are descriptive only when class coverage is incomplete."
            ),
        }
    provenance_disjoint = (
        historical_overlap is None
        or historical_overlap["training_overlap_rows"]
        <= GATE["historical_training_overlap_rows_max"]
    )
    checks = {
        "absolute_macro": equal_metrics["macro_f1"] >= GATE["equal_fusion_macro_f1_min"],
        "macro_gain": delta["macro_f1"] >= GATE["macro_f1_gain_vs_better_single_min"],
        "known_guardrail": delta["known_accuracy"] >= GATE["known_accuracy_delta_vs_better_single_min"],
        "ood_guardrail": delta["ood_f1"] >= GATE["ood_f1_delta_vs_better_single_min"],
        "rescue_rate": transitions["rescue_rate"] >= GATE["rescue_rate_vs_better_single_min"],
        "rescued_exceed_introduced": transitions["rescued_errors"] > transitions["introduced_errors"],
        "provenance_disjoint": provenance_disjoint,
    }
    result = {
        "rows": int(len(primary_files)),
        "better_single": better_name,
        "metrics": {
            "primary": metric1,
            "secondary": metric2,
            "equal_50_50": equal_metrics,
            "historical_60_40_diagnostic_only": historical_metrics,
        },
        "equal_delta_vs_better_single": delta,
        "equal_error_transitions": transitions,
        "gate": {
            "contract": dict(GATE),
            "checks": checks,
            "passed": all(checks.values()),
        },
        "selection_verdict": (
            "eligible_for_preregistered_gate"
            if provenance_disjoint
            else "rejected_historical_full_fold_gate_due_training_overlap"
        ),
    }
    if historical_overlap is not None:
        result["historical_provenance_audit"] = historical_overlap
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--secondary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    primary = load_oof(args.primary)
    secondary = load_oof(args.secondary)
    evaluation = evaluate_pair(primary, secondary)
    report = {
        "contract": {
            "primary_fusion": {"method": "probability_average", "weights": list(PRIMARY_WEIGHTS)},
            "search_dimensions": 0,
            "historical_60_40_selection_eligible": False,
            "leaderboard_used_for_selection": False,
            "single_fold_submission_authorized": False,
        },
        "provenance": {
            "primary": str(args.primary.resolve()),
            "primary_sha256": sha256_file(args.primary),
            "secondary": str(args.secondary.resolve()),
            "secondary_sha256": sha256_file(args.secondary),
        },
        "evaluation": evaluation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(evaluation, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
