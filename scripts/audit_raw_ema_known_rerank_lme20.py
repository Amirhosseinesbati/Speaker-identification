"""Binary-locked Raw/EMA known-identity reranker for CAM++ LME20.

The full fixed 50/50 Raw/EMA evidence ensemble reduced wrong-known errors but
failed because it rejected extra known files as unknown.  This single-variable
follow-up preserves every locked Raw known/unknown decision and uses the same
fixed Raw/EMA evidence average only to choose among competition classes
1..446 when Raw has already declared a file known.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_control_oof_centroid_crossfit import (  # noqa: E402
    NUM_FOLDS,
    metric_bundle,
    metric_delta,
)
from scripts.analyze_unknown_cluster_hypotheses_crossfit import (  # noqa: E402
    load_fold_inputs,
)
from scripts.audit_raw_ema_lme20 import (  # noqa: E402
    build_or_load_ema_cache,
    final_decision,
    fixed_raw_ema_decision,
    probability_evidence,
)
from scripts.audit_short_audio_repeat import (  # noqa: E402
    LOCKED_BASELINE_MACRO_F1,
    digest_names,
)


MIN_AGGREGATE_MACRO_GAIN = 0.001
MIN_AGGREGATE_KNOWN_GAIN = 0.001
INVARIANT_ATOL = 1e-12


def binary_locked_known_rerank(
    raw_predictions: np.ndarray,
    ensemble_probabilities: np.ndarray,
) -> np.ndarray:
    """Keep Raw binary decisions and rerank only its known predictions."""
    raw_predictions = np.asarray(raw_predictions, dtype=np.int64)
    probabilities = np.asarray(ensemble_probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[0] != len(raw_predictions):
        raise RuntimeError("Prediction/probability shape mismatch")
    if probabilities.shape[1] != 447:
        raise RuntimeError("Known reranker requires the fixed 447-way output")
    candidate = raw_predictions.copy()
    known = raw_predictions > 0
    candidate[known] = 1 + probabilities[known, 1:].argmax(axis=1)
    if not np.array_equal(candidate == 0, raw_predictions == 0):
        raise RuntimeError("Binary known/unknown invariant failed")
    return candidate


def gate(fold_rows: list[dict], aggregate: dict) -> dict:
    conditions = {
        "all_fold_macro_nonnegative": all(
            row["delta"]["macro_f1"] >= 0.0 for row in fold_rows
        ),
        "aggregate_macro_gain_at_least_0_001": (
            aggregate["delta"]["macro_f1"] >= MIN_AGGREGATE_MACRO_GAIN
        ),
        "aggregate_known_gain_at_least_0_001": (
            aggregate["delta"]["known_accuracy"] >= MIN_AGGREGATE_KNOWN_GAIN
        ),
        "fold_ood_metrics_exact": all(
            abs(row["delta"]["ood_f1"]) <= INVARIANT_ATOL
            for row in fold_rows
        ),
        "aggregate_ood_metric_exact": (
            abs(aggregate["delta"]["ood_f1"]) <= INVARIANT_ATOL
        ),
        "binary_decisions_exact": all(
            row["binary_changed"] == 0 for row in fold_rows
        ),
    }
    return {
        "passed": bool(all(conditions.values())),
        "conditions": conditions,
        "thresholds": {
            "minimum_aggregate_macro_gain": MIN_AGGREGATE_MACRO_GAIN,
            "minimum_aggregate_known_gain": MIN_AGGREGATE_KNOWN_GAIN,
            "fold_macro_direction": "nonnegative in every fold",
            "ood_and_binary_tolerance": INVARIANT_ATOL,
        },
    }


def change_summary(
    labels: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, int]:
    baseline_correct = baseline == labels
    candidate_correct = candidate == labels
    return {
        "rescued_errors": int(np.sum(~baseline_correct & candidate_correct)),
        "introduced_errors": int(np.sum(baseline_correct & ~candidate_correct)),
        "changed_predictions": int(np.sum(baseline != candidate)),
        "binary_changed": int(np.sum((baseline == 0) != (candidate == 0))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-root", type=Path, default=ROOT / "checkpoints"
    )
    parser.add_argument(
        "--raw-cache-dir",
        type=Path,
        default=ROOT / "data" / "experiments" / "campp_control_centroid_crossfit",
    )
    parser.add_argument(
        "--ema-cache-dir",
        type=Path,
        default=ROOT / "data" / "experiments" / "campp_raw_ema_lme20",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT / "reports" / "generated" / "campp_raw_ema_known_rerank_lme20.json"
        ),
    )
    args = parser.parse_args()

    raw_oofs, raw_artifacts, raw_metadata = load_fold_inputs(
        args.checkpoint_root, args.raw_cache_dir
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ema_arrays = []
    ema_metadata = []
    for fold in range(NUM_FOLDS):
        arrays, metadata = build_or_load_ema_cache(
            fold=fold,
            raw_artifact=raw_artifacts[fold],
            raw_metadata=raw_metadata[fold],
            raw_oof=raw_oofs[fold],
            checkpoint_root=args.checkpoint_root,
            cache_dir=args.ema_cache_dir,
            device=device,
            batch_size=48,
            num_workers=8,
        )
        if metadata.get("cache_status") != "reused":
            raise RuntimeError("Known reranker requires the preregistered EMA caches")
        ema_arrays.append(arrays)
        ema_metadata.append(metadata)

    fold_rows = []
    all_files = []
    all_labels = []
    all_baseline = []
    all_candidate = []
    for fold in range(NUM_FOLDS):
        raw_oof = raw_oofs[fold]
        raw_artifact = raw_artifacts[fold]
        ema = ema_arrays[fold]
        ema_artifact = {
            "train_embeddings": ema["train_embeddings"],
            "competition_labels": ema["competition_labels"],
            "unknown_cluster_ids": ema["unknown_cluster_ids"],
        }
        raw_evidence = probability_evidence(
            raw_artifact,
            raw_oof["embeddings"],
            raw_oof["competition_probs"],
        )
        ema_evidence = probability_evidence(
            ema_artifact,
            ema["validation_embeddings"],
            ema["validation_probabilities"],
        )
        _, raw_predictions = final_decision(*raw_evidence)
        ensemble_probabilities, _ = fixed_raw_ema_decision(
            raw_evidence, ema_evidence
        )
        candidate_predictions = binary_locked_known_rerank(
            raw_predictions, ensemble_probabilities
        )
        labels = raw_oof["labels"].astype(np.int64)
        baseline = metric_bundle(labels, raw_predictions)
        candidate = metric_bundle(labels, candidate_predictions)
        fold_rows.append({
            "fold": fold,
            "baseline": baseline,
            "candidate": candidate,
            "delta": metric_delta(candidate, baseline),
            **change_summary(labels, raw_predictions, candidate_predictions),
        })
        all_files.append(raw_oof["files"].astype(str))
        all_labels.append(labels)
        all_baseline.append(raw_predictions)
        all_candidate.append(candidate_predictions)

    files = np.concatenate(all_files)
    if len(set(files.tolist())) != len(files):
        raise RuntimeError("OOF validation files overlap across folds")
    labels = np.concatenate(all_labels)
    baseline_predictions = np.concatenate(all_baseline)
    candidate_predictions = np.concatenate(all_candidate)
    baseline = metric_bundle(labels, baseline_predictions)
    candidate = metric_bundle(labels, candidate_predictions)
    if abs(baseline["macro_f1"] - LOCKED_BASELINE_MACRO_F1) > 1e-10:
        raise RuntimeError("Locked Raw LME20 baseline reproduction failed")
    aggregate = {
        "baseline": baseline,
        "candidate": candidate,
        "delta": metric_delta(candidate, baseline),
        **change_summary(labels, baseline_predictions, candidate_predictions),
    }
    acceptance = gate(fold_rows, aggregate)
    report = {
        "contract": {
            "hypothesis": (
                "EMA identity evidence can reduce Raw wrong-known errors without "
                "altering any Raw known/unknown decision"
            ),
            "binary_decision_source": "locked Raw LME20",
            "known_rerank_source": "fixed 50/50 Raw/EMA LME20 probabilities",
            "candidate_classes": "known classes 1..446 only",
            "weights_selected": False,
            "thresholds_selected": False,
            "epochs_selected": False,
            "leaderboard_tuning": False,
        },
        "provenance": {
            "raw_cache_metadata": raw_metadata,
            "ema_cache_metadata": ema_metadata,
            "unique_oof_files": int(len(files)),
            "oof_file_sha256": digest_names(files),
        },
        "folds": fold_rows,
        "aggregate": aggregate,
        "acceptance_gate": acceptance,
        "decision": "accept" if acceptance["passed"] else "reject",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({
        "output": str(args.output),
        "decision": report["decision"],
        "fold_deltas": [row["delta"] for row in fold_rows],
        "aggregate": aggregate,
        "gate": acceptance,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
