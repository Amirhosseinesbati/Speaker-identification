"""Three-fold audit of binary-locked temporal known-speaker reranking.

The locked file-aggregate LME20 decision remains authoritative for whether a
row is unknown or known.  Only rows already accepted as known may change their
known identity, using the fixed flat temporal-view LME20 prototype evidence and
the existing head/prototype fusion.  This structurally prevents the harmful
known/unknown boundary shifts observed in both full multi-view replacements.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_control_oof_centroid_crossfit import (  # noqa: E402
    NUM_CLASSES,
    NUM_FOLDS,
    metric_bundle,
    metric_delta,
)
from scripts.analyze_lme20_asnorm_crossfit import (  # noqa: E402
    LOCKED_ALPHA,
    LOCKED_RAW_KAPPA,
    LOCKED_TAU,
    LOCKED_UNKNOWN_WEIGHT,
)
from scripts.analyze_prototype_aggregation_crossfit import group_indices  # noqa: E402
from scripts.analyze_unknown_cluster_hypotheses_crossfit import (  # noqa: E402
    load_fold_inputs,
)
from scripts.audit_multiview_lme20 import (  # noqa: E402
    build_or_load_view_cache,
    expand_enrollment_views,
    multiview_logmeanexp_scores,
)
from scripts.audit_short_audio_repeat import (  # noqa: E402
    LOCKED_BASELINE_MACRO_F1,
    LOCKED_LME_BETA,
    acceptance_gate,
    digest_names,
    locked_lme_predictions,
)
from submission.inference import _collapse_centroid_probs  # noqa: E402


def known_only_rerank_predictions(
    *,
    head: np.ndarray,
    temporal_scores: np.ndarray,
    baseline_predictions: np.ndarray,
) -> np.ndarray:
    """Preserve binary decisions and rerank only within known classes."""
    head = np.asarray(head, dtype=np.float64)
    temporal_scores = np.asarray(temporal_scores, dtype=np.float64)
    baseline = np.asarray(baseline_predictions, dtype=np.int64)
    internal = np.zeros(
        (len(temporal_scores), 1 + temporal_scores.shape[1]), dtype=np.float64
    )
    internal[:, 1:] = softmax(LOCKED_RAW_KAPPA * temporal_scores, axis=1)
    prototype = _collapse_centroid_probs(internal, NUM_CLASSES)
    fused = LOCKED_ALPHA * head + (1.0 - LOCKED_ALPHA) * prototype
    known_top = fused[:, 1:].argmax(axis=1).astype(np.int64) + 1
    predictions = baseline.copy()
    accepted_known = baseline > 0
    predictions[accepted_known] = known_top[accepted_known]
    if not np.array_equal(predictions == 0, baseline == 0):
        raise RuntimeError("Known-only reranker changed a locked binary decision")
    return predictions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-root", type=Path, default=ROOT / "checkpoints"
    )
    parser.add_argument(
        "--pad-cache-dir",
        type=Path,
        default=ROOT / "data" / "experiments" / "campp_control_centroid_crossfit",
    )
    parser.add_argument(
        "--view-cache-dir",
        type=Path,
        default=ROOT / "data" / "experiments" / "campp_multiview_lme20",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT / "reports" / "generated" / "campp_temporal_known_rerank_lme20.json"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--score-batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )

    oofs, artifacts, pad_metadata = load_fold_inputs(
        args.checkpoint_root, args.pad_cache_dir
    )
    caches = []
    cache_metadata = []
    for fold in range(NUM_FOLDS):
        cache, metadata = build_or_load_view_cache(
            fold=fold,
            pad_artifact=artifacts[fold],
            pad_metadata=pad_metadata[fold],
            pad_oof=oofs[fold],
            checkpoint_root=args.checkpoint_root,
            cache_dir=args.view_cache_dir,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        caches.append(cache)
        cache_metadata.append(metadata)

    fold_rows = []
    changed_rows = []
    all_files = []
    all_labels = []
    all_baseline = []
    all_candidate = []
    for fold in range(NUM_FOLDS):
        artifact = artifacts[fold]
        oof = oofs[fold]
        cache = caches[fold]
        enrollment_views, enrollment_group_ids, group_view_counts = (
            expand_enrollment_views(
                cache["train_view_embeddings"],
                cache["train_view_counts"],
                group_indices(artifact),
            )
        )
        scores = multiview_logmeanexp_scores(
            query_views=cache["validation_view_embeddings"],
            query_counts=cache["validation_view_counts"],
            enrollment_views=enrollment_views,
            enrollment_group_ids=enrollment_group_ids,
            group_view_counts=group_view_counts,
            beta=LOCKED_LME_BETA,
            device=device,
            file_batch_size=args.score_batch_size,
        )
        baseline_predictions = locked_lme_predictions(
            artifact, oof["embeddings"], oof["competition_probs"]
        )
        candidate_predictions = known_only_rerank_predictions(
            head=oof["competition_probs"],
            temporal_scores=scores,
            baseline_predictions=baseline_predictions,
        )
        labels = oof["labels"].astype(np.int64)
        baseline_metrics = metric_bundle(labels, baseline_predictions)
        candidate_metrics = metric_bundle(labels, candidate_predictions)
        if candidate_metrics["ood_f1"] != baseline_metrics["ood_f1"]:
            raise RuntimeError("Binary lock failed to preserve Fold OOD-F1")
        baseline_correct = baseline_predictions == labels
        candidate_correct = candidate_predictions == labels
        fold_rows.append({
            "fold": fold,
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
            "delta": metric_delta(candidate_metrics, baseline_metrics),
            "rescued_errors": int(np.sum(~baseline_correct & candidate_correct)),
            "introduced_errors": int(np.sum(baseline_correct & ~candidate_correct)),
            "changed_predictions": int(np.sum(
                baseline_predictions != candidate_predictions
            )),
            "binary_decisions_changed": 0,
        })
        for index in np.flatnonzero(
            baseline_predictions != candidate_predictions
        ):
            changed_rows.append({
                "fold": int(fold),
                "file": str(oof["files"][index]),
                "label": int(labels[index]),
                "baseline_prediction": int(baseline_predictions[index]),
                "candidate_prediction": int(candidate_predictions[index]),
                "baseline_correct": bool(baseline_correct[index]),
                "candidate_correct": bool(candidate_correct[index]),
                "query_real_views": int(cache["validation_view_counts"][index]),
            })
        all_files.append(oof["files"].astype(str))
        all_labels.append(labels)
        all_baseline.append(baseline_predictions)
        all_candidate.append(candidate_predictions)

    files = np.concatenate(all_files)
    if len(set(files.tolist())) != len(files):
        raise RuntimeError("OOF validation files overlap across folds")
    labels = np.concatenate(all_labels)
    baseline_predictions = np.concatenate(all_baseline)
    candidate_predictions = np.concatenate(all_candidate)
    if not np.array_equal(candidate_predictions == 0, baseline_predictions == 0):
        raise RuntimeError("Aggregate binary lock failed")
    baseline_metrics = metric_bundle(labels, baseline_predictions)
    candidate_metrics = metric_bundle(labels, candidate_predictions)
    baseline_correct = baseline_predictions == labels
    candidate_correct = candidate_predictions == labels
    aggregate = {
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "delta": metric_delta(candidate_metrics, baseline_metrics),
        "rescued_errors": int(np.sum(~baseline_correct & candidate_correct)),
        "introduced_errors": int(np.sum(baseline_correct & ~candidate_correct)),
        "changed_predictions": int(np.sum(
            baseline_predictions != candidate_predictions
        )),
        "binary_decisions_changed": 0,
    }
    if abs(baseline_metrics["macro_f1"] - LOCKED_BASELINE_MACRO_F1) > 1e-10:
        raise RuntimeError("Locked baseline reproduction failed")
    if candidate_metrics["ood_f1"] != baseline_metrics["ood_f1"]:
        raise RuntimeError("Aggregate OOD-F1 changed despite binary lock")
    gate = acceptance_gate(fold_rows, aggregate)
    report = {
        "contract": {
            "hypothesis": (
                "temporal views improve known identity reranking without changing OOD"
            ),
            "binary_decision_source": "locked file-aggregate LME20 baseline",
            "known_rerank_source": "fixed flat temporal-view LME20 head fusion",
            "lme_beta": LOCKED_LME_BETA,
            "alpha": LOCKED_ALPHA,
            "kappa": LOCKED_RAW_KAPPA,
            "baseline_tau": LOCKED_TAU,
            "baseline_lambda_unknown": LOCKED_UNKNOWN_WEIGHT,
            "parameter_selection": "none",
            "leaderboard_tuning": False,
        },
        "provenance": {
            "pad_cache_metadata": pad_metadata,
            "view_cache_metadata": cache_metadata,
            "unique_oof_files": int(len(files)),
            "oof_file_sha256": digest_names(files),
        },
        "folds": fold_rows,
        "aggregate": aggregate,
        "changed_rows": changed_rows,
        "acceptance_gate": gate,
        "decision": "accept" if gate["passed"] else "reject",
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
        "gate": gate,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
