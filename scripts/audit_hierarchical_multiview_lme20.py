"""Three-fold audit of equal-file hierarchical temporal-view LME20.

This candidate retains real temporal views but removes the length weighting of
the rejected flat view-pair audit.  With the already locked beta=20 it applies
normalised LME in three fixed levels: enrollment views within each enrollment
file, enrollment files within each speaker group, then query views within the
query file.  Thus every enrollment file has equal total mass regardless of its
number of real views.  No decision-layer parameter changes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_control_oof_centroid_crossfit import (
    NUM_FOLDS,
    metric_bundle,
    metric_delta,
)
from scripts.analyze_lme20_asnorm_crossfit import (
    LOCKED_ALPHA,
    LOCKED_RAW_KAPPA,
    LOCKED_TAU,
    LOCKED_UNKNOWN_WEIGHT,
    decision_predictions,
    enrollment_group_ids,
)
from scripts.analyze_prototype_aggregation_crossfit import group_indices
from scripts.analyze_unknown_cluster_hypotheses_crossfit import load_fold_inputs
from scripts.audit_multiview_lme20 import build_or_load_view_cache
from scripts.audit_short_audio_repeat import (
    LOCKED_BASELINE_MACRO_F1,
    LOCKED_LME_BETA,
    acceptance_gate,
    digest_names,
    locked_lme_predictions,
)

def flatten_valid_enrollment_views(
    view_embeddings: np.ndarray,
    view_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten only real views and return their dense enrollment-file ids."""
    chunks: list[np.ndarray] = []
    file_ids: list[np.ndarray] = []
    for file_id, count in enumerate(np.asarray(view_counts, dtype=np.int64)):
        count = int(count)
        if count < 1 or count > view_embeddings.shape[1]:
            raise RuntimeError(f"Invalid enrollment view count for file {file_id}")
        chunks.append(view_embeddings[file_id, :count])
        file_ids.append(np.full(count, file_id, dtype=np.int64))
    return (
        np.concatenate(chunks, axis=0).astype(np.float32),
        np.concatenate(file_ids).astype(np.int64),
    )


def _grouped_lme(
    values: torch.Tensor,
    ids: torch.Tensor,
    counts: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Row-wise normalised LME over dense column groups."""
    num_groups = int(len(counts))
    index = ids.unsqueeze(0).expand(values.shape[0], -1)
    scaled = float(beta) * values
    maxima = torch.full(
        (values.shape[0], num_groups),
        -torch.inf,
        dtype=values.dtype,
        device=values.device,
    )
    maxima.scatter_reduce_(1, index, scaled, reduce="amax", include_self=True)
    stable = torch.exp(scaled - maxima.gather(1, index))
    sums = torch.zeros_like(maxima)
    sums.scatter_add_(1, index, stable)
    return (
        maxima + torch.log(sums) - torch.log(counts.to(values.dtype))[None, :]
    ) / float(beta)


def hierarchical_logmeanexp_scores(
    *,
    query_views: np.ndarray,
    query_counts: np.ndarray,
    enrollment_views: np.ndarray,
    enrollment_view_file_ids: np.ndarray,
    enrollment_file_view_counts: np.ndarray,
    enrollment_file_group_ids: np.ndarray,
    group_file_counts: np.ndarray,
    beta: float,
    device: torch.device,
    file_batch_size: int,
) -> np.ndarray:
    """LME over enrollment views -> enrollment files -> query views."""
    query = torch.from_numpy(np.asarray(query_views, np.float32)).to(device)
    enroll = torch.from_numpy(np.asarray(enrollment_views, np.float32)).to(device)
    view_file_ids = torch.from_numpy(
        np.asarray(enrollment_view_file_ids, np.int64)
    ).to(device)
    file_view_counts = torch.from_numpy(
        np.asarray(enrollment_file_view_counts, np.float32)
    ).to(device)
    file_group_ids = torch.from_numpy(
        np.asarray(enrollment_file_group_ids, np.int64)
    ).to(device)
    group_counts = torch.from_numpy(
        np.asarray(group_file_counts, np.float32)
    ).to(device)
    if np.any(np.asarray(enrollment_file_view_counts) <= 0):
        raise RuntimeError("An enrollment file has no real views")
    if np.any(np.asarray(group_file_counts) <= 0):
        raise RuntimeError("A speaker group has no enrollment files")

    max_query_views = int(query.shape[1])
    num_groups = int(len(group_file_counts))
    result: list[np.ndarray] = []
    with torch.inference_mode():
        for start in tqdm(
            range(0, len(query), int(file_batch_size)),
            desc="Hierarchical view LME scores",
        ):
            stop = min(len(query), start + int(file_batch_size))
            batch = query[start:stop]
            flat_query = batch.reshape(-1, batch.shape[-1])
            similarities = flat_query @ enroll.T
            per_file = _grouped_lme(
                similarities,
                view_file_ids,
                file_view_counts,
                beta,
            )
            per_group = _grouped_lme(
                per_file,
                file_group_ids,
                group_counts,
                beta,
            ).reshape(len(batch), max_query_views, num_groups)
            counts = torch.from_numpy(
                np.asarray(query_counts[start:stop], dtype=np.int64)
            ).to(device)
            valid = (
                torch.arange(max_query_views, device=device)[None, :]
                < counts[:, None]
            )
            scaled = (float(beta) * per_group).masked_fill(
                ~valid[:, :, None], -torch.inf
            )
            scores = (
                torch.logsumexp(scaled, dim=1)
                - torch.log(counts.to(scaled.dtype))[:, None]
            ) / float(beta)
            result.append(scores.cpu().numpy())
    output = np.concatenate(result, axis=0).astype(np.float32)
    if not np.all(np.isfinite(output)):
        raise RuntimeError("Hierarchical view LME produced non-finite scores")
    return output


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
            ROOT / "reports" / "generated" / "campp_hierarchical_multiview_lme20.json"
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

    pad_oofs, pad_artifacts, pad_metadata = load_fold_inputs(
        args.checkpoint_root, args.pad_cache_dir
    )
    caches = []
    cache_metadata = []
    for fold in range(NUM_FOLDS):
        cache, metadata = build_or_load_view_cache(
            fold=fold,
            pad_artifact=pad_artifacts[fold],
            pad_metadata=pad_metadata[fold],
            pad_oof=pad_oofs[fold],
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
        artifact = pad_artifacts[fold]
        oof = pad_oofs[fold]
        cache = caches[fold]
        groups = group_indices(artifact)
        file_group_ids = enrollment_group_ids(artifact, groups)
        group_file_counts = np.bincount(
            file_group_ids, minlength=len(groups)
        ).astype(np.int64)
        enrollment_views, view_file_ids = flatten_valid_enrollment_views(
            cache["train_view_embeddings"], cache["train_view_counts"]
        )
        scores = hierarchical_logmeanexp_scores(
            query_views=cache["validation_view_embeddings"],
            query_counts=cache["validation_view_counts"],
            enrollment_views=enrollment_views,
            enrollment_view_file_ids=view_file_ids,
            enrollment_file_view_counts=cache["train_view_counts"],
            enrollment_file_group_ids=file_group_ids,
            group_file_counts=group_file_counts,
            beta=LOCKED_LME_BETA,
            device=device,
            file_batch_size=args.score_batch_size,
        )
        baseline_predictions = locked_lme_predictions(
            artifact, oof["embeddings"], oof["competition_probs"]
        )
        candidate_predictions = decision_predictions(
            head=oof["competition_probs"],
            scores=scores,
            probability_kappa=LOCKED_RAW_KAPPA,
            raw_max_scores=scores.max(axis=1),
        )
        labels = oof["labels"].astype(np.int64)
        baseline_metrics = metric_bundle(labels, baseline_predictions)
        candidate_metrics = metric_bundle(labels, candidate_predictions)
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
            "enrollment_files": int(len(file_group_ids)),
            "enrollment_real_views": int(len(enrollment_views)),
            "query_real_views": int(cache["validation_view_counts"].sum()),
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
    }
    if abs(baseline_metrics["macro_f1"] - LOCKED_BASELINE_MACRO_F1) > 1e-10:
        raise RuntimeError("Locked baseline reproduction failed")
    gate = acceptance_gate(fold_rows, aggregate)
    report = {
        "contract": {
            "hypothesis": (
                "retain real views while assigning equal total mass per enrollment file"
            ),
            "aggregation": (
                "LME20 enrollment views -> LME20 enrollment files -> LME20 query views"
            ),
            "weights": "fixed selected Raw CAM++ Control Fold0/1/2 checkpoints",
            "folds": "fixed kfold/folds3/seed42 OOF",
            "lme_beta_all_levels": LOCKED_LME_BETA,
            "alpha": LOCKED_ALPHA,
            "kappa": LOCKED_RAW_KAPPA,
            "tau": LOCKED_TAU,
            "lambda_unknown": LOCKED_UNKNOWN_WEIGHT,
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
