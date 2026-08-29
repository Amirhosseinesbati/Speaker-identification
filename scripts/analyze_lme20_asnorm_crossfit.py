"""Leak-free adaptive symmetric score-normalisation audit for CAM++ LME-20.

This experiment starts from the already locked LME-20 policy and changes only
the prototype score normalisation.  For every target OOF fold, the cohort size
and normalised-softmax scale are selected on the other two folds.  The target
fold therefore never chooses its own parameters.

The cohort is the matching fold's training embeddings.  Z statistics exclude
all enrollment rows belonging to the scored group, while T statistics use the
top scoring prototype groups for each query.  The locked raw LME maximum is
retained for the OOD threshold so this audit does not silently introduce a new
threshold family.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import logsumexp, softmax

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_control_oof_centroid_crossfit import (  # noqa: E402
    NUM_CLASSES,
    NUM_FOLDS,
    NUM_KNOWN,
    l2norm_rows,
    metric_bundle,
    metric_delta,
)
from scripts.analyze_prototype_aggregation_crossfit import (  # noqa: E402
    group_indices,
)
from scripts.analyze_unknown_cluster_hypotheses_crossfit import (  # noqa: E402
    load_fold_inputs,
)
from submission.inference import _collapse_centroid_probs  # noqa: E402


LME_BETA = 20.0
TOP_NS = (50, 100, 200)
NORMALISED_KAPPAS = (0.5, 1.0, 2.0, 4.0)
LOCKED_ALPHA = 0.15
LOCKED_RAW_KAPPA = 16.0
LOCKED_TAU = 0.50
LOCKED_UNKNOWN_WEIGHT = 0.75
EPSILON = 1e-6


@dataclass
class FoldScores:
    fold: int
    files: np.ndarray
    labels: np.ndarray
    head: np.ndarray
    raw_scores: np.ndarray
    baseline_predictions: np.ndarray
    candidate_predictions: dict[tuple[int, float], np.ndarray]
    diagnostics: dict


def logmeanexp_group_scores(
    queries: np.ndarray,
    enrollments: np.ndarray,
    groups: list[np.ndarray],
    beta: float = LME_BETA,
) -> np.ndarray:
    """Return normalised log-sum-exp scores from queries to enrollment groups."""
    queries = l2norm_rows(queries)
    enrollments = l2norm_rows(enrollments)
    similarities = queries @ enrollments.T
    scores = np.empty((len(queries), len(groups)), dtype=np.float32)
    for group_id, indices in enumerate(groups):
        values = similarities[:, indices]
        scores[:, group_id] = (
            logsumexp(float(beta) * values, axis=1)
            - np.log(values.shape[1])
        ) / float(beta)
    return scores


def enrollment_group_ids(
    artifact: dict[str, np.ndarray], groups: list[np.ndarray]
) -> np.ndarray:
    """Map every enrollment row to its 0-based known/pseudo-unknown group."""
    labels = artifact["competition_labels"].astype(np.int64)
    clusters = artifact["unknown_cluster_ids"].astype(np.int64)
    result = np.where(labels > 0, labels - 1, NUM_KNOWN + clusters)
    if np.any(result < 0) or int(result.max()) + 1 != len(groups):
        raise RuntimeError("Enrollment group ids are not dense and complete")
    for group_id, indices in enumerate(groups):
        if not np.all(result[indices] == group_id):
            raise RuntimeError(f"Enrollment group {group_id} membership mismatch")
    return result.astype(np.int64)


def cohort_z_statistics(
    cohort_scores: np.ndarray,
    cohort_group_ids: np.ndarray,
    top_n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute group-wise adaptive Z statistics excluding same-group rows."""
    cohort_scores = np.asarray(cohort_scores, dtype=np.float64)
    cohort_group_ids = np.asarray(cohort_group_ids, dtype=np.int64)
    means = np.empty(cohort_scores.shape[1], dtype=np.float64)
    stds = np.empty_like(means)
    for group_id in range(cohort_scores.shape[1]):
        impostors = cohort_scores[cohort_group_ids != group_id, group_id]
        count = min(int(top_n), len(impostors))
        if count < 2:
            raise RuntimeError(f"Insufficient impostor cohort for group {group_id}")
        selected = np.partition(impostors, len(impostors) - count)[-count:]
        means[group_id] = selected.mean()
        stds[group_id] = max(float(selected.std()), EPSILON)
    return means, stds


def adaptive_symmetric_normalise(
    query_scores: np.ndarray,
    z_means: np.ndarray,
    z_stds: np.ndarray,
    top_n: int,
) -> tuple[np.ndarray, dict]:
    """Apply 0.5 * (adaptive T-norm + adaptive Z-norm)."""
    scores = np.asarray(query_scores, dtype=np.float64)
    count = min(int(top_n), scores.shape[1])
    if count < 2:
        raise ValueError("top_n must select at least two prototype groups")
    selected = np.partition(scores, scores.shape[1] - count, axis=1)[:, -count:]
    t_means = selected.mean(axis=1)
    t_stds = np.maximum(selected.std(axis=1), EPSILON)
    t_normalised = (scores - t_means[:, None]) / t_stds[:, None]
    z_normalised = (scores - z_means[None, :]) / z_stds[None, :]
    normalised = 0.5 * (t_normalised + z_normalised)
    if not np.all(np.isfinite(normalised)):
        raise RuntimeError("AS-Norm produced non-finite scores")
    return normalised, {
        "top_n": count,
        "t_mean_min": float(t_means.min()),
        "t_mean_max": float(t_means.max()),
        "t_std_min": float(t_stds.min()),
        "t_std_median": float(np.median(t_stds)),
        "t_std_max": float(t_stds.max()),
        "z_mean_min": float(z_means.min()),
        "z_mean_max": float(z_means.max()),
        "z_std_min": float(z_stds.min()),
        "z_std_median": float(np.median(z_stds)),
        "z_std_max": float(z_stds.max()),
        "normalised_min": float(normalised.min()),
        "normalised_max": float(normalised.max()),
    }


def decision_predictions(
    *,
    head: np.ndarray,
    scores: np.ndarray,
    probability_kappa: float,
    raw_max_scores: np.ndarray,
) -> np.ndarray:
    internal = np.zeros((len(scores), 1 + scores.shape[1]), dtype=np.float64)
    internal[:, 1:] = softmax(float(probability_kappa) * scores, axis=1)
    prototype = _collapse_centroid_probs(internal, NUM_CLASSES)
    fused = LOCKED_ALPHA * head + (1.0 - LOCKED_ALPHA) * prototype
    fused[:, 0] *= LOCKED_UNKNOWN_WEIGHT
    fused /= fused.sum(axis=1, keepdims=True) + 1e-12
    predictions = fused.argmax(axis=1).astype(np.int64)
    predictions[raw_max_scores < LOCKED_TAU] = 0
    return predictions


def prepare_fold_scores(
    *, fold: int, artifact: dict[str, np.ndarray], oof: dict
) -> FoldScores:
    train = l2norm_rows(artifact["train_embeddings"])
    validation = l2norm_rows(oof["embeddings"])
    groups = group_indices(artifact)
    group_ids = enrollment_group_ids(artifact, groups)

    raw_query_scores = logmeanexp_group_scores(validation, train, groups)
    raw_cohort_scores = logmeanexp_group_scores(train, train, groups)
    raw_max = raw_query_scores.max(axis=1).astype(np.float64)
    head = oof["competition_probs"].astype(np.float64)
    baseline = decision_predictions(
        head=head,
        scores=raw_query_scores,
        probability_kappa=LOCKED_RAW_KAPPA,
        raw_max_scores=raw_max,
    )

    candidates: dict[tuple[int, float], np.ndarray] = {}
    diagnostics = {
        "train_files": int(len(train)),
        "validation_files": int(len(validation)),
        "groups": int(len(groups)),
        "raw_score_min": float(raw_query_scores.min()),
        "raw_score_max": float(raw_query_scores.max()),
        "cohort_same_group_excluded": True,
        "top_n": {},
    }
    for top_n in TOP_NS:
        z_means, z_stds = cohort_z_statistics(
            raw_cohort_scores, group_ids, top_n
        )
        normalised, normalisation_diagnostics = adaptive_symmetric_normalise(
            raw_query_scores, z_means, z_stds, top_n
        )
        diagnostics["top_n"][str(top_n)] = normalisation_diagnostics
        for kappa in NORMALISED_KAPPAS:
            candidates[(top_n, float(kappa))] = decision_predictions(
                head=head,
                scores=normalised,
                probability_kappa=kappa,
                raw_max_scores=raw_max,
            )
    return FoldScores(
        fold=fold,
        files=oof["files"].astype(str),
        labels=oof["labels"].astype(np.int64),
        head=head,
        raw_scores=raw_query_scores,
        baseline_predictions=baseline,
        candidate_predictions=candidates,
        diagnostics=diagnostics,
    )


def candidate_metrics(
    fold: FoldScores, candidate: tuple[int, float]
) -> tuple[dict, dict, dict]:
    baseline = metric_bundle(fold.labels, fold.baseline_predictions)
    metrics = metric_bundle(fold.labels, fold.candidate_predictions[candidate])
    return baseline, metrics, metric_delta(metrics, baseline)


def select_candidate(
    folds: list[FoldScores], calibration_folds: tuple[int, int]
) -> tuple[tuple[int, float], dict]:
    ranked = []
    for top_n, kappa in itertools.product(TOP_NS, NORMALISED_KAPPAS):
        key = (int(top_n), float(kappa))
        rows = []
        for fold_id in calibration_folds:
            baseline, metrics, delta = candidate_metrics(folds[fold_id], key)
            rows.append({
                "fold": fold_id,
                "baseline": baseline,
                "candidate": metrics,
                "delta": delta,
            })
        macro = [row["delta"]["macro_f1"] for row in rows]
        feasible = all(
            row["delta"]["macro_f1"] > 0.0
            and row["delta"]["known_accuracy"] >= -0.001
            and row["delta"]["ood_f1"] >= -0.001
            for row in rows
        )
        distance = abs(np.log2(kappa)) + abs(np.log2(top_n / 100.0))
        rank = (int(feasible), min(macro), float(np.mean(macro)), -distance)
        ranked.append((rank, key, rows, feasible))
    rank, key, rows, feasible = max(ranked, key=lambda item: item[0])
    return key, {
        "calibration_folds": list(calibration_folds),
        "selection_objective": (
            "prefer candidates passing positive Macro-F1 and -0.001 Known/OOD "
            "guardrails on both calibration folds, then maximise minimum and "
            "mean calibration-fold Macro-F1 gain"
        ),
        "calibration_feasible": bool(feasible),
        "rank_tuple": [float(value) for value in rank],
        "per_fold": rows,
    }


def aggregate_evaluation(
    folds: list[FoldScores], predictions: list[np.ndarray]
) -> dict:
    files = np.concatenate([fold.files for fold in folds])
    if len(set(files.tolist())) != len(files):
        raise RuntimeError("OOF files overlap across folds")
    labels = np.concatenate([fold.labels for fold in folds])
    baseline_predictions = np.concatenate([
        fold.baseline_predictions for fold in folds
    ])
    candidate_predictions = np.concatenate(predictions)
    baseline = metric_bundle(labels, baseline_predictions)
    candidate = metric_bundle(labels, candidate_predictions)
    baseline_correct = labels == baseline_predictions
    candidate_correct = labels == candidate_predictions
    return {
        "baseline": baseline,
        "candidate": candidate,
        "delta": metric_delta(candidate, baseline),
        "rescued_errors": int(np.sum(~baseline_correct & candidate_correct)),
        "introduced_errors": int(np.sum(baseline_correct & ~candidate_correct)),
        "baseline_errors": int(np.sum(~baseline_correct)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-root", type=Path, default=ROOT / "checkpoints"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "data" / "experiments" / "campp_control_centroid_crossfit",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "generated" / "campp_lme20_asnorm_crossfit.json",
    )
    args = parser.parse_args()

    oofs, artifacts, metadata = load_fold_inputs(args.checkpoint_root, args.cache_dir)
    folds = [
        prepare_fold_scores(fold=fold, artifact=artifacts[fold], oof=oofs[fold])
        for fold in range(NUM_FOLDS)
    ]

    selections = []
    predictions = []
    for target in range(NUM_FOLDS):
        calibration = tuple(fold for fold in range(NUM_FOLDS) if fold != target)
        key, selection = select_candidate(
            folds, calibration  # type: ignore[arg-type]
        )
        baseline, metrics, delta = candidate_metrics(folds[target], key)
        selections.append({
            "target_fold": target,
            "parameters": {"top_n": key[0], "normalised_kappa": key[1]},
            "calibration": selection,
            "held_out": {
                "baseline": baseline,
                "candidate": metrics,
                "delta": delta,
            },
        })
        predictions.append(folds[target].candidate_predictions[key])

    aggregate = aggregate_evaluation(folds, predictions)
    baseline_macro = aggregate["baseline"]["macro_f1"]
    if abs(baseline_macro - 0.9633564052154656) > 1e-12:
        raise RuntimeError(
            f"Locked LME-20 baseline mismatch: {baseline_macro:.16f}"
        )
    per_fold_pass = [
        row["held_out"]["delta"]["macro_f1"] > 0.0
        and row["held_out"]["delta"]["known_accuracy"] >= -0.001
        and row["held_out"]["delta"]["ood_f1"] >= -0.001
        for row in selections
    ]
    gate = {
        "minimum_aggregate_macro_gain": 0.001,
        "maximum_known_accuracy_drop": 0.001,
        "maximum_ood_f1_drop": 0.001,
        "requires_all_held_out_folds_positive_with_guardrails": True,
        "per_fold_pass": per_fold_pass,
        "passed": bool(
            all(per_fold_pass)
            and aggregate["delta"]["macro_f1"] >= 0.001
            and aggregate["delta"]["known_accuracy"] >= -0.001
            and aggregate["delta"]["ood_f1"] >= -0.001
        ),
    }

    fixed_diagnostics = []
    for key in itertools.product(TOP_NS, NORMALISED_KAPPAS):
        candidate = (int(key[0]), float(key[1]))
        evaluation = aggregate_evaluation(
            folds, [fold.candidate_predictions[candidate] for fold in folds]
        )
        fixed_diagnostics.append({
            "parameters": {
                "top_n": candidate[0],
                "normalised_kappa": candidate[1],
            },
            "aggregate": evaluation,
        })
    fixed_diagnostics.sort(
        key=lambda row: row["aggregate"]["candidate"]["macro_f1"],
        reverse=True,
    )

    report = {
        "contract": {
            "baseline": "locked CAM++ LME-20 fixed policy",
            "lme_beta": LME_BETA,
            "locked_decision": {
                "alpha": LOCKED_ALPHA,
                "raw_kappa": LOCKED_RAW_KAPPA,
                "tau": LOCKED_TAU,
                "lambda_unknown": LOCKED_UNKNOWN_WEIGHT,
            },
            "candidate_top_n": list(TOP_NS),
            "candidate_normalised_kappa": list(NORMALISED_KAPPAS),
            "target_fold_selection": "leave-one-fold-out; target never selects itself",
            "z_cohort": "matching train fold; same prototype group excluded",
            "ood_threshold_score": "locked raw LME-20 max score",
            "leaderboard_used": False,
        },
        "provenance": {
            "cache_metadata": metadata,
            "fold_diagnostics": [fold.diagnostics for fold in folds],
        },
        "crossfit": {
            "selections": selections,
            "aggregate": aggregate,
            "gate": gate,
        },
        "all_oof_fixed_candidate_diagnostics_non_decisional": fixed_diagnostics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "selections": [
            {
                "target_fold": row["target_fold"],
                "parameters": row["parameters"],
                "held_out_delta": row["held_out"]["delta"],
            }
            for row in selections
        ],
        "aggregate": aggregate,
        "gate": gate,
        "best_fixed_diagnostic": fixed_diagnostics[0],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
