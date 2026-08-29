"""Cross-fit shrinkage-LDA backend for the locked CAM++ LME-20 rule.

The LDA transform is estimated from only the 446 ground-truth known speakers
in each matching training fold.  Unknown enrollment and held-out embeddings
are transformed by that train-only backend before the unchanged LME-20 and
decision rule.  Projection dimensionality for a target fold is selected only
on the other two folds.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_control_oof_centroid_crossfit import (  # noqa: E402
    NUM_FOLDS,
    NUM_KNOWN,
    l2norm_rows,
    metric_bundle,
    metric_delta,
)
from scripts.analyze_lme20_asnorm_crossfit import (  # noqa: E402
    LME_BETA,
    LOCKED_ALPHA,
    LOCKED_RAW_KAPPA,
    LOCKED_TAU,
    LOCKED_UNKNOWN_WEIGHT,
    logmeanexp_group_scores,
)
from scripts.analyze_lme20_wccn_crossfit import (  # noqa: E402
    fixed_lme_predictions,
    within_group_covariance,
)
from scripts.analyze_prototype_aggregation_crossfit import (  # noqa: E402
    group_indices,
)
from scripts.analyze_unknown_cluster_hypotheses_crossfit import (  # noqa: E402
    load_fold_inputs,
)


PROJECTION_DIMS = (64, 128, 160)
WITHIN_SHRINKAGE = 0.10


@dataclass
class FoldLda:
    fold: int
    files: np.ndarray
    labels: np.ndarray
    baseline_predictions: np.ndarray
    candidate_predictions: dict[int, np.ndarray]
    diagnostics: dict


def shrinkage_lda_transform(
    embeddings: np.ndarray,
    groups: list[np.ndarray],
    projection_dims: int,
    within_shrinkage: float = WITHIN_SHRINKAGE,
) -> tuple[np.ndarray, np.ndarray, dict]:
    embeddings = l2norm_rows(embeddings).astype(np.float64)
    if not 0 < int(projection_dims) < embeddings.shape[1]:
        raise ValueError("projection_dims must be between 1 and dimension - 1")
    if not 0.0 < float(within_shrinkage) <= 1.0:
        raise ValueError("within_shrinkage must be in (0, 1]")
    selected = np.concatenate(groups)
    if len(set(selected.tolist())) != len(selected):
        raise RuntimeError("LDA groups overlap")
    known_embeddings = embeddings[selected]
    global_mean = known_embeddings.mean(axis=0)
    within, degrees_of_freedom = within_group_covariance(embeddings, groups)
    dimension = embeddings.shape[1]
    trace_scale = float(np.trace(within) / dimension)
    regularised_within = (
        (1.0 - float(within_shrinkage)) * within
        + float(within_shrinkage) * trace_scale * np.eye(dimension)
    )
    between = np.zeros((dimension, dimension), dtype=np.float64)
    for indices in groups:
        offset = embeddings[indices].mean(axis=0) - global_mean
        between += len(indices) * np.outer(offset, offset)
    between /= float(len(known_embeddings))
    between = 0.5 * (between + between.T)
    eigenvalues, eigenvectors = eigh(
        between, regularised_within, check_finite=True
    )
    transform = eigenvectors[:, -int(projection_dims):]
    selected_eigenvalues = eigenvalues[-int(projection_dims):]
    return global_mean, transform, {
        "projection_dims": int(projection_dims),
        "input_dims": int(dimension),
        "known_groups": int(len(groups)),
        "known_files": int(len(known_embeddings)),
        "within_degrees_of_freedom": int(degrees_of_freedom),
        "within_shrinkage": float(within_shrinkage),
        "within_trace_scale": trace_scale,
        "generalised_eigenvalue_min": float(eigenvalues.min()),
        "generalised_eigenvalue_median": float(np.median(eigenvalues)),
        "generalised_eigenvalue_max": float(eigenvalues.max()),
        "selected_eigenvalue_min": float(selected_eigenvalues.min()),
        "selected_eigenvalue_max": float(selected_eigenvalues.max()),
    }


def prepare_fold(
    *, fold: int, artifact: dict[str, np.ndarray], oof: dict
) -> FoldLda:
    train = l2norm_rows(artifact["train_embeddings"])
    validation = l2norm_rows(oof["embeddings"])
    groups = group_indices(artifact)
    known_groups = groups[:NUM_KNOWN]
    head = oof["competition_probs"].astype(np.float64)
    raw_scores = logmeanexp_group_scores(validation, train, groups)
    baseline = fixed_lme_predictions(head=head, raw_scores=raw_scores)

    candidates = {}
    diagnostics = {
        "train_files": int(len(train)),
        "validation_files": int(len(validation)),
        "lme_groups": int(len(groups)),
        "lda_groups": int(len(known_groups)),
        "projection_dims": {},
    }
    for projection_dims in PROJECTION_DIMS:
        mean, transform, transform_diagnostics = shrinkage_lda_transform(
            train, known_groups, projection_dims
        )
        transformed_train = l2norm_rows((train - mean) @ transform)
        transformed_validation = l2norm_rows((validation - mean) @ transform)
        transformed_scores = logmeanexp_group_scores(
            transformed_validation, transformed_train, groups
        )
        candidates[int(projection_dims)] = fixed_lme_predictions(
            head=head, raw_scores=transformed_scores
        )
        diagnostics["projection_dims"][str(projection_dims)] = {
            **transform_diagnostics,
            "score_min": float(transformed_scores.min()),
            "score_max": float(transformed_scores.max()),
        }
    return FoldLda(
        fold=fold,
        files=oof["files"].astype(str),
        labels=oof["labels"].astype(np.int64),
        baseline_predictions=baseline,
        candidate_predictions=candidates,
        diagnostics=diagnostics,
    )


def fold_metrics(
    fold: FoldLda, projection_dims: int
) -> tuple[dict, dict, dict]:
    baseline = metric_bundle(fold.labels, fold.baseline_predictions)
    candidate = metric_bundle(
        fold.labels, fold.candidate_predictions[projection_dims]
    )
    return baseline, candidate, metric_delta(candidate, baseline)


def select_projection_dims(
    folds: list[FoldLda], calibration_folds: tuple[int, int]
) -> tuple[int, dict]:
    ranked = []
    for projection_dims in PROJECTION_DIMS:
        rows = []
        for fold_id in calibration_folds:
            baseline, candidate, delta = fold_metrics(
                folds[fold_id], projection_dims
            )
            rows.append({
                "fold": fold_id,
                "baseline": baseline,
                "candidate": candidate,
                "delta": delta,
            })
        gains = [row["delta"]["macro_f1"] for row in rows]
        feasible = all(
            row["delta"]["macro_f1"] > 0.0
            and row["delta"]["known_accuracy"] >= -0.001
            and row["delta"]["ood_f1"] >= -0.001
            for row in rows
        )
        rank = (
            int(feasible),
            min(gains),
            float(np.mean(gains)),
            -abs(projection_dims - 128),
        )
        ranked.append((rank, int(projection_dims), rows, feasible))
    rank, projection_dims, rows, feasible = max(
        ranked, key=lambda item: item[0]
    )
    return projection_dims, {
        "calibration_folds": list(calibration_folds),
        "calibration_feasible": bool(feasible),
        "selection_objective": (
            "prefer positive Macro-F1 with -0.001 Known/OOD guardrails on both "
            "calibration folds, then maximise minimum and mean Macro-F1 gain"
        ),
        "rank_tuple": [float(value) for value in rank],
        "per_fold": rows,
    }


def aggregate_evaluation(
    folds: list[FoldLda], predictions: list[np.ndarray]
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
        "rescued_baseline_errors": int(
            np.sum(~baseline_correct & candidate_correct)
        ),
        "introduced_errors": int(np.sum(baseline_correct & ~candidate_correct)),
        "changed_predictions": int(
            np.sum(baseline_predictions != candidate_predictions)
        ),
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
        default=ROOT / "reports" / "generated" / "campp_lme20_lda_crossfit.json",
    )
    args = parser.parse_args()

    oofs, artifacts, metadata = load_fold_inputs(args.checkpoint_root, args.cache_dir)
    folds = [
        prepare_fold(fold=fold, artifact=artifacts[fold], oof=oofs[fold])
        for fold in range(NUM_FOLDS)
    ]
    selections = []
    predictions = []
    for target in range(NUM_FOLDS):
        calibration = tuple(fold for fold in range(NUM_FOLDS) if fold != target)
        projection_dims, selection = select_projection_dims(
            folds, calibration  # type: ignore[arg-type]
        )
        baseline, candidate, delta = fold_metrics(
            folds[target], projection_dims
        )
        selections.append({
            "target_fold": target,
            "parameters": {"projection_dims": projection_dims},
            "calibration": selection,
            "held_out": {
                "baseline": baseline,
                "candidate": candidate,
                "delta": delta,
            },
        })
        predictions.append(folds[target].candidate_predictions[projection_dims])

    aggregate = aggregate_evaluation(folds, predictions)
    if abs(aggregate["baseline"]["macro_f1"] - 0.9633564052154656) > 1e-12:
        raise RuntimeError("Locked LME-20 baseline mismatch")
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
    for projection_dims in PROJECTION_DIMS:
        evaluation = aggregate_evaluation(
            folds,
            [fold.candidate_predictions[int(projection_dims)] for fold in folds],
        )
        fixed_diagnostics.append({
            "projection_dims": int(projection_dims),
            "aggregate": evaluation,
        })
    fixed_diagnostics.sort(
        key=lambda row: row["aggregate"]["candidate"]["macro_f1"],
        reverse=True,
    )

    report = {
        "contract": {
            "baseline": "locked CAM++ LME-20",
            "lda_groups": "train-only 446 ground-truth known speakers",
            "candidate_projection_dims": list(PROJECTION_DIMS),
            "within_shrinkage": WITHIN_SHRINKAGE,
            "lme_beta": LME_BETA,
            "fixed_decision": {
                "alpha": LOCKED_ALPHA,
                "kappa": LOCKED_RAW_KAPPA,
                "tau": LOCKED_TAU,
                "lambda_unknown": LOCKED_UNKNOWN_WEIGHT,
            },
            "target_fold_selection": "leave-one-fold-out; target never selects itself",
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
        "all_oof_fixed_dimension_diagnostics_non_decisional": fixed_diagnostics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "selections": [
            {
                "target_fold": row["target_fold"],
                "projection_dims": row["parameters"]["projection_dims"],
                "calibration_feasible": row["calibration"][
                    "calibration_feasible"
                ],
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
