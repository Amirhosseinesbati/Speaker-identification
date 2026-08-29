"""Leak-free shrinkage-WCCN backend audit for the locked CAM++ LME-20 rule.

Each fold estimates within-group covariance exclusively from its matching
training embeddings and the fixed known/pseudo-unknown enrollment groups.  A
shrinkage interpolation between identity and full WCCN is applied before the
unchanged LME-20 and decision rule.  The WCCN strength for a held-out fold is
selected only on the other two folds.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

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
from scripts.analyze_lme20_entropy_fusion_crossfit import (  # noqa: E402
    fused_predictions,
    prototype_probabilities,
)
from scripts.analyze_prototype_aggregation_crossfit import (  # noqa: E402
    group_indices,
)
from scripts.analyze_unknown_cluster_hypotheses_crossfit import (  # noqa: E402
    load_fold_inputs,
)


STRENGTHS = (0.10, 0.25, 0.50, 1.00)
EIGENVALUE_FLOOR = 1e-4


@dataclass
class FoldWccn:
    fold: int
    files: np.ndarray
    labels: np.ndarray
    baseline_predictions: np.ndarray
    candidate_predictions: dict[float, np.ndarray]
    diagnostics: dict


def within_group_covariance(
    embeddings: np.ndarray, groups: list[np.ndarray]
) -> tuple[np.ndarray, int]:
    embeddings = l2norm_rows(embeddings).astype(np.float64)
    covariance = np.zeros(
        (embeddings.shape[1], embeddings.shape[1]), dtype=np.float64
    )
    degrees_of_freedom = 0
    for indices in groups:
        if len(indices) < 2:
            continue
        members = embeddings[indices]
        residuals = members - members.mean(axis=0, keepdims=True)
        covariance += residuals.T @ residuals
        degrees_of_freedom += len(indices) - 1
    if degrees_of_freedom < embeddings.shape[1]:
        raise RuntimeError(
            "Insufficient within-group degrees of freedom for WCCN"
        )
    covariance /= float(degrees_of_freedom)
    covariance = 0.5 * (covariance + covariance.T)
    return covariance, degrees_of_freedom


def shrinkage_wccn_transform(
    covariance: np.ndarray, strength: float
) -> tuple[np.ndarray, dict]:
    covariance = np.asarray(covariance, dtype=np.float64)
    if not 0.0 <= float(strength) <= 1.0:
        raise ValueError("WCCN strength must be in [0, 1]")
    if float(strength) == 0.0:
        return np.eye(covariance.shape[0], dtype=np.float64), {
            "strength": 0.0,
            "condition_number": 1.0,
        }
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    mean_eigenvalue = float(np.mean(eigenvalues))
    if mean_eigenvalue <= 0.0:
        raise RuntimeError("Within-group covariance has non-positive trace")
    normalised = np.maximum(
        eigenvalues / mean_eigenvalue, EIGENVALUE_FLOOR
    )
    shrunk = (1.0 - float(strength)) + float(strength) * normalised
    inverse_sqrt = 1.0 / np.sqrt(shrunk)
    transform = (eigenvectors * inverse_sqrt[None, :]) @ eigenvectors.T
    return transform, {
        "strength": float(strength),
        "raw_eigenvalue_min": float(eigenvalues.min()),
        "raw_eigenvalue_median": float(np.median(eigenvalues)),
        "raw_eigenvalue_max": float(eigenvalues.max()),
        "normalised_eigenvalue_min": float(normalised.min()),
        "normalised_eigenvalue_max": float(normalised.max()),
        "shrunk_eigenvalue_min": float(shrunk.min()),
        "shrunk_eigenvalue_max": float(shrunk.max()),
        "condition_number": float(shrunk.max() / shrunk.min()),
    }


def fixed_lme_predictions(
    *, head: np.ndarray, raw_scores: np.ndarray
) -> np.ndarray:
    prototype = prototype_probabilities(raw_scores)
    return fused_predictions(
        head=head,
        prototype=prototype,
        raw_max=raw_scores.max(axis=1).astype(np.float64),
        head_weights=np.full(len(head), LOCKED_ALPHA, dtype=np.float64),
    )


def prepare_fold(
    *,
    fold: int,
    artifact: dict[str, np.ndarray],
    oof: dict,
    covariance_group_scope: str,
) -> FoldWccn:
    train = l2norm_rows(artifact["train_embeddings"])
    validation = l2norm_rows(oof["embeddings"])
    groups = group_indices(artifact)
    if covariance_group_scope == "all":
        covariance_groups = groups
    elif covariance_group_scope == "known":
        covariance_groups = groups[:NUM_KNOWN]
    else:
        raise ValueError(f"Unknown covariance group scope: {covariance_group_scope}")
    head = oof["competition_probs"].astype(np.float64)
    raw_scores = logmeanexp_group_scores(validation, train, groups)
    baseline = fixed_lme_predictions(head=head, raw_scores=raw_scores)
    covariance, degrees_of_freedom = within_group_covariance(
        train, covariance_groups
    )

    candidates = {}
    diagnostics = {
        "train_files": int(len(train)),
        "validation_files": int(len(validation)),
        "groups": int(len(groups)),
        "covariance_group_scope": covariance_group_scope,
        "covariance_groups": int(len(covariance_groups)),
        "within_group_degrees_of_freedom": int(degrees_of_freedom),
        "strength": {},
    }
    for strength in STRENGTHS:
        transform, transform_diagnostics = shrinkage_wccn_transform(
            covariance, strength
        )
        transformed_train = l2norm_rows(train @ transform)
        transformed_validation = l2norm_rows(validation @ transform)
        transformed_scores = logmeanexp_group_scores(
            transformed_validation, transformed_train, groups
        )
        candidates[float(strength)] = fixed_lme_predictions(
            head=head, raw_scores=transformed_scores
        )
        diagnostics["strength"][str(strength)] = {
            **transform_diagnostics,
            "score_min": float(transformed_scores.min()),
            "score_max": float(transformed_scores.max()),
        }
    return FoldWccn(
        fold=fold,
        files=oof["files"].astype(str),
        labels=oof["labels"].astype(np.int64),
        baseline_predictions=baseline,
        candidate_predictions=candidates,
        diagnostics=diagnostics,
    )


def fold_metrics(fold: FoldWccn, strength: float) -> tuple[dict, dict, dict]:
    baseline = metric_bundle(fold.labels, fold.baseline_predictions)
    candidate = metric_bundle(fold.labels, fold.candidate_predictions[strength])
    return baseline, candidate, metric_delta(candidate, baseline)


def select_strength(
    folds: list[FoldWccn], calibration_folds: tuple[int, int]
) -> tuple[float, dict]:
    ranked = []
    for strength in STRENGTHS:
        rows = []
        for fold_id in calibration_folds:
            baseline, candidate, delta = fold_metrics(folds[fold_id], strength)
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
        rank = (int(feasible), min(gains), float(np.mean(gains)), -strength)
        ranked.append((rank, float(strength), rows, feasible))
    rank, strength, rows, feasible = max(ranked, key=lambda item: item[0])
    return strength, {
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
    folds: list[FoldWccn], predictions: list[np.ndarray]
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
        default=ROOT / "reports" / "generated" / "campp_lme20_wccn_crossfit.json",
    )
    parser.add_argument(
        "--covariance-groups",
        choices=("all", "known"),
        default="all",
        help=(
            "Use all fixed enrollment groups or only the 446 ground-truth "
            "known speakers when estimating within-group covariance."
        ),
    )
    args = parser.parse_args()

    oofs, artifacts, metadata = load_fold_inputs(args.checkpoint_root, args.cache_dir)
    folds = [
        prepare_fold(
            fold=fold,
            artifact=artifacts[fold],
            oof=oofs[fold],
            covariance_group_scope=args.covariance_groups,
        )
        for fold in range(NUM_FOLDS)
    ]
    selections = []
    predictions = []
    for target in range(NUM_FOLDS):
        calibration = tuple(fold for fold in range(NUM_FOLDS) if fold != target)
        strength, selection = select_strength(
            folds, calibration  # type: ignore[arg-type]
        )
        baseline, candidate, delta = fold_metrics(folds[target], strength)
        selections.append({
            "target_fold": target,
            "parameters": {"strength": strength},
            "calibration": selection,
            "held_out": {
                "baseline": baseline,
                "candidate": candidate,
                "delta": delta,
            },
        })
        predictions.append(folds[target].candidate_predictions[strength])

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
    for strength in STRENGTHS:
        evaluation = aggregate_evaluation(
            folds, [fold.candidate_predictions[float(strength)] for fold in folds]
        )
        fixed_diagnostics.append({
            "strength": float(strength),
            "aggregate": evaluation,
        })
    fixed_diagnostics.sort(
        key=lambda row: row["aggregate"]["candidate"]["macro_f1"],
        reverse=True,
    )

    report = {
        "contract": {
            "baseline": "locked CAM++ LME-20",
            "wccn_groups": (
                "train-only 446 ground-truth known"
                if args.covariance_groups == "known"
                else "train-only 446 known + 554 fixed pseudo-unknown"
            ),
            "covariance_group_scope": args.covariance_groups,
            "candidate_strengths": list(STRENGTHS),
            "eigenvalue_floor": EIGENVALUE_FLOOR,
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
        "all_oof_fixed_strength_diagnostics_non_decisional": fixed_diagnostics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "selections": [
            {
                "target_fold": row["target_fold"],
                "strength": row["parameters"]["strength"],
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
