"""Cross-fit entropy-reliability fusion on top of the locked LME-20 policy.

The fixed LME-20 policy rescues many CAM++ head errors but occasionally
overrides a correct head decision.  This audit keeps its prototypes, LME beta,
softmax scale, unknown weight and raw-score OOD threshold fixed.  It changes
only the per-file head/prototype mixture weight using the difference between
their normalised entropy reliabilities.

For each held-out OOF fold, the single adaptation strength is selected on the
other two folds.  The held-out fold never chooses its own parameter.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import expit, logit, softmax

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
    LME_BETA,
    LOCKED_ALPHA,
    LOCKED_RAW_KAPPA,
    LOCKED_TAU,
    LOCKED_UNKNOWN_WEIGHT,
    logmeanexp_group_scores,
)
from scripts.analyze_prototype_aggregation_crossfit import (  # noqa: E402
    group_indices,
)
from scripts.analyze_unknown_cluster_hypotheses_crossfit import (  # noqa: E402
    load_fold_inputs,
)
from submission.inference import _collapse_centroid_probs  # noqa: E402


GAMMAS = (0.5, 1.0, 2.0, 4.0)


@dataclass
class FoldFusion:
    fold: int
    files: np.ndarray
    labels: np.ndarray
    head: np.ndarray
    prototype: np.ndarray
    raw_max: np.ndarray
    head_predictions: np.ndarray
    baseline_predictions: np.ndarray
    candidate_predictions: dict[float, np.ndarray]
    diagnostics: dict


def prototype_probabilities(scores: np.ndarray) -> np.ndarray:
    internal = np.zeros((len(scores), 1 + scores.shape[1]), dtype=np.float64)
    internal[:, 1:] = softmax(LOCKED_RAW_KAPPA * scores, axis=1)
    return _collapse_centroid_probs(internal, NUM_CLASSES)


def entropy_reliability(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    clipped = np.clip(probabilities, 1e-15, 1.0)
    entropy = -np.sum(clipped * np.log(clipped), axis=1)
    reliability = 1.0 - entropy / np.log(probabilities.shape[1])
    return np.clip(reliability, 0.0, 1.0)


def adaptive_head_weights(
    head: np.ndarray, prototype: np.ndarray, gamma: float
) -> tuple[np.ndarray, dict]:
    head_reliability = entropy_reliability(head)
    prototype_reliability = entropy_reliability(prototype)
    reliability_delta = head_reliability - prototype_reliability
    weights = expit(logit(LOCKED_ALPHA) + float(gamma) * reliability_delta)
    return weights, {
        "gamma": float(gamma),
        "head_reliability_min": float(head_reliability.min()),
        "head_reliability_median": float(np.median(head_reliability)),
        "head_reliability_max": float(head_reliability.max()),
        "prototype_reliability_min": float(prototype_reliability.min()),
        "prototype_reliability_median": float(np.median(prototype_reliability)),
        "prototype_reliability_max": float(prototype_reliability.max()),
        "reliability_delta_min": float(reliability_delta.min()),
        "reliability_delta_median": float(np.median(reliability_delta)),
        "reliability_delta_max": float(reliability_delta.max()),
        "head_weight_min": float(weights.min()),
        "head_weight_median": float(np.median(weights)),
        "head_weight_max": float(weights.max()),
    }


def fused_predictions(
    *,
    head: np.ndarray,
    prototype: np.ndarray,
    raw_max: np.ndarray,
    head_weights: np.ndarray,
) -> np.ndarray:
    fused = (
        head_weights[:, None] * head
        + (1.0 - head_weights[:, None]) * prototype
    )
    fused[:, 0] *= LOCKED_UNKNOWN_WEIGHT
    fused /= fused.sum(axis=1, keepdims=True) + 1e-12
    predictions = fused.argmax(axis=1).astype(np.int64)
    predictions[raw_max < LOCKED_TAU] = 0
    return predictions


def prepare_fold(
    *, fold: int, artifact: dict[str, np.ndarray], oof: dict
) -> FoldFusion:
    groups = group_indices(artifact)
    raw_scores = logmeanexp_group_scores(
        oof["embeddings"], artifact["train_embeddings"], groups
    )
    raw_max = raw_scores.max(axis=1).astype(np.float64)
    head = oof["competition_probs"].astype(np.float64)
    prototype = prototype_probabilities(raw_scores)
    fixed_weights = np.full(len(head), LOCKED_ALPHA, dtype=np.float64)
    baseline = fused_predictions(
        head=head,
        prototype=prototype,
        raw_max=raw_max,
        head_weights=fixed_weights,
    )
    candidates = {}
    diagnostics = {"gamma": {}}
    for gamma in GAMMAS:
        weights, weight_diagnostics = adaptive_head_weights(
            head, prototype, gamma
        )
        candidates[float(gamma)] = fused_predictions(
            head=head,
            prototype=prototype,
            raw_max=raw_max,
            head_weights=weights,
        )
        diagnostics["gamma"][str(gamma)] = weight_diagnostics
    head_predictions = head.argmax(axis=1).astype(np.int64)
    labels = oof["labels"].astype(np.int64)
    head_correct = head_predictions == labels
    baseline_correct = baseline == labels
    diagnostics["fixed_lme_vs_head"] = {
        "rescued_head_errors": int(np.sum(~head_correct & baseline_correct)),
        "introduced_vs_head": int(np.sum(head_correct & ~baseline_correct)),
        "both_wrong": int(np.sum(~head_correct & ~baseline_correct)),
        "both_correct": int(np.sum(head_correct & baseline_correct)),
        "prediction_disagreements": int(np.sum(head_predictions != baseline)),
    }
    return FoldFusion(
        fold=fold,
        files=oof["files"].astype(str),
        labels=labels,
        head=head,
        prototype=prototype,
        raw_max=raw_max,
        head_predictions=head_predictions,
        baseline_predictions=baseline,
        candidate_predictions=candidates,
        diagnostics=diagnostics,
    )


def fold_metrics(fold: FoldFusion, gamma: float) -> tuple[dict, dict, dict]:
    baseline = metric_bundle(fold.labels, fold.baseline_predictions)
    candidate = metric_bundle(fold.labels, fold.candidate_predictions[gamma])
    return baseline, candidate, metric_delta(candidate, baseline)


def select_gamma(
    folds: list[FoldFusion], calibration_folds: tuple[int, int]
) -> tuple[float, dict]:
    ranked = []
    for gamma in GAMMAS:
        rows = []
        for fold_id in calibration_folds:
            baseline, candidate, delta = fold_metrics(folds[fold_id], gamma)
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
        rank = (int(feasible), min(gains), float(np.mean(gains)), -gamma)
        ranked.append((rank, float(gamma), rows, feasible))
    rank, gamma, rows, feasible = max(ranked, key=lambda item: item[0])
    return gamma, {
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
    folds: list[FoldFusion], predictions: list[np.ndarray]
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
        default=ROOT / "reports" / "generated"
        / "campp_lme20_entropy_fusion_crossfit.json",
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
        gamma, selection = select_gamma(
            folds, calibration  # type: ignore[arg-type]
        )
        baseline, candidate, delta = fold_metrics(folds[target], gamma)
        selections.append({
            "target_fold": target,
            "parameters": {"gamma": gamma},
            "calibration": selection,
            "held_out": {
                "baseline": baseline,
                "candidate": candidate,
                "delta": delta,
            },
        })
        predictions.append(folds[target].candidate_predictions[gamma])

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
    for gamma in GAMMAS:
        evaluation = aggregate_evaluation(
            folds, [fold.candidate_predictions[float(gamma)] for fold in folds]
        )
        fixed_diagnostics.append({
            "gamma": float(gamma),
            "aggregate": evaluation,
        })
    fixed_diagnostics.sort(
        key=lambda row: row["aggregate"]["candidate"]["macro_f1"],
        reverse=True,
    )

    report = {
        "contract": {
            "baseline": "locked CAM++ LME-20 fixed alpha=0.15",
            "lme_beta": LME_BETA,
            "fixed_prototype_kappa": LOCKED_RAW_KAPPA,
            "fixed_tau": LOCKED_TAU,
            "fixed_unknown_weight": LOCKED_UNKNOWN_WEIGHT,
            "adaptive_weight": (
                "sigmoid(logit(0.15) + gamma * "
                "(head_entropy_reliability - prototype_entropy_reliability))"
            ),
            "candidate_gammas": list(GAMMAS),
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
        "all_oof_fixed_gamma_diagnostics_non_decisional": fixed_diagnostics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "selections": [
            {
                "target_fold": row["target_fold"],
                "gamma": row["parameters"]["gamma"],
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
