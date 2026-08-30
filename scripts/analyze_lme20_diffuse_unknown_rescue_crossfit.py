"""Cross-fit a conservative diffuse-unknown rescue on locked CAM++ LME-20.

The residual-topology audit shows that most remaining errors are known files
classified as ``unknown``.  This exploratory rule only changes an LME-20
unknown decision when the Raw head and the strongest known prototype agree on
the same known identity, the query is inside the locked prototype-distance
gate, and the aggregate unknown evidence is spread across many latent unknown
clusters.  Concentrated unknown evidence is left untouched.

The hypothesis family was generated from descriptive three-fold topology, so
this is exploratory rather than an independent confirmation.  Nevertheless,
every target fold is evaluated with thresholds selected only on the other two
folds and the leaderboard is never used.
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
    metric_bundle,
    metric_delta,
)
from scripts.analyze_lme20_asnorm_crossfit import (  # noqa: E402
    LOCKED_TAU,
)
from scripts.analyze_lme20_residual_topology import fold_rows  # noqa: E402
from scripts.analyze_unknown_cluster_hypotheses_crossfit import (  # noqa: E402
    load_fold_inputs,
)


EFFECTIVE_CLUSTER_THRESHOLDS = (4.0, 8.0, 16.0)
HEAD_MARGIN_THRESHOLDS = (0.1, 0.2, 0.3)
PARAMETER_GRID = tuple(
    (clusters, margin)
    for clusters in EFFECTIVE_CLUSTER_THRESHOLDS
    for margin in HEAD_MARGIN_THRESHOLDS
)


@dataclass
class FoldRescue:
    fold: int
    files: np.ndarray
    labels: np.ndarray
    baseline_predictions: np.ndarray
    candidate_predictions: dict[tuple[float, float], np.ndarray]
    candidate_masks: dict[tuple[float, float], np.ndarray]
    diagnostics: dict


def diffuse_unknown_rescue(
    rows: list[dict],
    *,
    minimum_effective_clusters: float,
    minimum_head_margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return predictions and the exact rows changed by the rescue rule."""

    predictions = np.array(
        [int(row["prediction"]) for row in rows], dtype=np.int64
    )
    changed = np.zeros(len(rows), dtype=bool)
    for index, row in enumerate(rows):
        head_prediction = int(row["head_prediction"])
        eligible = (
            predictions[index] == 0
            and head_prediction > 0
            and head_prediction == int(row["winner_known_id"])
            and float(row["raw_max_score"]) >= LOCKED_TAU
            and float(row["unknown_effective_clusters"])
            >= float(minimum_effective_clusters)
            and float(row["head_margin"]) >= float(minimum_head_margin)
        )
        if eligible:
            predictions[index] = head_prediction
            changed[index] = True
    return predictions, changed


def prepare_fold(
    *, fold: int, artifact: dict[str, np.ndarray], oof: dict
) -> FoldRescue:
    rows, topology_diagnostics = fold_rows(
        fold=fold, artifact=artifact, oof=oof
    )
    candidates = {}
    masks = {}
    for parameters in PARAMETER_GRID:
        prediction, mask = diffuse_unknown_rescue(
            rows,
            minimum_effective_clusters=parameters[0],
            minimum_head_margin=parameters[1],
        )
        candidates[parameters] = prediction
        masks[parameters] = mask
    return FoldRescue(
        fold=fold,
        files=np.array([str(row["file"]) for row in rows]),
        labels=np.array([int(row["label"]) for row in rows], dtype=np.int64),
        baseline_predictions=np.array(
            [int(row["prediction"]) for row in rows], dtype=np.int64
        ),
        candidate_predictions=candidates,
        candidate_masks=masks,
        diagnostics=topology_diagnostics,
    )


def fold_metrics(
    fold: FoldRescue, parameters: tuple[float, float]
) -> tuple[dict, dict, dict]:
    baseline = metric_bundle(fold.labels, fold.baseline_predictions)
    candidate = metric_bundle(
        fold.labels, fold.candidate_predictions[parameters]
    )
    return baseline, candidate, metric_delta(candidate, baseline)


def select_parameters(
    folds: list[FoldRescue], calibration_folds: tuple[int, int]
) -> tuple[tuple[float, float], dict]:
    ranked = []
    for parameters in PARAMETER_GRID:
        rows = []
        for fold_id in calibration_folds:
            baseline, candidate, delta = fold_metrics(
                folds[fold_id], parameters
            )
            rows.append({
                "fold": fold_id,
                "baseline": baseline,
                "candidate": candidate,
                "delta": delta,
                "changed_predictions": int(
                    folds[fold_id].candidate_masks[parameters].sum()
                ),
            })
        gains = [row["delta"]["macro_f1"] for row in rows]
        feasible = all(
            row["delta"]["macro_f1"] > 0.0
            and row["delta"]["known_accuracy"] >= -0.001
            and row["delta"]["ood_f1"] >= -0.001
            for row in rows
        )
        changed = sum(row["changed_predictions"] for row in rows)
        rank = (
            int(feasible),
            min(gains),
            float(np.mean(gains)),
            -changed,
            parameters[0],
            parameters[1],
        )
        ranked.append((rank, parameters, rows, feasible))
    rank, parameters, rows, feasible = max(ranked, key=lambda item: item[0])
    return parameters, {
        "calibration_folds": list(calibration_folds),
        "calibration_feasible": bool(feasible),
        "selection_objective": (
            "prefer positive Macro-F1 with -0.001 Known/OOD guardrails on "
            "both source folds, then maximise minimum and mean gain; ties "
            "prefer fewer changed predictions and stricter thresholds"
        ),
        "rank_tuple": [float(value) for value in rank],
        "per_fold": rows,
    }


def aggregate_evaluation(
    folds: list[FoldRescue], predictions: list[np.ndarray]
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
        "introduced_errors": int(
            np.sum(baseline_correct & ~candidate_correct)
        ),
        "changed_predictions": int(
            np.sum(baseline_predictions != candidate_predictions)
        ),
    }


def crossfit_gate(selections: list[dict], aggregate: dict) -> dict:
    """Apply the preregistered source-fold and held-out acceptance gates."""

    calibration_feasible = [
        bool(row["calibration"]["calibration_feasible"])
        for row in selections
    ]
    per_fold_pass = [
        row["held_out"]["delta"]["macro_f1"] > 0.0
        and row["held_out"]["delta"]["known_accuracy"] >= -0.001
        and row["held_out"]["delta"]["ood_f1"] >= -0.001
        for row in selections
    ]
    return {
        "scientific_role": "exploratory hypothesis screen only",
        "submission_authorized": False,
        "next_action_if_passed": (
            "freeze the selected rule and preregister an independent "
            "confirmation; do not promote this pooled-topology screen alone"
        ),
        "minimum_aggregate_macro_gain": 0.001,
        "maximum_known_accuracy_drop": 0.001,
        "maximum_ood_f1_drop": 0.001,
        "requires_all_calibration_pairs_feasible": True,
        "requires_all_held_out_folds_positive_with_guardrails": True,
        "calibration_feasible": calibration_feasible,
        "per_fold_pass": per_fold_pass,
        "passed": bool(
            all(calibration_feasible)
            and all(per_fold_pass)
            and aggregate["delta"]["macro_f1"] >= 0.001
            and aggregate["delta"]["known_accuracy"] >= -0.001
            and aggregate["delta"]["ood_f1"] >= -0.001
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
        / "campp_lme20_diffuse_unknown_rescue_crossfit.json",
    )
    args = parser.parse_args()

    oofs, artifacts, metadata = load_fold_inputs(
        args.checkpoint_root, args.cache_dir
    )
    folds = [
        prepare_fold(fold=fold, artifact=artifacts[fold], oof=oofs[fold])
        for fold in range(NUM_FOLDS)
    ]
    selections = []
    predictions = []
    for target in range(NUM_FOLDS):
        calibration = tuple(fold for fold in range(NUM_FOLDS) if fold != target)
        parameters, selection = select_parameters(
            folds, calibration  # type: ignore[arg-type]
        )
        baseline, candidate, delta = fold_metrics(folds[target], parameters)
        selections.append({
            "target_fold": target,
            "parameters": {
                "minimum_effective_clusters": parameters[0],
                "minimum_head_margin": parameters[1],
            },
            "calibration": selection,
            "held_out": {
                "baseline": baseline,
                "candidate": candidate,
                "delta": delta,
                "changed_predictions": int(
                    folds[target].candidate_masks[parameters].sum()
                ),
            },
        })
        predictions.append(folds[target].candidate_predictions[parameters])

    aggregate = aggregate_evaluation(folds, predictions)
    if abs(aggregate["baseline"]["macro_f1"] - 0.9633564052154656) > 1e-12:
        raise RuntimeError("Locked LME-20 baseline mismatch")
    gate = crossfit_gate(selections, aggregate)

    fixed_diagnostics = []
    for parameters in PARAMETER_GRID:
        evaluation = aggregate_evaluation(
            folds,
            [fold.candidate_predictions[parameters] for fold in folds],
        )
        fixed_diagnostics.append({
            "parameters": {
                "minimum_effective_clusters": parameters[0],
                "minimum_head_margin": parameters[1],
            },
            "aggregate": evaluation,
        })
    fixed_diagnostics.sort(
        key=lambda row: row["aggregate"]["candidate"]["macro_f1"],
        reverse=True,
    )

    report = {
        "contract": {
            "baseline": "locked CAM++ LME-20",
            "hypothesis": (
                "rescue LME unknown only when Raw head and strongest known "
                "prototype agree and latent-unknown evidence is diffuse"
            ),
            "raw_score_gate": LOCKED_TAU,
            "effective_cluster_thresholds": list(
                EFFECTIVE_CLUSTER_THRESHOLDS
            ),
            "head_margin_thresholds": list(HEAD_MARGIN_THRESHOLDS),
            "target_fold_selection": (
                "leave-one-fold-out; target never selects itself"
            ),
            "scientific_role": (
                "exploratory cross-fit after pooled descriptive topology; "
                "not independent confirmation and never a standalone "
                "submission authorization"
            ),
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
        "all_oof_fixed_parameter_diagnostics_non_decisional": (
            fixed_diagnostics
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "selections": [
            {
                "target_fold": row["target_fold"],
                "parameters": row["parameters"],
                "calibration_feasible": row["calibration"][
                    "calibration_feasible"
                ],
                "held_out_delta": row["held_out"]["delta"],
                "changed_predictions": row["held_out"][
                    "changed_predictions"
                ],
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
