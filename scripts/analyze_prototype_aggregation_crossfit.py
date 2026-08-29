"""Cross-fit multi-enrollment prototype aggregation for CAM++ embeddings.

The shipped 1000-centroid rule represents every known or pseudo-unknown
speaker by one L2-normalised mean.  Sparse-enrollment and mismatch literature
suggests that robust centroids, medoids, or set-to-vector scores can be better.
This audit changes only that aggregation while keeping the validated KMeans-554
partition, checkpoints, folds, and decision grid fixed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import logsumexp, softmax

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_control_oof_centroid_crossfit import (  # noqa: E402
    HISTORICAL_PARAMS,
    KAPPAS,
    NUM_CLASSES,
    NUM_FOLDS,
    NUM_KNOWN,
    FoldEvidence,
    evaluate_policy,
    l2norm_rows,
    metric_bundle,
    metric_delta,
    parameter_grid,
    predict,
)
from scripts.analyze_unknown_cluster_hypotheses_crossfit import (  # noqa: E402
    load_fold_inputs,
)
from submission.inference import _collapse_centroid_probs  # noqa: E402


WEIGHTED_BETAS = (5.0, 10.0, 20.0)
LOGMEANEXP_BETAS = (10.0, 20.0, 40.0)


def group_indices(artifact: dict[str, np.ndarray]) -> list[np.ndarray]:
    labels = artifact["competition_labels"].astype(np.int64)
    cluster_ids = artifact["unknown_cluster_ids"].astype(np.int64)
    groups = []
    for speaker_id in range(1, NUM_KNOWN + 1):
        indices = np.flatnonzero(labels == speaker_id)
        if len(indices) == 0:
            raise RuntimeError(f"Missing known enrollment group {speaker_id}")
        groups.append(indices)
    for cluster_id in range(554):
        indices = np.flatnonzero(cluster_ids == cluster_id)
        if len(indices) == 0:
            raise RuntimeError(f"Missing unknown enrollment group {cluster_id}")
        groups.append(indices)
    return groups


def centroid_matrix(
    embeddings: np.ndarray,
    groups: list[np.ndarray],
    mode: str,
    beta: float | None = None,
) -> np.ndarray:
    prototypes = np.zeros((len(groups), embeddings.shape[1]), dtype=np.float32)
    for group_id, indices in enumerate(groups):
        members = embeddings[indices]
        mean = members.mean(axis=0)
        if mode == "mean":
            prototype = mean
        elif mode == "medoid":
            similarities = members @ members.T
            prototype = members[int(np.argmax(similarities.mean(axis=1)))]
        elif mode == "centrality_weighted":
            assert beta is not None
            unit_mean = mean / (np.linalg.norm(mean) + 1e-12)
            weights = softmax(float(beta) * (members @ unit_mean))
            prototype = np.sum(weights[:, None] * members, axis=0)
        else:
            raise ValueError(mode)
        prototypes[group_id] = prototype
    return l2norm_rows(prototypes)


def score_matrix_to_evidence(
    *,
    fold: int,
    oof: dict,
    scores: np.ndarray,
) -> FoldEvidence:
    probabilities = {}
    max_cosines = {}
    max_score = scores.max(axis=1).astype(np.float64)
    for kappa in KAPPAS:
        internal = np.zeros((len(scores), 1 + scores.shape[1]), dtype=np.float64)
        internal[:, 1:] = softmax(float(kappa) * scores, axis=1)
        probabilities[float(kappa)] = _collapse_centroid_probs(
            internal, NUM_CLASSES
        )
        max_cosines[float(kappa)] = max_score
    return FoldEvidence(
        fold=fold,
        files=oof["files"].astype(str),
        labels=oof["labels"].astype(np.int64),
        head=oof["competition_probs"].astype(np.float64),
        embeddings=l2norm_rows(oof["embeddings"]),
        known_centroids=np.empty((0, 0), dtype=np.float32),
        unknown_centroids=np.empty((0, 0), dtype=np.float32),
        centroid_probabilities=probabilities,
        max_cosines=max_cosines,
    )


def fold_variants(
    *, fold: int, artifact: dict[str, np.ndarray], oof: dict
) -> tuple[dict[str, FoldEvidence], dict]:
    train = l2norm_rows(artifact["train_embeddings"])
    validation = l2norm_rows(oof["embeddings"])
    groups = group_indices(artifact)
    variants = {}
    diagnostics = {}

    mean_centroids = centroid_matrix(train, groups, "mean")
    shipped_centroids = np.vstack([
        artifact["known_centroids"], artifact["unknown_centroids"]
    ])
    centroid_diff = float(np.max(np.abs(mean_centroids - shipped_centroids)))
    if centroid_diff > 1e-5:
        raise RuntimeError(
            f"Fold {fold} reconstructed mean centroid mismatch: {centroid_diff}"
        )

    def register_scores(name: str, scores: np.ndarray, details: dict) -> None:
        variants[name] = score_matrix_to_evidence(
            fold=fold, oof=oof, scores=scores.astype(np.float64)
        )
        diagnostics[name] = {
            **details,
            "score_min": float(scores.min()),
            "score_mean": float(scores.mean()),
            "score_max": float(scores.max()),
        }

    register_scores(
        "mean_centroid",
        validation @ mean_centroids.T,
        {"family": "prototype", "centroid_max_abs_diff_vs_shipped": centroid_diff},
    )
    medoids = centroid_matrix(train, groups, "medoid")
    register_scores(
        "medoid", validation @ medoids.T, {"family": "prototype"}
    )
    for beta in WEIGHTED_BETAS:
        centroids = centroid_matrix(
            train, groups, "centrality_weighted", beta=beta
        )
        register_scores(
            f"centrality_weighted_b{int(beta)}",
            validation @ centroids.T,
            {"family": "prototype", "centrality_beta": beta},
        )

    # Compute every enrollment similarity once; set-to-vector variants only
    # differ in the deterministic within-speaker reduction.
    all_similarities = validation @ train.T
    raw_mean_scores = np.empty((len(validation), len(groups)), dtype=np.float32)
    max_scores = np.empty_like(raw_mean_scores)
    top2_scores = np.empty_like(raw_mean_scores)
    logmeanexp_scores = {
        beta: np.empty_like(raw_mean_scores) for beta in LOGMEANEXP_BETAS
    }
    for group_id, indices in enumerate(groups):
        values = all_similarities[:, indices]
        raw_mean_scores[:, group_id] = values.mean(axis=1)
        max_scores[:, group_id] = values.max(axis=1)
        if values.shape[1] == 1:
            top2_scores[:, group_id] = values[:, 0]
        else:
            top2_scores[:, group_id] = np.partition(
                values, values.shape[1] - 2, axis=1
            )[:, -2:].mean(axis=1)
        for beta in LOGMEANEXP_BETAS:
            logmeanexp_scores[beta][:, group_id] = (
                logsumexp(float(beta) * values, axis=1)
                - np.log(values.shape[1])
            ) / float(beta)

    register_scores(
        "raw_mean_similarity",
        raw_mean_scores,
        {"family": "set_score", "normalises_group_centroid": False},
    )
    register_scores(
        "max_exemplar", max_scores, {"family": "set_score"}
    )
    register_scores(
        "top2_mean_exemplar", top2_scores, {"family": "set_score"}
    )
    for beta, scores in logmeanexp_scores.items():
        register_scores(
            f"logmeanexp_b{int(beta)}",
            scores,
            {"family": "set_score", "logmeanexp_beta": beta},
        )
    return variants, diagnostics


def select_candidate(
    variants: dict[str, list[FoldEvidence]], calibration_folds: tuple[int, int]
) -> tuple[str, dict[str, float], dict]:
    reference = variants["mean_centroid"]
    baselines = {
        fold: metric_bundle(
            reference[fold].labels, reference[fold].baseline_predictions
        )
        for fold in calibration_folds
    }
    ranked = []
    preference = {"mean_centroid": 0, "centrality_weighted_b10": 1}
    for name, folds in variants.items():
        for params in parameter_grid():
            metrics = {}
            gains = []
            for fold in calibration_folds:
                candidate = metric_bundle(folds[fold].labels, predict(folds[fold], params))
                metrics[fold] = candidate
                gains.append(candidate["macro_f1"] - baselines[fold]["macro_f1"])
            baseline_preference = -float(preference.get(name, 2))
            distance = (
                abs(params["alpha"] - HISTORICAL_PARAMS["alpha"])
                + abs(params["kappa"] - HISTORICAL_PARAMS["kappa"]) / 32.0
                + abs(params["tau"] - HISTORICAL_PARAMS["tau"])
                + abs(
                    params["lambda_unknown"]
                    - HISTORICAL_PARAMS["lambda_unknown"]
                )
            )
            rank = (
                min(gains), float(np.mean(gains)), baseline_preference, -distance
            )
            ranked.append((rank, name, params, gains, metrics))
    rank, name, params, gains, metrics = max(ranked, key=lambda row: row[0])
    return name, params, {
        "calibration_folds": list(calibration_folds),
        "selection_objective": (
            "maximise minimum calibration-fold Macro-F1 gain across aggregation "
            "and decision parameters"
        ),
        "minimum_gain": float(min(gains)),
        "mean_gain": float(np.mean(gains)),
        "per_fold_metrics": {str(key): value for key, value in metrics.items()},
        "rank_tuple": [float(value) for value in rank],
    }


def diagnostic_ranking(variants: dict[str, list[FoldEvidence]]) -> list[dict]:
    rows = []
    for name, folds in variants.items():
        best = None
        for params in parameter_grid():
            evaluation = evaluate_policy(
                folds, [predict(fold, params) for fold in folds]
            )
            deltas = [row["delta"]["macro_f1"] for row in evaluation["folds"]]
            rank = (min(deltas), float(np.mean(deltas)))
            if best is None or rank > best[0]:
                best = (rank, params, evaluation)
        assert best is not None
        rows.append({
            "variant": name,
            "parameters": best[1],
            "minimum_fold_gain": float(best[0][0]),
            "mean_fold_gain": float(best[0][1]),
            "evaluation": best[2],
        })
    return sorted(
        rows,
        key=lambda row: (row["minimum_fold_gain"], row["mean_fold_gain"]),
        reverse=True,
    )


def select_parameters_for_fixed_aggregation(
    folds: list[FoldEvidence], calibration_folds: tuple[int, int]
) -> tuple[dict[str, float], dict]:
    baselines = {
        fold: metric_bundle(folds[fold].labels, folds[fold].baseline_predictions)
        for fold in calibration_folds
    }
    ranked = []
    for params in parameter_grid():
        gains = []
        metrics = {}
        for fold in calibration_folds:
            candidate = metric_bundle(folds[fold].labels, predict(folds[fold], params))
            metrics[fold] = candidate
            gains.append(candidate["macro_f1"] - baselines[fold]["macro_f1"])
        distance = (
            abs(params["alpha"] - HISTORICAL_PARAMS["alpha"])
            + abs(params["kappa"] - HISTORICAL_PARAMS["kappa"]) / 32.0
            + abs(params["tau"] - HISTORICAL_PARAMS["tau"])
            + abs(
                params["lambda_unknown"]
                - HISTORICAL_PARAMS["lambda_unknown"]
            )
        )
        rank = (min(gains), float(np.mean(gains)), -distance)
        ranked.append((rank, params, gains, metrics))
    rank, params, gains, metrics = max(ranked, key=lambda row: row[0])
    return params, {
        "calibration_folds": list(calibration_folds),
        "minimum_gain": float(min(gains)),
        "mean_gain": float(np.mean(gains)),
        "per_fold_metrics": {str(key): value for key, value in metrics.items()},
        "rank_tuple": [float(value) for value in rank],
    }


def per_aggregation_crossfit(
    variants: dict[str, list[FoldEvidence]]
) -> list[dict]:
    rows = []
    for name, folds in variants.items():
        predictions = []
        selections = []
        for target in range(NUM_FOLDS):
            calibration = tuple(
                fold for fold in range(NUM_FOLDS) if fold != target
            )
            params, selection = select_parameters_for_fixed_aggregation(
                folds, calibration  # type: ignore[arg-type]
            )
            prediction = predict(folds[target], params)
            baseline = metric_bundle(
                folds[target].labels, folds[target].baseline_predictions
            )
            candidate = metric_bundle(folds[target].labels, prediction)
            predictions.append(prediction)
            selections.append({
                "target_fold": target,
                "parameters": params,
                "calibration": selection,
                "held_out": {
                    "baseline": baseline,
                    "candidate": candidate,
                    "delta": metric_delta(candidate, baseline),
                },
            })
        evaluation = evaluate_policy(folds, predictions)
        deltas = [row["delta"]["macro_f1"] for row in evaluation["folds"]]
        rows.append({
            "variant": name,
            "selections": selections,
            "evaluation": evaluation,
            "minimum_fold_gain": float(min(deltas)),
            "mean_fold_gain": float(np.mean(deltas)),
        })
    return sorted(
        rows,
        key=lambda row: (
            row["minimum_fold_gain"],
            row["mean_fold_gain"],
            row["evaluation"]["aggregate"]["candidate"]["macro_f1"],
        ),
        reverse=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-root", type=Path, default=ROOT / "checkpoints"
    )
    parser.add_argument(
        "--cache-dir", type=Path,
        default=ROOT / "data" / "experiments" / "campp_control_centroid_crossfit",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "reports" / "generated"
        / "campp_prototype_aggregation_crossfit.json",
    )
    args = parser.parse_args()

    oofs, artifacts, metadata = load_fold_inputs(args.checkpoint_root, args.cache_dir)
    per_fold = []
    diagnostics = []
    for fold in range(NUM_FOLDS):
        variants, fold_diagnostics = fold_variants(
            fold=fold, artifact=artifacts[fold], oof=oofs[fold]
        )
        per_fold.append(variants)
        diagnostics.append(fold_diagnostics)
    names = sorted(per_fold[0])
    if any(sorted(fold) != names for fold in per_fold):
        raise RuntimeError("Aggregation variants differ between folds")
    variants = {
        name: [per_fold[fold][name] for fold in range(NUM_FOLDS)]
        for name in names
    }

    selections = []
    predictions = []
    for target in range(NUM_FOLDS):
        calibration = tuple(fold for fold in range(NUM_FOLDS) if fold != target)
        name, params, selection = select_candidate(
            variants, calibration  # type: ignore[arg-type]
        )
        held = variants[name][target]
        prediction = predict(held, params)
        baseline = metric_bundle(held.labels, held.baseline_predictions)
        candidate = metric_bundle(held.labels, prediction)
        selections.append({
            "target_fold": target,
            "variant": name,
            "parameters": params,
            "calibration": selection,
            "held_out": {
                "baseline": baseline,
                "candidate": candidate,
                "delta": metric_delta(candidate, baseline),
            },
        })
        predictions.append(prediction)

    crossfit = evaluate_policy(variants["mean_centroid"], predictions)
    fixed_historical = {
        name: evaluate_policy(
            folds, [predict(fold, HISTORICAL_PARAMS) for fold in folds]
        )
        for name, folds in variants.items()
    }
    aggregation_crossfit = per_aggregation_crossfit(variants)
    ranking = diagnostic_ranking(variants)
    report = {
        "contract": {
            "weights": "fixed CAM++ Control Fold0/1/2",
            "unknown_partition": "fixed shipped train-only KMeans-554",
            "aggregations": names,
            "selection": (
                "leave-one-fold-out jointly selects aggregation and decision "
                "parameters; target fold is held out"
            ),
            "decision_candidates_per_aggregation": len(parameter_grid()),
        },
        "provenance": {
            "cache_metadata": metadata,
            "aggregation_diagnostics": diagnostics,
        },
        "crossfit": {"selections": selections, "evaluation": crossfit},
        "fixed_historical_parameters": {
            "parameters": HISTORICAL_PARAMS,
            "variants": fixed_historical,
        },
        "per_aggregation_crossfit": aggregation_crossfit,
        "all_oof_diagnostic_ranking": ranking,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "crossfit_selections": [
            {
                "target_fold": row["target_fold"],
                "variant": row["variant"],
                "parameters": row["parameters"],
                "held_out_delta": row["held_out"]["delta"],
            }
            for row in selections
        ],
        "crossfit_aggregate": crossfit["aggregate"],
        "top_per_aggregation_crossfit": [
            {
                "variant": row["variant"],
                "minimum_fold_gain": row["minimum_fold_gain"],
                "aggregate_macro_f1": row["evaluation"]["aggregate"]["candidate"]["macro_f1"],
                "aggregate_delta": row["evaluation"]["aggregate"]["delta"],
            }
            for row in aggregation_crossfit[:6]
        ],
        "top_all_oof_diagnostics": [
            {
                "variant": row["variant"],
                "parameters": row["parameters"],
                "minimum_fold_gain": row["minimum_fold_gain"],
                "aggregate_macro_f1": row["evaluation"]["aggregate"]["candidate"]["macro_f1"],
            }
            for row in ranking[:6]
        ],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
