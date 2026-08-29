"""Cross-fit unknown-clustering hypotheses using cached CAM++ embeddings.

This is the second stage of the 1000-centroid audit.  It holds model weights,
folds, known centroids and OOF embeddings fixed, and changes only the latent
unknown-speaker partition.  A target fold cannot select its clustering method,
cluster count or decision parameters.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_control_oof_centroid_crossfit import (  # noqa: E402
    HISTORICAL_PARAMS,
    NUM_FOLDS,
    FoldEvidence,
    aggregate_predictions,
    evaluate_policy,
    l2norm_rows,
    load_oof,
    metric_bundle,
    metric_delta,
    parameter_grid,
    predict,
    prepare_fold_evidence,
    sha256_file,
)
from src.unknown_clustering import (  # noqa: E402
    build_centroids,
    cluster_agglomerative,
    cluster_kmeans,
)


KMEANS_COUNTS = (455, 500, 554, 600, 700)
AHC_COUNTS = (455, 554, 700)


def load_fold_inputs(
    checkpoint_root: Path, cache_dir: Path
) -> tuple[list[dict], list[dict], list[dict]]:
    oofs = []
    artifacts = []
    metadata = []
    seen_files: set[str] = set()
    for fold in range(NUM_FOLDS):
        profile = f"p0-campp-known446-ood-control-oof-f{fold}"
        oof_path = (
            checkpoint_root / profile / "campp_best_bundle" / "oof_predictions.npz"
        )
        # The split match was already asserted when this cache was created.
        with np.load(oof_path) as data:
            expected = set(data["files"].astype(str).tolist())
        if seen_files & expected:
            raise RuntimeError(f"Fold {fold} overlaps an earlier OOF fold")
        seen_files |= expected
        oofs.append(load_oof(oof_path, fold, expected))

        artifact_path = cache_dir / f"fold{fold}_train_embeddings_centroids.npz"
        metadata_path = artifact_path.with_suffix(".json")
        item = json.loads(metadata_path.read_text(encoding="utf-8"))
        if sha256_file(artifact_path) != item["artifact_sha256"]:
            raise RuntimeError(f"Fold {fold} cached artifact hash mismatch")
        # The artifact hash is checked above before permitting the legacy
        # object-dtype filename array to be unpickled.
        with np.load(artifact_path, allow_pickle=True) as data:
            artifacts.append({key: data[key].copy() for key in data.files})
        metadata.append(item)
    if len(seen_files) != 4447:
        raise RuntimeError(f"Expected 4447 unique OOF files, got {len(seen_files)}")
    return oofs, artifacts, metadata


def cluster_variants_for_fold(
    *, fold: int, artifact: dict[str, np.ndarray], oof: dict
) -> tuple[dict[str, FoldEvidence], dict]:
    train_embeddings = l2norm_rows(artifact["train_embeddings"])
    train_labels = artifact["competition_labels"].astype(np.int64)
    unknown_mask = train_labels == 0
    unknown_embeddings = train_embeddings[unknown_mask]
    shipped_ids = artifact["unknown_cluster_ids"][unknown_mask].astype(np.int64)
    known_centroids = artifact["known_centroids"]

    variants: dict[str, FoldEvidence] = {}
    diagnostics: dict[str, dict] = {}

    def register(name: str, cluster_ids: np.ndarray, method: str, requested_k: int) -> None:
        centroids, sizes = build_centroids(unknown_embeddings, cluster_ids)
        if np.any(sizes == 0):
            raise RuntimeError(f"{name} Fold {fold} produced empty clusters")
        variants[name] = prepare_fold_evidence(
            fold=fold,
            oof=oof,
            artifact={
                "known_centroids": known_centroids,
                "unknown_centroids": centroids,
            },
        )
        diagnostics[name] = {
            "method": method,
            "requested_k": requested_k,
            "actual_k": int(len(sizes)),
            "size_min": int(sizes.min()),
            "size_median": float(np.median(sizes)),
            "size_max": int(sizes.max()),
            "singletons": int(np.sum(sizes == 1)),
            "ari_vs_shipped": float(adjusted_rand_score(shipped_ids, cluster_ids)),
        }

    register("shipped_kmeans_k554", shipped_ids, "shipped", 554)
    for count in KMEANS_COUNTS:
        register(
            f"kmeans_k{count}",
            cluster_kmeans(unknown_embeddings, count, seed=42),
            "kmeans",
            count,
        )
    for count in AHC_COUNTS:
        ids = cluster_agglomerative(unknown_embeddings, count)
        # scipy's maxclust may return fewer than requested clusters. Relabel to
        # dense ids so centroid construction and provenance remain explicit.
        _, ids = np.unique(ids, return_inverse=True)
        register(f"ahc_k{count}", ids, "average_linkage_cosine", count)
    return variants, diagnostics


def all_fold_policy(
    variant_folds: list[FoldEvidence], params: dict[str, float]
) -> dict:
    return evaluate_policy(
        variant_folds, [predict(fold, params) for fold in variant_folds]
    )


def select_candidate(
    variants: dict[str, list[FoldEvidence]], calibration_folds: tuple[int, int]
) -> tuple[str, dict[str, float], dict]:
    baseline = {
        fold: metric_bundle(
            next(iter(variants.values()))[fold].labels,
            next(iter(variants.values()))[fold].baseline_predictions,
        )
        for fold in calibration_folds
    }
    ranked = []
    for name, folds in variants.items():
        for params in parameter_grid():
            gains = []
            fold_metrics = {}
            for fold in calibration_folds:
                candidate = metric_bundle(folds[fold].labels, predict(folds[fold], params))
                fold_metrics[fold] = candidate
                gains.append(candidate["macro_f1"] - baseline[fold]["macro_f1"])
            method_penalty = 0.0 if name == "shipped_kmeans_k554" else 1.0
            if "k554" in name:
                count_penalty = 0.0
            else:
                count = int(name.rsplit("k", 1)[-1])
                count_penalty = abs(count - 554) / 554.0
            parameter_distance = (
                abs(params["alpha"] - HISTORICAL_PARAMS["alpha"])
                + abs(params["kappa"] - HISTORICAL_PARAMS["kappa"]) / 32.0
                + abs(params["tau"] - HISTORICAL_PARAMS["tau"])
                + abs(
                    params["lambda_unknown"]
                    - HISTORICAL_PARAMS["lambda_unknown"]
                )
            )
            rank = (
                min(gains),
                float(np.mean(gains)),
                -method_penalty,
                -count_penalty,
                -parameter_distance,
            )
            ranked.append((rank, name, params, gains, fold_metrics))
    rank, name, params, gains, fold_metrics = max(ranked, key=lambda item: item[0])
    return name, params, {
        "calibration_folds": list(calibration_folds),
        "selection_objective": (
            "maximise minimum calibration-fold Macro-F1 gain across clustering "
            "hypothesis and decision parameters"
        ),
        "minimum_gain": float(min(gains)),
        "mean_gain": float(np.mean(gains)),
        "per_fold_metrics": {str(key): value for key, value in fold_metrics.items()},
        "rank_tuple": [float(value) for value in rank],
    }


def robust_full_oof_ranking(variants: dict[str, list[FoldEvidence]]) -> list[dict]:
    """Diagnostic all-OOF ranking; cross-fit remains the unbiased estimate."""
    rows = []
    for name, folds in variants.items():
        best = None
        for params in parameter_grid():
            evaluation = all_fold_policy(folds, params)
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
        / "campp_unknown_cluster_hypotheses_crossfit.json",
    )
    args = parser.parse_args()

    oofs, artifacts, metadata = load_fold_inputs(args.checkpoint_root, args.cache_dir)
    per_fold_variants = []
    diagnostics = []
    for fold in range(NUM_FOLDS):
        variants, fold_diagnostics = cluster_variants_for_fold(
            fold=fold, artifact=artifacts[fold], oof=oofs[fold]
        )
        per_fold_variants.append(variants)
        diagnostics.append(fold_diagnostics)

    variant_names = sorted(per_fold_variants[0])
    if any(sorted(fold) != variant_names for fold in per_fold_variants):
        raise RuntimeError("Variant names differ between folds")
    variants = {
        name: [per_fold_variants[fold][name] for fold in range(NUM_FOLDS)]
        for name in variant_names
    }

    selections = []
    selected_predictions = []
    for target in range(NUM_FOLDS):
        calibration = tuple(fold for fold in range(NUM_FOLDS) if fold != target)
        name, params, selection = select_candidate(
            variants, calibration  # type: ignore[arg-type]
        )
        target_fold = variants[name][target]
        prediction = predict(target_fold, params)
        baseline = metric_bundle(
            target_fold.labels, target_fold.baseline_predictions
        )
        candidate = metric_bundle(target_fold.labels, prediction)
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
        selected_predictions.append(prediction)

    reference_folds = next(iter(variants.values()))
    crossfit = evaluate_policy(reference_folds, selected_predictions)
    fixed_historical = {
        name: all_fold_policy(folds, HISTORICAL_PARAMS)
        for name, folds in variants.items()
    }
    ranking = robust_full_oof_ranking(variants)
    report = {
        "contract": {
            "weights": "fixed CAM++ Control checkpoints",
            "known_centroids": "fixed train-only centroids",
            "unknown_hypotheses": {
                "kmeans_counts": KMEANS_COUNTS,
                "average_linkage_counts": AHC_COUNTS,
                "shipped_reference": "kmeans k=554 maps already embedded in checkpoints",
            },
            "decision_candidates_per_hypothesis": len(parameter_grid()),
            "selection": (
                "leave-one-fold-out jointly selects hypothesis and parameters; "
                "target fold is held out"
            ),
        },
        "provenance": {
            "cache_metadata": metadata,
            "cluster_diagnostics": diagnostics,
        },
        "crossfit": {
            "selections": selections,
            "evaluation": crossfit,
        },
        "fixed_historical_parameters": {
            "parameters": HISTORICAL_PARAMS,
            "variants": fixed_historical,
        },
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
        "top_all_oof_diagnostics": [
            {
                "variant": row["variant"],
                "parameters": row["parameters"],
                "minimum_fold_gain": row["minimum_fold_gain"],
                "aggregate_macro_f1": row["evaluation"]["aggregate"]["candidate"]["macro_f1"],
            }
            for row in ranking[:5]
        ],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
