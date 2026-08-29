"""Descriptive error-topology audit of the locked CAM++ LME-20 policy.

This script does not select or evaluate a new decision rule.  It explains the
remaining OOF errors by measuring how the 554 latent-unknown prototype groups
share probability mass, whether the head and prototype disagree, and where the
true known speaker ranks.  The output is hypothesis-generation evidence only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import softmax

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_control_oof_centroid_crossfit import (  # noqa: E402
    NUM_CLASSES,
    NUM_FOLDS,
    NUM_KNOWN,
    metric_bundle,
)
from scripts.analyze_lme20_asnorm_crossfit import (  # noqa: E402
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


TOP_COUNTS = (1, 4, 16, 64, 256)


def probability_margin(probabilities: np.ndarray) -> np.ndarray:
    top2 = np.partition(probabilities, probabilities.shape[1] - 2, axis=1)[
        :, -2:
    ]
    top2.sort(axis=1)
    return top2[:, 1] - top2[:, 0]


def unknown_mass_features(
    unknown_probabilities: np.ndarray,
) -> dict[str, np.ndarray]:
    unknown = np.asarray(unknown_probabilities, dtype=np.float64)
    mass = unknown.sum(axis=1)
    sorted_probabilities = np.sort(unknown, axis=1)[:, ::-1]
    features = {"unknown_mass": mass}
    cumulative = np.cumsum(sorted_probabilities, axis=1)
    for count in TOP_COUNTS:
        actual = min(count, unknown.shape[1])
        features[f"unknown_top{count}_mass"] = cumulative[:, actual - 1]
        features[f"unknown_top{count}_fraction"] = (
            cumulative[:, actual - 1] / np.maximum(mass, 1e-15)
        )
    conditional = unknown / np.maximum(mass[:, None], 1e-15)
    entropy = -np.sum(
        np.where(
            conditional > 0.0,
            conditional * np.log(np.maximum(conditional, 1e-300)),
            0.0,
        ),
        axis=1,
    )
    features["unknown_effective_clusters"] = np.exp(entropy)
    return features


def summarise(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {"mean": 0.0, "median": 0.0, "q10": 0.0, "q90": 0.0}
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "q10": float(np.quantile(values, 0.10)),
        "q90": float(np.quantile(values, 0.90)),
    }


def topology_name(label: int, prediction: int) -> str:
    if label == prediction:
        return "correct"
    if label > 0 and prediction == 0:
        return "known_to_unknown"
    if label > 0 and prediction > 0:
        return "known_to_wrong_known"
    if label == 0 and prediction > 0:
        return "unknown_to_known"
    raise RuntimeError(f"Unhandled label/prediction pair: {label}/{prediction}")


def fold_rows(
    *, fold: int, artifact: dict[str, np.ndarray], oof: dict
) -> tuple[list[dict], dict]:
    groups = group_indices(artifact)
    scores = logmeanexp_group_scores(
        oof["embeddings"], artifact["train_embeddings"], groups
    ).astype(np.float64)
    internal = np.zeros((len(scores), 1 + scores.shape[1]), dtype=np.float64)
    internal[:, 1:] = softmax(LOCKED_RAW_KAPPA * scores, axis=1)
    prototype = _collapse_centroid_probs(internal, NUM_CLASSES)
    head = oof["competition_probs"].astype(np.float64)
    fused = LOCKED_ALPHA * head + (1.0 - LOCKED_ALPHA) * prototype
    fused[:, 0] *= LOCKED_UNKNOWN_WEIGHT
    fused /= fused.sum(axis=1, keepdims=True) + 1e-12
    predictions = fused.argmax(axis=1).astype(np.int64)
    raw_max = scores.max(axis=1)
    predictions[raw_max < LOCKED_TAU] = 0
    labels = oof["labels"].astype(np.int64)
    head_predictions = head.argmax(axis=1).astype(np.int64)
    prototype_predictions = prototype.argmax(axis=1).astype(np.int64)

    unknown_group_probabilities = internal[:, 1 + NUM_KNOWN:]
    mass_features = unknown_mass_features(unknown_group_probabilities)
    known_probabilities = prototype[:, 1:]
    head_margin = probability_margin(head)
    prototype_margin = probability_margin(prototype)
    fused_margin = probability_margin(fused)
    known_sizes = artifact["known_sizes"].astype(np.int64)
    train = artifact["train_embeddings"].astype(np.float64)
    known_dispersion = np.zeros(NUM_KNOWN, dtype=np.float64)
    for known_id, indices in enumerate(groups[:NUM_KNOWN]):
        members = train[indices]
        centroid = members.mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-12
        known_dispersion[known_id] = float(
            np.mean(1.0 - members @ centroid)
        )

    rows = []
    for index in range(len(labels)):
        label = int(labels[index])
        prediction = int(predictions[index])
        known_scores = scores[index, :NUM_KNOWN]
        true_known_rank = None
        true_known_score = None
        true_known_size = None
        true_known_dispersion = None
        if label > 0:
            true_index = label - 1
            true_known_score = float(known_scores[true_index])
            true_known_rank = int(
                1 + np.sum(known_scores > known_scores[true_index])
            )
            true_known_size = int(known_sizes[true_index])
            true_known_dispersion = float(known_dispersion[true_index])
        winner_known_index = int(np.argmax(known_scores))
        row = {
            "fold": fold,
            "file": str(oof["files"][index]),
            "label": label,
            "prediction": prediction,
            "head_prediction": int(head_predictions[index]),
            "prototype_prediction": int(prototype_predictions[index]),
            "topology": topology_name(label, prediction),
            "head_correct": bool(head_predictions[index] == label),
            "prototype_correct": bool(prototype_predictions[index] == label),
            "head_confidence": float(head[index].max()),
            "head_margin": float(head_margin[index]),
            "prototype_confidence": float(prototype[index].max()),
            "prototype_margin": float(prototype_margin[index]),
            "fused_confidence": float(fused[index].max()),
            "fused_margin": float(fused_margin[index]),
            "raw_max_score": float(raw_max[index]),
            "known_probability_max": float(known_probabilities[index].max()),
            "winner_known_id": winner_known_index + 1,
            "winner_known_size": int(known_sizes[winner_known_index]),
            "winner_known_dispersion": float(
                known_dispersion[winner_known_index]
            ),
            "true_known_rank": true_known_rank,
            "true_known_score": true_known_score,
            "true_known_size": true_known_size,
            "true_known_dispersion": true_known_dispersion,
        }
        for name, values in mass_features.items():
            row[name] = float(values[index])
        rows.append(row)

    diagnostics = {
        "fold": fold,
        "metrics": metric_bundle(labels, predictions),
        "files": int(len(rows)),
        "errors": int(np.sum(labels != predictions)),
        "head_correct_lme_wrong": int(
            np.sum((head_predictions == labels) & (predictions != labels))
        ),
        "head_wrong_lme_correct": int(
            np.sum((head_predictions != labels) & (predictions == labels))
        ),
    }
    return rows, diagnostics


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
        / "campp_lme20_residual_topology.json",
    )
    args = parser.parse_args()

    oofs, artifacts, metadata = load_fold_inputs(args.checkpoint_root, args.cache_dir)
    all_rows = []
    fold_diagnostics = []
    for fold in range(NUM_FOLDS):
        rows, diagnostics = fold_rows(
            fold=fold, artifact=artifacts[fold], oof=oofs[fold]
        )
        all_rows.extend(rows)
        fold_diagnostics.append(diagnostics)

    if len(all_rows) != 4447:
        raise RuntimeError(f"Expected 4447 OOF rows, got {len(all_rows)}")
    topology_summary = {}
    numeric_features = (
        "head_confidence",
        "head_margin",
        "prototype_confidence",
        "prototype_margin",
        "fused_confidence",
        "fused_margin",
        "raw_max_score",
        "known_probability_max",
        "unknown_mass",
        "unknown_top1_fraction",
        "unknown_top16_fraction",
        "unknown_top64_fraction",
        "unknown_effective_clusters",
    )
    for topology in (
        "correct",
        "known_to_unknown",
        "known_to_wrong_known",
        "unknown_to_known",
    ):
        selected = [row for row in all_rows if row["topology"] == topology]
        topology_summary[topology] = {
            "count": len(selected),
            "features": {
                feature: summarise(
                    np.array([row[feature] for row in selected], dtype=np.float64)
                )
                for feature in numeric_features
            },
            "head_correct": int(sum(row["head_correct"] for row in selected)),
            "prototype_correct": int(
                sum(row["prototype_correct"] for row in selected)
            ),
        }

    residual_rows = [row for row in all_rows if row["topology"] != "correct"]
    known_residuals = [row for row in residual_rows if row["label"] > 0]
    report = {
        "contract": {
            "policy": "locked CAM++ LME-20",
            "purpose": "descriptive hypothesis generation only",
            "decision_selection": False,
            "leaderboard_used": False,
        },
        "provenance": {"cache_metadata": metadata},
        "fold_diagnostics": fold_diagnostics,
        "topology_summary": topology_summary,
        "known_residual_rank_summary": {
            "count": len(known_residuals),
            "true_known_rank": summarise(np.array([
                row["true_known_rank"] for row in known_residuals
            ])),
            "true_known_rank1": int(sum(
                row["true_known_rank"] == 1 for row in known_residuals
            )),
            "true_known_rank_le3": int(sum(
                row["true_known_rank"] <= 3 for row in known_residuals
            )),
        },
        "residual_rows": residual_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "fold_diagnostics": fold_diagnostics,
        "topology_counts": {
            name: values["count"] for name, values in topology_summary.items()
        },
        "known_residual_rank_summary": report["known_residual_rank_summary"],
        "topology_feature_medians": {
            name: {
                feature: values["features"][feature]["median"]
                for feature in (
                    "unknown_mass",
                    "unknown_top16_fraction",
                    "unknown_effective_clusters",
                    "head_margin",
                    "prototype_margin",
                )
            }
            for name, values in topology_summary.items()
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
