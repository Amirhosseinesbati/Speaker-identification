"""Compare open-set decision rules on Random, Hard, and Channel stress sets.

The script consumes caches produced by ``dump_decision_evidence.py``.  Decision
thresholds are selected only on a disjoint calibration set (one training file
per known identity plus one training file per historical unknown cluster).
The official historical validation split remains untouched until reporting.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = ROOT / "data" / "processed" / "decision_evidence"
HISTORICAL_DIR = (
    ROOT / "data" / "processed" / "forensics" / "historical_4a47c98"
)
NUM_CLASSES = 447
NUM_KNOWN = 446
EPS = 1e-9


def load_cache(tag: str) -> dict[str, np.ndarray]:
    path = EVIDENCE_DIR / f"{tag}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; run scripts/dump_decision_evidence.py first"
        )
    # The cache is generated locally by dump_decision_evidence.py; pandas may
    # serialize the filename columns as object-string arrays.
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    packed = y_true.astype(np.int64) * NUM_CLASSES + y_pred.astype(np.int64)
    matrix = np.bincount(packed, minlength=NUM_CLASSES**2).reshape(
        NUM_CLASSES, NUM_CLASSES
    )
    tp = np.diag(matrix).astype(np.float64)
    fp = matrix.sum(axis=0) - tp
    fn = matrix.sum(axis=1) - tp
    denominator = 2.0 * tp + fp + fn
    per_class = np.divide(
        2.0 * tp,
        denominator,
        out=np.zeros_like(tp),
        where=denominator > 0,
    )
    return float(per_class.mean())


def report_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    known = y_true > 0
    unknown = ~known
    known_correct = known & (y_pred == y_true)
    return {
        "n": int(len(y_true)),
        "n_known": int(known.sum()),
        "n_unknown": int(unknown.sum()),
        "macro_f1_447": macro_f1(y_true, y_pred),
        "accuracy": float(np.mean(y_true == y_pred)),
        "known_accuracy": float(np.mean(y_pred[known] == y_true[known])),
        "unknown_recall": float(np.mean(y_pred[unknown] == 0)),
        "known_to_unknown": int(np.sum(known & (y_pred == 0))),
        "known_to_wrong_known": int(np.sum(known & (y_pred > 0) & ~known_correct)),
        "unknown_to_known": int(np.sum(unknown & (y_pred > 0))),
    }


def assert_aligned(caches: list[dict[str, np.ndarray]]) -> None:
    reference = caches[0]
    for cache in caches[1:]:
        for key in ("audio_file", "speaker_id", "label", "historical_split"):
            if not np.array_equal(reference[key], cache[key]):
                raise RuntimeError(f"Evidence caches are not aligned on {key}")


def choose_splits(
    reference: dict[str, np.ndarray],
    embeddings: list[np.ndarray],
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Build disjoint calibration and representation-hard known selections."""
    rng = np.random.default_rng(seed)
    labels = reference["label"].astype(np.int64)
    files = reference["audio_file"].astype(str)
    historical_split = reference["historical_split"].astype(str)
    train = historical_split == "train"
    val = historical_split == "val"

    calibration_known: list[int] = []
    hard_known: list[int] = []
    for identity in range(1, NUM_KNOWN + 1):
        candidates = np.flatnonzero(train & (labels == identity))
        if len(candidates) < 2:
            raise RuntimeError(f"Known identity {identity} has <2 train files")
        calibration = int(rng.choice(candidates))
        calibration_known.append(calibration)
        remaining = candidates[candidates != calibration]

        # Leave-one-out distance averaged over all available model spaces.
        difficulty = np.zeros(len(remaining), dtype=np.float64)
        for matrix in embeddings:
            class_embeddings = matrix[candidates].astype(np.float64)
            total = class_embeddings.sum(axis=0)
            for row, index in enumerate(remaining):
                own = matrix[index].astype(np.float64)
                peer_centroid = total - own
                peer_centroid /= np.linalg.norm(peer_centroid) + EPS
                difficulty[row] += 1.0 - float(own @ peer_centroid)
        hard_known.append(int(remaining[np.argmax(difficulty)]))

    cluster_map = json.loads(
        (HISTORICAL_DIR / "unknown_clusters.json").read_text(encoding="utf-8")
    )
    by_cluster: dict[int, list[int]] = {cluster: [] for cluster in range(554)}
    file_to_index = {name: index for index, name in enumerate(files)}
    for name, cluster in cluster_map.items():
        if name in file_to_index:
            by_cluster[int(cluster)].append(file_to_index[name])
    calibration_unknown = []
    for cluster in range(554):
        members = by_cluster[cluster]
        if not members:
            raise RuntimeError(f"Historical unknown cluster {cluster} is empty")
        calibration_unknown.append(int(rng.choice(members)))

    calibration = np.asarray(calibration_known + calibration_unknown, dtype=np.int64)
    random_val = np.flatnonzero(val)
    val_unknown = np.flatnonzero(val & (labels == 0))
    hard = np.asarray(hard_known + val_unknown.tolist(), dtype=np.int64)
    if np.intersect1d(calibration, random_val).size:
        raise RuntimeError("Calibration overlaps official validation")
    if np.intersect1d(calibration, np.asarray(hard_known)).size:
        raise RuntimeError("Calibration overlaps hard-known selection")
    return {
        "calibration": calibration,
        "random_val": random_val,
        "hard": hard,
        "hard_known": np.asarray(hard_known, dtype=np.int64),
        "val_unknown": val_unknown,
    }


def topk_mean(values: np.ndarray, k: int) -> np.ndarray:
    k = min(k, values.shape[1])
    return np.partition(values, values.shape[1] - k, axis=1)[:, -k:].mean(axis=1)


def generalized_logmeanexp_probability(values: np.ndarray, temperature: float) -> np.ndarray:
    """Power mean in probability space; T->0 approaches max, T=1 is mean."""
    power = 1.0 / max(float(temperature), 1e-3)
    return np.mean(np.clip(values, EPS, 1.0) ** power, axis=1) ** (1.0 / power)


def threshold_grid(score: np.ndarray) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, min(2001, max(101, len(score))))
    values = np.unique(np.quantile(score, quantiles))
    return np.concatenate(([values[0] - 1e-6], values, [values[-1] + 1e-6]))


def tune_threshold(
    score: np.ndarray,
    known_identity: np.ndarray,
    labels: np.ndarray,
    calibration: np.ndarray,
) -> tuple[float, float]:
    best_threshold = 0.0
    best_f1 = -1.0
    for threshold in threshold_grid(score[calibration]):
        prediction = known_identity[calibration].copy()
        prediction[score[calibration] > threshold] = 0
        value = macro_f1(labels[calibration], prediction)
        if value > best_f1:
            best_threshold = float(threshold)
            best_f1 = value
    return best_threshold, best_f1


def predictions(score: np.ndarray, known_identity: np.ndarray, threshold: float) -> np.ndarray:
    output = known_identity.copy()
    output[score > threshold] = 0
    return output


def pair_features(
    no_proto: dict[str, np.ndarray],
    metric_only: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    weights = np.asarray([0.6, 0.4], dtype=np.float64)
    speaker = (
        weights[0] * no_proto["speaker_probs"].astype(np.float64)
        + weights[1] * metric_only["speaker_probs"].astype(np.float64)
    )
    known = speaker[:, :NUM_KNOWN]
    tail = speaker[:, NUM_KNOWN:]
    known_order = np.partition(known, -2, axis=1)[:, -2:]
    known_top1 = known_order[:, 1]
    known_top2 = known_order[:, 0]
    known_identity = known.argmax(axis=1).astype(np.int64) + 1
    ood = no_proto["ood_prob"].astype(np.float64)
    head = (
        weights[0] * no_proto["head_probs"].astype(np.float64)
        + weights[1] * metric_only["head_probs"].astype(np.float64)
    )
    agreement = (
        weights[0] * no_proto["window_agreement"].astype(np.float64)
        + weights[1] * metric_only["window_agreement"].astype(np.float64)
    )
    model_known_agreement = (
        no_proto["speaker_probs"][:, :NUM_KNOWN].argmax(axis=1)
        == metric_only["speaker_probs"][:, :NUM_KNOWN].argmax(axis=1)
    ).astype(np.float64)

    common = {
        "ood_logit": np.log(np.clip(ood, EPS, 1 - EPS) / np.clip(1 - ood, EPS, 1)),
        "tail_sum_ratio": np.log(np.clip(tail.sum(axis=1), EPS, 1))
        - np.log(np.clip(known_top1, EPS, 1)),
        "tail_max_ratio": np.log(np.clip(tail.max(axis=1), EPS, 1))
        - np.log(np.clip(known_top1, EPS, 1)),
        "tail_top3_ratio": np.log(np.clip(topk_mean(tail, 3), EPS, 1))
        - np.log(np.clip(known_top1, EPS, 1)),
        "tail_top5_ratio": np.log(np.clip(topk_mean(tail, 5), EPS, 1))
        - np.log(np.clip(known_top1, EPS, 1)),
        "known_margin": np.log(np.clip(known_top1, EPS, 1))
        - np.log(np.clip(known_top2, EPS, 1)),
        "window_agreement": agreement,
        "model_known_agreement": model_known_agreement,
        "head_unknown_ratio": np.log(np.clip(head[:, 0], EPS, 1))
        - np.log(np.clip(head[:, 1:].max(axis=1), EPS, 1)),
    }
    return known_identity, head, common | {"tail": tail, "known_top1": known_top1}


def evaluate_rules(
    no_proto: dict[str, np.ndarray],
    metric_only: dict[str, np.ndarray],
    split: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    labels = no_proto["label"].astype(np.int64)
    known_identity, head, feature = pair_features(no_proto, metric_only)
    rule_scores: dict[str, np.ndarray] = {
        "sum_current": feature["head_unknown_ratio"],
        "tail_max_plus_ood": feature["tail_max_ratio"] + 0.5 * feature["ood_logit"],
        "tail_top3_mean_plus_ood": feature["tail_top3_ratio"] + 0.5 * feature["ood_logit"],
        "tail_top5_mean_plus_ood": feature["tail_top5_ratio"] + 0.5 * feature["ood_logit"],
    }
    for temperature in (0.25, 0.5, 1.0):
        evidence = generalized_logmeanexp_probability(feature["tail"], temperature)
        rule_scores[f"tail_logmeanexp_t{temperature:g}_plus_ood"] = (
            np.log(np.clip(evidence, EPS, 1))
            - np.log(np.clip(feature["known_top1"], EPS, 1))
            + 0.5 * feature["ood_logit"]
        )

    # A low-capacity learned score tests whether the independent evidence is
    # complementary.  It is fit only on calibration and its threshold is tuned
    # on that same calibration; generalisation is judged solely on held-out sets.
    linear_names = [
        "ood_logit",
        "tail_sum_ratio",
        "tail_max_ratio",
        "tail_top5_ratio",
        "known_margin",
        "window_agreement",
        "model_known_agreement",
    ]
    matrix = np.column_stack([feature[name] for name in linear_names])
    scaler = StandardScaler().fit(matrix[split["calibration"]])
    scaled = scaler.transform(matrix)
    classifier = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced")
    classifier.fit(
        scaled[split["calibration"]],
        (labels[split["calibration"]] == 0).astype(np.int64),
    )
    rule_scores["linear_evidence"] = classifier.decision_function(scaled)

    results: dict[str, Any] = {
        "linear_model": {
            "features": linear_names,
            "coefficients": classifier.coef_[0].tolist(),
            "intercept": float(classifier.intercept_[0]),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
        },
        "rules": {},
    }
    rule_predictions: dict[str, np.ndarray] = {}

    # Exact package behaviour is reported without retuning.
    package_prediction = head.argmax(axis=1).astype(np.int64)
    results["rules"]["package_argmax"] = {
        "threshold": None,
        "calibration_macro_f1": None,
        "sets": {
            name: report_metrics(labels[indices], package_prediction[indices])
            for name, indices in split.items()
            if name in {"random_val", "hard"}
        },
    }
    rule_predictions["package_argmax"] = package_prediction

    for name, score in rule_scores.items():
        threshold, calibration_f1 = tune_threshold(
            score, known_identity, labels, split["calibration"]
        )
        prediction = predictions(score, known_identity, threshold)
        results["rules"][name] = {
            "threshold": threshold,
            "calibration_macro_f1": calibration_f1,
            "sets": {
                set_name: report_metrics(labels[indices], prediction[indices])
                for set_name, indices in split.items()
                if set_name in {"random_val", "hard"}
            },
        }
        rule_predictions[name] = prediction
    return results, rule_predictions


def softmax(values: np.ndarray, axis: int = 1) -> np.ndarray:
    shifted = values - values.max(axis=axis, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / (exponent.sum(axis=axis, keepdims=True) + EPS)


def historical_submission_metrics(
    historical: dict[str, np.ndarray],
    split: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Evaluate the exact Aug-19 rule, with LOO correction on hard train files."""
    with np.load(HISTORICAL_DIR / "centroids_campp.npz") as data:
        known_centroids = data["centroids"].astype(np.float64)
    with np.load(HISTORICAL_DIR / "centroids_unknown_campp.npz") as data:
        unknown_centroids = data["centroids"].astype(np.float64)
    embeddings = historical["embedding"].astype(np.float64)
    labels = historical["label"].astype(np.int64)
    train = historical["historical_split"].astype(str) == "train"

    cosine_known = embeddings @ known_centroids.T
    cosine_unknown = embeddings @ unknown_centroids.T
    cosine_known_hard = cosine_known.copy()
    for index in split["hard_known"]:
        identity = int(labels[index])
        peers = np.flatnonzero(train & (labels == identity))
        peers = peers[peers != index]
        centroid = embeddings[peers].mean(axis=0)
        centroid /= np.linalg.norm(centroid) + EPS
        cosine_known_hard[index, identity - 1] = embeddings[index] @ centroid

    def predict(indices: np.ndarray, leave_one_out: bool) -> np.ndarray:
        known_cos = cosine_known_hard[indices] if leave_one_out else cosine_known[indices]
        cosine = np.hstack([known_cos, cosine_unknown[indices]])
        maximum = cosine.max(axis=1)
        mass = softmax(24.0 * cosine)
        distance_unknown = np.clip(1.0 - maximum, 0.0, 1.0)
        centroid_probs = np.zeros((len(indices), NUM_CLASSES), dtype=np.float64)
        centroid_probs[:, 0] = (
            distance_unknown
            + (1.0 - distance_unknown) * mass[:, NUM_KNOWN:].sum(axis=1)
        )
        centroid_probs[:, 1:] = (1.0 - distance_unknown[:, None]) * mass[:, :NUM_KNOWN]
        fused = (
            0.35 * historical["head_probs"][indices].astype(np.float64)
            + 0.65 * centroid_probs
        )
        fused[:, 0] *= 0.5
        fused /= fused.sum(axis=1, keepdims=True) + EPS
        return fused.argmax(axis=1).astype(np.int64)

    random_prediction = predict(split["random_val"], leave_one_out=False)
    hard_prediction = predict(split["hard"], leave_one_out=True)
    return {
        "checkpoint_sha256_12": "ff5108b0e037",
        "decision": {"alpha": 0.35, "kappa": 24.0, "tau": 0.0, "lambda_unknown": 0.5},
        "random_val": report_metrics(
            labels[split["random_val"]], random_prediction,
        ),
        "hard_leave_one_out": report_metrics(
            labels[split["hard"]], hard_prediction,
        ),
    }


def channel_breakdown(
    reference: dict[str, np.ndarray],
    predictions_by_rule: dict[str, np.ndarray],
) -> dict[str, Any]:
    path = ROOT / "data" / "processed" / "val_acoustic_features.csv"
    if not path.exists():
        return {"error": f"missing {path}"}
    acoustic = pd.read_csv(path)
    index_by_file = {
        str(name): index for index, name in enumerate(reference["audio_file"].astype(str))
    }
    acoustic["evidence_index"] = acoustic["audio_file"].astype(str).map(index_by_file)
    acoustic = acoustic[acoustic["evidence_index"].notna()].copy()
    acoustic["evidence_index"] = acoustic["evidence_index"].astype(int)
    labels = reference["label"].astype(np.int64)
    output: dict[str, Any] = {}
    for bucket, rows in acoustic.groupby("bucket"):
        indices = rows["evidence_index"].to_numpy(dtype=np.int64)
        output[str(bucket)] = {
            rule: report_metrics(labels[indices], prediction[indices])
            for rule, prediction in predictions_by_rule.items()
        }
    return output


def select_recommendation(results: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for name, payload in results["rules"].items():
        random_metrics = payload["sets"]["random_val"]
        hard_metrics = payload["sets"]["hard"]
        rows.append(
            {
                "rule": name,
                "random_macro_f1": random_metrics["macro_f1_447"],
                "hard_macro_f1": hard_metrics["macro_f1_447"],
                "random_unknown_recall": random_metrics["unknown_recall"],
                "hard_unknown_recall": hard_metrics["unknown_recall"],
                "robust_score": min(
                    random_metrics["macro_f1_447"], hard_metrics["macro_f1_447"]
                ),
            }
        )
    ranked = sorted(rows, key=lambda row: row["robust_score"], reverse=True)
    accepted = [
        row for row in ranked
        if row["random_unknown_recall"] >= 0.98
        and row["hard_unknown_recall"] >= 0.98
    ]
    return {
        "ranking": ranked,
        "accepted_unknown_recall_ge_0.98": accepted,
        "recommended": accepted[0] if accepted else None,
    }


def stability_across_seeds(
    historical: dict[str, np.ndarray],
    no_proto: dict[str, np.ndarray],
    metric_only: dict[str, np.ndarray],
    seeds: tuple[int, ...] = (0, 1, 2, 17, 42, 123, 2026),
) -> dict[str, Any]:
    collected: dict[str, list[dict[str, float]]] = {}
    for seed in seeds:
        split = choose_splits(
            historical,
            [
                historical["embedding"],
                no_proto["embedding"],
                metric_only["embedding"],
            ],
            seed=seed,
        )
        result, _ = evaluate_rules(no_proto, metric_only, split)
        package_random = result["rules"]["package_argmax"]["sets"]["random_val"]
        package_hard = result["rules"]["package_argmax"]["sets"]["hard"]
        for name, payload in result["rules"].items():
            random_metrics = payload["sets"]["random_val"]
            hard_metrics = payload["sets"]["hard"]
            collected.setdefault(name, []).append(
                {
                    "seed": float(seed),
                    "threshold": (
                        float(payload["threshold"])
                        if payload["threshold"] is not None else math.nan
                    ),
                    "random_macro_f1": random_metrics["macro_f1_447"],
                    "hard_macro_f1": hard_metrics["macro_f1_447"],
                    "random_unknown_recall": random_metrics["unknown_recall"],
                    "hard_unknown_recall": hard_metrics["unknown_recall"],
                    "delta_random_vs_package": (
                        random_metrics["macro_f1_447"] - package_random["macro_f1_447"]
                    ),
                    "delta_hard_vs_package": (
                        hard_metrics["macro_f1_447"] - package_hard["macro_f1_447"]
                    ),
                }
            )

    summary: dict[str, Any] = {}
    for name, rows in collected.items():
        aggregate: dict[str, Any] = {"per_seed": rows}
        for metric in (
            "random_macro_f1",
            "hard_macro_f1",
            "random_unknown_recall",
            "hard_unknown_recall",
            "delta_random_vs_package",
            "delta_hard_vs_package",
            "threshold",
        ):
            values = np.asarray([row[metric] for row in rows], dtype=np.float64)
            aggregate[metric] = {
                "mean": float(values.mean()),
                "min": float(values.min()),
                "max": float(values.max()),
                "std": float(values.std()),
            }
        aggregate["passes_stability_gate"] = bool(
            aggregate["delta_random_vs_package"]["min"] >= -1e-12
            and aggregate["delta_hard_vs_package"]["min"] >= -1e-12
            and aggregate["random_unknown_recall"]["min"] >= 0.98
            and aggregate["hard_unknown_recall"]["min"] >= 0.98
        )
        summary[name] = aggregate
    passing = [
        name for name, payload in summary.items()
        if payload["passes_stability_gate"] and name != "package_argmax"
    ]
    passing.sort(
        key=lambda name: min(
            summary[name]["random_macro_f1"]["mean"],
            summary[name]["hard_macro_f1"]["mean"],
        ),
        reverse=True,
    )
    return {
        "seeds": list(seeds),
        "gate": "non-negative Random and Hard delta on every seed; unknown recall >= 0.98",
        "rules": summary,
        "passing_rules": passing,
        "recommended_stable_rule": passing[0] if passing else None,
    }


def fixed_top5_candidate_stability(
    historical: dict[str, np.ndarray],
    no_proto: dict[str, np.ndarray],
    metric_only: dict[str, np.ndarray],
    threshold: float = -2.279,
    seeds: tuple[int, ...] = (0, 1, 2, 17, 42, 123, 2026),
) -> dict[str, Any]:
    labels = historical["label"].astype(np.int64)
    known_identity, _, feature = pair_features(no_proto, metric_only)
    score = feature["tail_top5_ratio"] + 0.5 * feature["ood_logit"]
    prediction = predictions(score, known_identity, threshold)
    rows = []
    for seed in seeds:
        split = choose_splits(
            historical,
            [
                historical["embedding"],
                no_proto["embedding"],
                metric_only["embedding"],
            ],
            seed=seed,
        )
        package = (
            0.6 * no_proto["head_probs"].astype(np.float64)
            + 0.4 * metric_only["head_probs"].astype(np.float64)
        ).argmax(axis=1).astype(np.int64)
        hard = report_metrics(labels[split["hard"]], prediction[split["hard"]])
        package_hard = report_metrics(labels[split["hard"]], package[split["hard"]])
        rows.append({
            "seed": seed,
            "hard_macro_f1": hard["macro_f1_447"],
            "hard_known_accuracy": hard["known_accuracy"],
            "hard_unknown_recall": hard["unknown_recall"],
            "delta_hard_vs_package": hard["macro_f1_447"] - package_hard["macro_f1_447"],
        })
    random_indices = np.flatnonzero(
        historical["historical_split"].astype(str) == "val"
    )
    random_metrics = report_metrics(labels[random_indices], prediction[random_indices])
    return {
        "rule": "tail_top5_mean_plus_ood",
        "threshold": threshold,
        "threshold_source": "seven-seed median plus 0.001 FP16/batch-size safety margin",
        "random_val": random_metrics,
        "per_seed_hard": rows,
        "hard_macro_f1": {
            "mean": float(np.mean([row["hard_macro_f1"] for row in rows])),
            "min": float(np.min([row["hard_macro_f1"] for row in rows])),
            "max": float(np.max([row["hard_macro_f1"] for row in rows])),
        },
        "delta_hard_vs_package": {
            "mean": float(np.mean([row["delta_hard_vs_package"] for row in rows])),
            "min": float(np.min([row["delta_hard_vs_package"] for row in rows])),
            "max": float(np.max([row["delta_hard_vs_package"] for row in rows])),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "generated" / "open_set_decision_ablation.json",
    )
    parser.add_argument(
        "--errors-csv",
        type=Path,
        default=ROOT / "reports" / "generated" / "open_set_decision_errors.csv",
    )
    args = parser.parse_args()

    historical = load_cache("historical")
    no_proto = load_cache("no_proto")
    metric_only = load_cache("metric_only")
    assert_aligned([historical, no_proto, metric_only])
    split = choose_splits(
        historical,
        [
            historical["embedding"],
            no_proto["embedding"],
            metric_only["embedding"],
        ],
    )
    pair_results, predictions_by_rule = evaluate_rules(no_proto, metric_only, split)
    historical_results = historical_submission_metrics(historical, split)
    recommendation = select_recommendation(pair_results)
    channel = channel_breakdown(historical, predictions_by_rule)
    stability = stability_across_seeds(historical, no_proto, metric_only)
    fixed_candidate = fixed_top5_candidate_stability(
        historical, no_proto, metric_only,
    )

    report = {
        "generated_at": datetime.now().isoformat(),
        "protocol": {
            "calibration": "one historical-train file per known identity plus one per historical unknown cluster",
            "random_val": "the untouched 891-file validation used before leaderboard submissions",
            "hard": "one disjoint, maximum leave-one-out embedding-distance train file per known identity plus untouched validation unknowns",
            "channel": "the untouched validation partitioned by acoustic EDA bucket",
            "selection_guard": "all thresholds and the linear score are fit only on calibration",
        },
        "split_sizes": {name: int(len(indices)) for name, indices in split.items()},
        "pair_60_40": pair_results,
        "historical_0_9625_control": historical_results,
        "recommendation": recommendation,
        "stability": stability,
        "fixed_submission_candidate": fixed_candidate,
        "channel_breakdown": channel,
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    labels = historical["label"].astype(np.int64)
    files = historical["audio_file"].astype(str)
    rows = []
    for rule, prediction in predictions_by_rule.items():
        for set_name in ("random_val", "hard"):
            for index in split[set_name]:
                if prediction[index] != labels[index]:
                    rows.append(
                        {
                            "rule": rule,
                            "set": set_name,
                            "audio_file": files[index],
                            "true_label": int(labels[index]),
                            "predicted_label": int(prediction[index]),
                            "error_type": (
                                "known_to_unknown"
                                if labels[index] > 0 and prediction[index] == 0
                                else "unknown_to_known"
                                if labels[index] == 0 and prediction[index] > 0
                                else "known_to_wrong_known"
                            ),
                        }
                    )
    errors_path = args.errors_csv if args.errors_csv.is_absolute() else ROOT / args.errors_csv
    pd.DataFrame(rows).to_csv(errors_path, index=False)

    print(json.dumps({
        "split_sizes": report["split_sizes"],
        "recommendation": recommendation,
        "output": str(output.relative_to(ROOT)),
        "errors_csv": str(errors_path.relative_to(ROOT)),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
