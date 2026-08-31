"""Evaluate a fixed spherical-PLDA known-speaker reranker on one OOF fold.

The two-scalar spherical-PLDA equations follow Sholokhov et al. (ICASSP
2023) and the authors' MIT-licensed reference implementation; see
``THIRD_PARTY_NOTICES.md``.  This phase-one
audit deliberately does *not* calibrate an OOD threshold: it preserves every
raw CAM++ known/unknown decision and only replaces the identity of rows that
CAM++ already predicts as known.  Consequently any OOD-F1 movement is a hard
implementation error rather than an optimisation opportunity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.analyze_control_oof_centroid_crossfit import (
    NUM_CLASSES,
    NUM_KNOWN,
    metric_bundle,
    metric_delta,
)


EM_ITERATIONS = 10
GATE = {
    "macro_f1_gain_min": 0.001,
    "known_accuracy_gain_min": 0.001,
    "ood_f1_absolute_delta_max": 1e-12,
    "rescued_errors_must_exceed_introduced_errors": True,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def center_and_normalize(values: np.ndarray, center: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64) - np.asarray(
        center, dtype=np.float64
    )
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise RuntimeError("Centered embedding has near-zero norm")
    return values / norms


def fit_spherical_plda(
    embeddings: np.ndarray, labels: np.ndarray, n_iter: int = EM_ITERATIONS
) -> tuple[float, float]:
    """Fit the paper's isotropic between/within variances by EM."""

    x = np.asarray(embeddings, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if x.ndim != 2 or len(x) != len(y) or len(x) == 0:
        raise RuntimeError("Invalid spherical-PLDA training arrays")
    classes, inverse = np.unique(y, return_inverse=True)
    if len(classes) < 2:
        raise RuntimeError("Spherical PLDA requires at least two speakers")
    counts = np.bincount(inverse, minlength=len(classes)).astype(np.float64)
    sums = np.zeros((len(classes), x.shape[1]), dtype=np.float64)
    np.add.at(sums, inverse, x)
    means = sums / counts[:, None]
    residuals = x - means[inverse]
    w = float(np.mean(residuals ** 2))
    b = float(np.mean(means ** 2))
    if b <= 0.0 or w <= 0.0:
        raise RuntimeError(f"Degenerate spherical-PLDA initialisation b={b}, w={w}")
    for _ in range(int(n_iter)):
        sigma = 1.0 / (1.0 / b + counts / w)
        means = sigma[:, None] * (sums / w)
        b = float(np.mean(sigma) + np.mean(means ** 2))
        residuals = x - means[inverse]
        w = float(np.mean(residuals ** 2) + np.sum(counts * sigma) / len(x))
    if not np.isfinite([b, w]).all() or b <= 0.0 or w <= 0.0:
        raise RuntimeError(f"Invalid fitted spherical-PLDA parameters b={b}, w={w}")
    return b, w


def spherical_plda_score_matrix(
    enrollment_centroids: np.ndarray,
    enrollment_counts: np.ndarray,
    test_embeddings: np.ndarray,
    b: float,
    w: float,
) -> np.ndarray:
    """Set-to-vector LLR matrix equivalent to the reference implementation."""

    enroll = np.asarray(enrollment_centroids, dtype=np.float64)
    counts = np.asarray(enrollment_counts, dtype=np.float64).reshape(-1, 1)
    tests = np.asarray(test_embeddings, dtype=np.float64)
    if enroll.ndim != 2 or tests.ndim != 2 or enroll.shape[1] != tests.shape[1]:
        raise RuntimeError("Spherical-PLDA enrollment/test dimension mismatch")
    if len(enroll) != len(counts) or np.any(counts <= 0):
        raise RuntimeError("Invalid enrollment counts")
    b_inv = 1.0 / float(b)
    w_inv = 1.0 / float(w)
    test_counts = np.ones((1, len(tests)), dtype=np.float64)
    a_enroll = counts * enroll * w_inv
    a_test = tests * w_inv
    sigma_e_inv = b_inv + counts * w_inv
    sigma_t_inv = b_inv + test_counts * w_inv
    sigma_e = 1.0 / sigma_e_inv
    sigma_t = 1.0 / sigma_t_inv
    sigma_inv_sum = sigma_e_inv + sigma_t_inv - b_inv
    dim = enroll.shape[1]
    constant = dim * (
        0.5 * np.log(b)
        + 0.5 * np.log(sigma_e_inv)
        + 0.5 * np.log(sigma_t_inv)
        - 0.5 * np.log(sigma_inv_sum)
    )
    a_e_sq = np.sum(a_enroll ** 2, axis=1, keepdims=True)
    a_t_sq = np.sum(a_test ** 2, axis=1, keepdims=True).T
    separate = -0.5 * (a_e_sq * sigma_e + a_t_sq * sigma_t)
    joint_sq = a_e_sq + a_t_sq + 2.0 * (a_enroll @ a_test.T)
    scores = 0.5 * joint_sq / sigma_inv_sum + separate + constant
    if scores.shape != (len(enroll), len(tests)) or not np.isfinite(scores).all():
        raise RuntimeError("Invalid spherical-PLDA score matrix")
    return scores


def build_known_enrollment(
    embeddings: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    centroids = []
    counts = []
    for speaker in range(1, NUM_KNOWN + 1):
        members = embeddings[labels == speaker]
        if len(members) == 0:
            raise RuntimeError(f"Missing known enrollment speaker {speaker}")
        centroids.append(members.mean(axis=0))
        counts.append(len(members))
    return np.asarray(centroids), np.asarray(counts, dtype=np.int64)


def collapse_labels(labels: np.ndarray) -> np.ndarray:
    values = np.asarray(labels, dtype=np.int64)
    return np.where(values > NUM_KNOWN, 0, values).astype(np.int64)


def rerank_known_only(
    competition_probs: np.ndarray, known_scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.asarray(competition_probs, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != NUM_CLASSES:
        raise RuntimeError("Unexpected competition probability shape")
    baseline = probabilities.argmax(axis=1).astype(np.int64)
    if known_scores.shape != (NUM_KNOWN, len(baseline)):
        raise RuntimeError("Unexpected known score shape")
    candidate = baseline.copy()
    predicted_known = baseline > 0
    candidate[predicted_known] = (
        known_scores[:, predicted_known].argmax(axis=0).astype(np.int64) + 1
    )
    if not np.array_equal(candidate == 0, baseline == 0):
        raise RuntimeError("Known reranker changed an OOD decision")
    return baseline, candidate


def transition_bundle(
    labels: np.ndarray, baseline: np.ndarray, candidate: np.ndarray
) -> dict[str, int | float]:
    baseline_correct = baseline == labels
    candidate_correct = candidate == labels
    rescued = ~baseline_correct & candidate_correct
    introduced = baseline_correct & ~candidate_correct
    errors = int(np.sum(~baseline_correct))
    return {
        "baseline_errors": errors,
        "rescued_errors": int(np.sum(rescued)),
        "introduced_errors": int(np.sum(introduced)),
        "rescue_rate": float(np.sum(rescued) / max(errors, 1)),
    }


def evaluate(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    validation_embeddings: np.ndarray,
    labels: np.ndarray,
    competition_probs: np.ndarray,
) -> dict:
    collapsed_train = collapse_labels(train_labels)
    known = collapsed_train > 0
    center = np.asarray(train_embeddings, dtype=np.float64)[known].mean(axis=0)
    train = center_and_normalize(train_embeddings[known], center)
    validation = center_and_normalize(validation_embeddings, center)
    known_labels = collapsed_train[known]
    b, w = fit_spherical_plda(train, known_labels)
    enrollment, counts = build_known_enrollment(train, known_labels)
    scores = spherical_plda_score_matrix(enrollment, counts, validation, b, w)
    baseline_predictions, candidate_predictions = rerank_known_only(
        competition_probs, scores
    )
    collapsed_validation = collapse_labels(labels)
    baseline = metric_bundle(collapsed_validation, baseline_predictions)
    candidate = metric_bundle(collapsed_validation, candidate_predictions)
    delta = metric_delta(candidate, baseline)
    transitions = transition_bundle(
        collapsed_validation, baseline_predictions, candidate_predictions
    )
    ood_delta = abs(delta["ood_f1"])
    checks = {
        "macro_gain": delta["macro_f1"] >= GATE["macro_f1_gain_min"],
        "known_gain": delta["known_accuracy"] >= GATE["known_accuracy_gain_min"],
        "ood_decision_invariant": ood_delta <= GATE["ood_f1_absolute_delta_max"],
        "rescued_exceed_introduced": (
            transitions["rescued_errors"] > transitions["introduced_errors"]
        ),
    }
    return {
        "parameters": {"b": b, "w": w, "em_iterations": EM_ITERATIONS},
        "enrollment_count_min": int(counts.min()),
        "enrollment_count_max": int(counts.max()),
        "baseline": baseline,
        "candidate": candidate,
        "delta": delta,
        "transitions": transitions,
        "gate": {"contract": dict(GATE), "checks": checks, "passed": all(checks.values())},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--expected-fold", type=int, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    metadata_path = args.train_cache.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("checkpoint_sha256") != args.expected_checkpoint_sha256:
        raise RuntimeError("Train cache checkpoint SHA mismatch")
    if int(metadata.get("fold", -1)) != args.expected_fold:
        raise RuntimeError("Train cache fold mismatch")
    with np.load(args.train_cache, allow_pickle=False) as archive:
        train_embeddings = archive["train_embeddings"].copy()
        train_labels = archive["competition_labels"].copy()
    with np.load(args.oof, allow_pickle=False) as archive:
        required = {"labels", "competition_probs", "embeddings", "split_fold", "split_folds", "split_seed"}
        missing = required - set(archive.files)
        if missing:
            raise RuntimeError(f"OOF lacks keys: {sorted(missing)}")
        if int(archive["split_fold"].reshape(-1)[0]) != args.expected_fold:
            raise RuntimeError("OOF fold mismatch")
        if int(archive["split_folds"].reshape(-1)[0]) != 3:
            raise RuntimeError("OOF fold-count mismatch")
        if int(archive["split_seed"].reshape(-1)[0]) != 42:
            raise RuntimeError("OOF seed mismatch")
        labels = archive["labels"].copy()
        probabilities = archive["competition_probs"].copy()
        validation_embeddings = archive["embeddings"].copy()
    result = evaluate(
        train_embeddings,
        train_labels,
        validation_embeddings,
        labels,
        probabilities,
    )
    report = {
        "contract": {
            "method": "fixed_method_faithful_spherical_plda_known_rerank",
            "ood_decision": "unchanged_raw_probability_argmax",
            "search_dimensions": 0,
            "submission_authorized": False,
        },
        "provenance": {
            "train_cache": str(args.train_cache.resolve()),
            "train_cache_sha256": sha256_file(args.train_cache),
            "train_cache_metadata_sha256": sha256_file(metadata_path),
            "oof": str(args.oof.resolve()),
            "oof_sha256": sha256_file(args.oof),
            "checkpoint_sha256": args.expected_checkpoint_sha256,
            "fold": args.expected_fold,
            "folds": 3,
            "seed": 42,
        },
        "evaluation": result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
