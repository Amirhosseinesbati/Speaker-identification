"""Leak-aware, no-retraining evaluation of an open-set calibration layer.

This experiment uses only already-dumped validation probabilities, embeddings,
centroids and the pseudo-speaker map.  It never updates an audio encoder or a
speaker classifier.  A small logistic-regression *decision layer* is fitted
only inside each CV training fold to decide ``known`` vs ``unknown``; the
identity itself is always the existing decision layer's best known identity.

Two protocols are reported:
  * ordinary stratified CV -- allows the same pseudo-unknown speaker in train
    and test, so it is intentionally an optimistic reference;
  * group-aware CV -- keeps each pseudo-unknown speaker cluster wholly in one
    fold.  This is the useful estimate for generalising to unseen OOD people.

Usage:
    uv run --no-sync python scripts/evaluate_open_set_calibration.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.cli_utils import setup_utf8_stdio  # noqa: E402
from src.decision_engine import load_decision_artifacts  # noqa: E402
from src.metrics import macro_f1_score  # noqa: E402
from submission.inference import centroid_probs_matrix, _collapse_centroid_probs  # noqa: E402

setup_utf8_stdio()
DATA = ROOT / "data" / "processed"
OUT = ROOT / "reports" / "generated" / "open_set_calibration_report.json"
NUM_CLASSES = 447


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)


def _decision_arrays(artifacts: dict, params: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return final probabilities, max cosine, centroid margin, head entropy."""
    weights = artifacts["weights"]
    heads = np.tensordot(weights, np.stack(artifacts["probs"]), axes=(0, 0))
    cps, max_scores, margins = [], [], []
    for emb, cents, sids in zip(artifacts["emb"], artifacts["centroids"], artifacts["speaker_ids"]):
        cols = int(sids.max()) + 1
        cp, _ = centroid_probs_matrix(emb, cents, sids, cols, params["kappa"])
        cosine = emb @ cents.T
        ordered = np.partition(cosine, -2, axis=1)[:, -2:]
        margins.append(ordered[:, 1] - ordered[:, 0])
        max_scores.append(cosine.max(axis=1))
        if cols > NUM_CLASSES:
            cp = _collapse_centroid_probs(cp, NUM_CLASSES)
        cps.append(cp)
    cents = np.tensordot(weights, np.stack(cps), axes=(0, 0))
    max_cosine = np.tensordot(weights, np.stack(max_scores), axes=(0, 0))
    centroid_margin = np.tensordot(weights, np.stack(margins), axes=(0, 0))
    fused = params["alpha"] * heads + (1.0 - params["alpha"]) * cents
    fused[:, 0] *= params["lambda_unknown"]
    fused /= fused.sum(axis=1, keepdims=True) + 1e-12
    entropy = -(heads * np.log(heads + 1e-12)).sum(axis=1)
    return fused, max_cosine, centroid_margin, entropy


def _features(final_probs: np.ndarray, max_cos: np.ndarray, centroid_margin: np.ndarray,
              entropy: np.ndarray) -> np.ndarray:
    known = final_probs[:, 1:]
    top2 = np.partition(known, -2, axis=1)[:, -2:]
    return np.column_stack([
        final_probs[:, 0],                 # already fused unknown mass
        known.max(axis=1),                  # known confidence
        top2[:, 1] - top2[:, 0],            # known top-1/top-2 margin
        max_cos,                            # nearest prototype similarity
        centroid_margin,                    # nearest-centroid ambiguity
        entropy,                            # head uncertainty
    ])


def _best_threshold(p_unknown: np.ndarray, labels: np.ndarray, known_pred: np.ndarray) -> float:
    """Tune only on the CV train fold, against the real 447-class Macro-F1."""
    best_t, best = 0.5, -1.0
    for threshold in np.arange(0.05, 0.951, 0.01):
        pred = known_pred.copy()
        pred[p_unknown >= threshold] = 0
        score = macro_f1_score(labels, pred, NUM_CLASSES)
        if score > best:
            best, best_t = score, float(threshold)
    return best_t


def _evaluate_cv(name: str, splitter, X: np.ndarray, binary_y: np.ndarray,
                 labels: np.ndarray, known_pred: np.ndarray, groups=None) -> dict:
    oof_prob = np.zeros(len(labels), dtype=np.float64)
    oof_pred = np.zeros(len(labels), dtype=np.int64)
    thresholds = []
    folds = splitter.split(X, binary_y, groups) if groups is not None else splitter.split(X, binary_y)
    for train_idx, test_idx in folds:
        clf = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000,
                                 random_state=42)
        clf.fit(X[train_idx], binary_y[train_idx])
        p_train = clf.predict_proba(X[train_idx])[:, 0]  # class 0 = unknown
        threshold = _best_threshold(p_train, labels[train_idx], known_pred[train_idx])
        oof_prob[test_idx] = clf.predict_proba(X[test_idx])[:, 0]
        oof_pred[test_idx] = known_pred[test_idx]
        oof_pred[test_idx[oof_prob[test_idx] >= threshold]] = 0
        thresholds.append(threshold)
    return {
        "protocol": name,
        "macro_f1": float(macro_f1_score(labels, oof_pred, NUM_CLASSES)),
        "unknown_recall": float((oof_pred[labels == 0] == 0).mean()),
        "known_accuracy": float((oof_pred[labels > 0] == labels[labels > 0]).mean()),
        "thresholds": thresholds,
        "mean_predicted_unknown_probability": {
            "unknown": float(oof_prob[labels == 0].mean()),
            "known": float(oof_prob[labels > 0].mean()),
        },
    }


def _evaluate_simple_cv(name: str, splitter, unknown_score: np.ndarray,
                        labels: np.ndarray, known_pred: np.ndarray, groups=None) -> dict:
    """Fair CV baseline: tune one existing unknown score, no learned layer."""
    oof_pred = np.zeros(len(labels), dtype=np.int64)
    thresholds = []
    folds = splitter.split(unknown_score, labels == 0, groups) if groups is not None else splitter.split(unknown_score, labels == 0)
    for train_idx, test_idx in folds:
        threshold = _best_threshold(unknown_score[train_idx], labels[train_idx], known_pred[train_idx])
        oof_pred[test_idx] = known_pred[test_idx]
        oof_pred[test_idx[unknown_score[test_idx] >= threshold]] = 0
        thresholds.append(threshold)
    return {
        "protocol": name,
        "macro_f1": float(macro_f1_score(labels, oof_pred, NUM_CLASSES)),
        "unknown_recall": float((oof_pred[labels == 0] == 0).mean()),
        "known_accuracy": float((oof_pred[labels > 0] == labels[labels > 0]).mean()),
        "thresholds": thresholds,
    }


def main() -> int:
    params = json.loads((DATA / "decision_config.json").read_text(encoding="utf-8"))["decision_params"]
    artifacts = load_decision_artifacts()
    labels = artifacts["labels"]
    files = json.loads((DATA / "val_campp_vad_files.json").read_text(encoding="utf-8"))
    if len(files) != len(labels):
        raise RuntimeError("Validation file list and labels are misaligned; refusing to score.")
    final_probs, max_cos, centroid_margin, entropy = _decision_arrays(artifacts, params)
    X = _features(final_probs, max_cos, centroid_margin, entropy)
    binary_y = (labels == 0).astype(np.int64)  # sklearn class order: 0 known, 1 unknown
    # Make class 0 be unknown so predict_proba[:, 0] convention below is true.
    binary_y = 1 - binary_y
    # Pseudo-cluster ids are trusted only as a grouping proxy; each known item
    # receives its own group because there is one held-out item per known id.
    # The map deliberately contains only *training* unknown files: validation
    # files must never be used while discovering pseudo identities.  Assign a
    # validation unknown sample to its nearest training-only CAM++ cluster
    # centroid, then keep all samples assigned to that centroid in one fold.
    # This is conservative only insofar as the pseudo clusters are accurate.
    campp_index = artifacts["encoder_names"].index("campp")
    unknown_centroids = np.load(DATA / "centroids_unknown_campp.npz")["centroids"].astype(np.float32)
    val_to_unknown_cluster = (artifacts["emb"][campp_index] @ unknown_centroids.T).argmax(axis=1)
    groups = np.array([
        f"u{val_to_unknown_cluster[i]}" if labels[i] == 0 else f"k{i}"
        for i in range(len(files))
    ])
    known_pred = final_probs[:, 1:].argmax(axis=1).astype(np.int64) + 1
    # The ratio makes the existing argmax gate explicit: raw argmax is exactly
    # equivalent to this score being at least 0.5.
    simple_unknown_score = final_probs[:, 0] / (final_probs[:, 0] + final_probs[:, 1:].max(axis=1) + 1e-12)
    baseline_pred = final_probs.argmax(axis=1).astype(np.int64)
    baseline = {
        "macro_f1": float(macro_f1_score(labels, baseline_pred, NUM_CLASSES)),
        "unknown_recall": float((baseline_pred[labels == 0] == 0).mean()),
        "known_accuracy": float((baseline_pred[labels > 0] == labels[labels > 0]).mean()),
    }
    # StratifiedGroupKFold needs a recent sklearn and does not improve the
    # guarantee we need; GroupKFold guarantees zero cluster overlap.
    ordinary = _evaluate_cv("ordinary_stratified_5fold", StratifiedKFold(5, shuffle=True, random_state=42),
                            X, binary_y, labels, known_pred)
    grouped = _evaluate_cv("pseudo_speaker_group_5fold", GroupKFold(5), X, binary_y,
                           labels, known_pred, groups=groups)
    simple_ordinary = _evaluate_simple_cv("ordinary_stratified_5fold", StratifiedKFold(5, shuffle=True, random_state=42),
                                          simple_unknown_score, labels, known_pred)
    simple_grouped = _evaluate_simple_cv("pseudo_speaker_group_5fold", GroupKFold(5), simple_unknown_score,
                                         labels, known_pred, groups=groups)
    unknown_groups = groups[labels == 0]
    report = {
        "scope": "post-hoc decision layer only; no audio encoder or speaker head was trained",
        "n_validation": int(len(labels)),
        "n_known": int((labels > 0).sum()),
        "n_unknown": int((labels == 0).sum()),
        "n_unknown_pseudo_speaker_groups": int(len(set(unknown_groups))),
        "largest_unknown_group_in_validation": int(max((unknown_groups == g).sum() for g in set(unknown_groups))),
        "features": ["fused_unknown_mass", "top_known_probability", "known_margin",
                     "max_centroid_cosine", "centroid_margin", "head_entropy"],
        "baseline_existing_decision": baseline,
        "one_dimensional_gate_cv": {"ordinary": simple_ordinary, "group_aware": simple_grouped},
        "six_feature_logistic_calibration_cv": {"ordinary": ordinary, "group_aware": grouped},
        "interpretation": "Ordinary CV is intentionally optimistic because it may contain the same pseudo-speaker on both sides. Group-aware CV is the deployment guardrail.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nSaved: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
