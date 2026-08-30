"""Enrollment-only speaker-specific rejection on the locked CAM++ LME-20 OOF.

The paper-derived rule (Chaubey et al., arXiv:2306.00952, Eq. 7) assigns each
enrolled speaker the maximum cosine similarity between any of that speaker's
enrollment utterances and any enrollment utterance from another known speaker.
After the already-locked LME-20 policy predicts a known identity, the query is
kept only when its maximum enrollment-exemplar cosine exceeds that identity's
threshold; otherwise it is mapped to the competition ``unknown`` class.

The rule has no validation-selected parameter.  Each Fold builds thresholds
only from its training enrollment embeddings and is evaluated on that Fold's
held-out OOF rows.  This is intentionally a cheap falsification test: the
paper evaluated watchlists of 5 and 10 speakers, whereas this competition has
446, so maximum inter-speaker similarity may become too strict.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_control_oof_centroid_crossfit import (  # noqa: E402
    HISTORICAL_PARAMS,
    NUM_FOLDS,
    NUM_KNOWN,
    aggregate_predictions,
    evaluate_policy,
    l2norm_rows,
    metric_bundle,
    metric_delta,
    predict,
)
from scripts.analyze_prototype_aggregation_crossfit import (  # noqa: E402
    group_indices,
    score_matrix_to_evidence,
)
from scripts.analyze_unknown_cluster_hypotheses_crossfit import (  # noqa: E402
    load_fold_inputs,
)


LME_BETA = 20.0
EXPECTED_LOCKED_LME20_MACRO_F1 = 0.9633564052


def speaker_specific_thresholds(
    embeddings: np.ndarray,
    known_groups: list[np.ndarray],
) -> np.ndarray:
    """Return Eq. 7 maximum inter-speaker cosine for every known identity."""

    embeddings = l2norm_rows(np.asarray(embeddings, dtype=np.float32))
    if len(known_groups) == 0:
        raise ValueError("known_groups must not be empty")
    owner = np.full(len(embeddings), -1, dtype=np.int64)
    for speaker_index, indices in enumerate(known_groups):
        indices = np.asarray(indices, dtype=np.int64)
        if len(indices) == 0:
            raise ValueError(f"known speaker {speaker_index + 1} has no enrollment")
        if np.any(indices < 0) or np.any(indices >= len(embeddings)):
            raise ValueError("known group contains an out-of-range enrollment index")
        if np.any(owner[indices] != -1):
            raise ValueError("known enrollment groups overlap")
        owner[indices] = speaker_index

    known_indices = np.flatnonzero(owner >= 0)
    known_embeddings = embeddings[known_indices]
    known_owner = owner[known_indices]
    pairwise = known_embeddings @ known_embeddings.T
    pairwise[known_owner[:, None] == known_owner[None, :]] = -np.inf

    thresholds = np.empty(len(known_groups), dtype=np.float64)
    for speaker_index in range(len(known_groups)):
        rows = pairwise[known_owner == speaker_index]
        threshold = float(np.max(rows))
        if not np.isfinite(threshold):
            raise ValueError("at least two known speakers are required")
        thresholds[speaker_index] = threshold
    return thresholds


def apply_speaker_specific_rejection(
    predictions: np.ndarray,
    known_max_exemplar_scores: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reject known predictions whose paper-rule score is not above threshold."""

    predictions = np.asarray(predictions, dtype=np.int64)
    scores = np.asarray(known_max_exemplar_scores, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if scores.shape != (len(predictions), len(thresholds)):
        raise ValueError("known score matrix does not match predictions/thresholds")
    if np.any(predictions < 0) or np.any(predictions > len(thresholds)):
        raise ValueError("prediction outside competition known/unknown range")

    output = predictions.copy()
    rejected = np.zeros(len(predictions), dtype=bool)
    known_rows = np.flatnonzero(predictions > 0)
    if len(known_rows):
        columns = predictions[known_rows] - 1
        query_scores = scores[known_rows, columns]
        # Eq. 4 uses a strict greater-than decision at the threshold boundary.
        rejected_known = query_scores <= thresholds[columns]
        rejected[known_rows] = rejected_known
        output[known_rows[rejected_known]] = 0
    return output, rejected


def lme20_scores(
    validation_embeddings: np.ndarray,
    train_embeddings: np.ndarray,
    groups: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Return locked LME-20 group scores and known max-exemplar gate scores."""

    validation = l2norm_rows(validation_embeddings)
    train = l2norm_rows(train_embeddings)
    similarities = validation @ train.T
    scores = np.empty((len(validation), len(groups)), dtype=np.float32)
    known_max = np.empty((len(validation), NUM_KNOWN), dtype=np.float32)
    for group_id, indices in enumerate(groups):
        values = similarities[:, indices]
        scores[:, group_id] = (
            logsumexp(LME_BETA * values, axis=1) - np.log(values.shape[1])
        ) / LME_BETA
        if group_id < NUM_KNOWN:
            known_max[:, group_id] = values.max(axis=1)
    return scores, known_max


def evaluate_against_reference(
    folds: list,
    reference_predictions: list[np.ndarray],
    candidate_predictions: list[np.ndarray],
) -> dict:
    """Evaluate a candidate against LME-20, never against the Raw head."""

    if len(reference_predictions) != len(folds) or len(candidate_predictions) != len(
        folds
    ):
        raise ValueError("prediction lists must contain exactly one array per Fold")
    fold_rows = []
    for fold, (reference, candidate) in enumerate(
        zip(reference_predictions, candidate_predictions, strict=True)
    ):
        labels = folds[fold].labels
        baseline_metrics = metric_bundle(labels, reference)
        candidate_metrics = metric_bundle(labels, candidate)
        fold_rows.append(
            {
                "fold": fold,
                "baseline": baseline_metrics,
                "candidate": candidate_metrics,
                "delta": metric_delta(candidate_metrics, baseline_metrics),
            }
        )

    _, labels, reference = aggregate_predictions(folds, reference_predictions)
    _, _, candidate = aggregate_predictions(folds, candidate_predictions)
    baseline_metrics = metric_bundle(labels, reference)
    candidate_metrics = metric_bundle(labels, candidate)
    baseline_correct = reference == labels
    candidate_correct = candidate == labels
    return {
        "folds": fold_rows,
        "aggregate": {
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
            "delta": metric_delta(candidate_metrics, baseline_metrics),
            "rescued_errors": int(np.sum(~baseline_correct & candidate_correct)),
            "introduced_errors": int(np.sum(baseline_correct & ~candidate_correct)),
            "baseline_errors": int(np.sum(~baseline_correct)),
            "rescue_rate": float(
                np.sum(~baseline_correct & candidate_correct)
                / max(np.sum(~baseline_correct), 1)
            ),
        },
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
        default=ROOT
        / "reports"
        / "generated"
        / "campp_lme20_speaker_specific_threshold_crossfit.json",
    )
    args = parser.parse_args()

    oofs, artifacts, metadata = load_fold_inputs(args.checkpoint_root, args.cache_dir)
    folds = []
    baseline_predictions = []
    candidate_predictions = []
    diagnostics = []
    for fold in range(NUM_FOLDS):
        artifact = artifacts[fold]
        oof = oofs[fold]
        groups = group_indices(artifact)
        scores, known_max = lme20_scores(
            oof["embeddings"], artifact["train_embeddings"], groups
        )
        evidence = score_matrix_to_evidence(fold=fold, oof=oof, scores=scores)
        baseline = predict(evidence, HISTORICAL_PARAMS)
        thresholds = speaker_specific_thresholds(
            artifact["train_embeddings"], groups[:NUM_KNOWN]
        )
        candidate, rejected = apply_speaker_specific_rejection(
            baseline, known_max, thresholds
        )
        folds.append(evidence)
        baseline_predictions.append(baseline)
        candidate_predictions.append(candidate)

        truth = evidence.labels
        rejected_known_truth = rejected & (truth > 0)
        rejected_unknown_truth = rejected & (truth == 0)
        diagnostics.append(
            {
                "fold": fold,
                "threshold_min": float(thresholds.min()),
                "threshold_median": float(np.median(thresholds)),
                "threshold_max": float(thresholds.max()),
                "known_predictions_before": int(np.sum(baseline > 0)),
                "rejected_predictions": int(rejected.sum()),
                "rejected_known_truth": int(rejected_known_truth.sum()),
                "rejected_unknown_truth": int(rejected_unknown_truth.sum()),
                "baseline": metric_bundle(truth, baseline),
                "candidate": metric_bundle(truth, candidate),
                "delta": metric_delta(
                    metric_bundle(truth, candidate), metric_bundle(truth, baseline)
                ),
            }
        )

    baseline_evaluation = evaluate_policy(folds, baseline_predictions)
    candidate_evaluation = evaluate_against_reference(
        folds, baseline_predictions, candidate_predictions
    )
    reproduced = baseline_evaluation["aggregate"]["candidate"]["macro_f1"]
    if abs(reproduced - EXPECTED_LOCKED_LME20_MACRO_F1) > 5e-10:
        raise RuntimeError(
            "Locked LME-20 baseline mismatch: "
            f"{reproduced} != {EXPECTED_LOCKED_LME20_MACRO_F1}"
        )

    fold_gains = [row["delta"]["macro_f1"] for row in candidate_evaluation["folds"]]
    aggregate_delta = candidate_evaluation["aggregate"]["delta"]
    accepted = bool(
        min(fold_gains) >= 0.0
        and aggregate_delta["macro_f1"] >= 0.001
        and aggregate_delta["known_accuracy"] >= -0.001
        and aggregate_delta["ood_f1"] >= -0.001
    )
    report = {
        "contract": {
            "source": "Chaubey et al., arXiv:2306.00952, equations 4 and 7",
            "baseline": "locked CAM++ LME-20 with historical decision parameters",
            "threshold_data": "target Fold train enrollment only; no OOF labels",
            "threshold_rule": "maximum cross-known-speaker enrollment cosine",
            "query_gate_score": "maximum query-to-predicted-speaker enrollment cosine",
            "threshold_boundary": "strict greater-than",
            "num_known_speakers": NUM_KNOWN,
            "paper_watchlists": [5, 10],
            "selection": "none; deterministic enrollment-only rule",
            "leaderboard_tuning": False,
            "acceptance_gate": {
                "minimum_fold_macro_f1_gain": 0.0,
                "aggregate_macro_f1_gain": 0.001,
                "aggregate_known_accuracy_delta_min": -0.001,
                "aggregate_ood_f1_delta_min": -0.001,
            },
        },
        "provenance": {"cache_metadata": metadata},
        "fold_diagnostics": diagnostics,
        "baseline_evaluation": baseline_evaluation,
        "candidate_evaluation": candidate_evaluation,
        "decision": {
            "accepted": accepted,
            "fold_macro_f1_gains": fold_gains,
            "aggregate_delta": aggregate_delta,
            "interpretation": (
                "eligible for a separately preregistered package audit"
                if accepted
                else "rejected; do not deploy or tune the rule"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "baseline_macro_f1": reproduced,
                "candidate_macro_f1": candidate_evaluation["aggregate"]["candidate"]
                ["macro_f1"],
                "aggregate_delta": aggregate_delta,
                "fold_macro_f1_gains": fold_gains,
                "accepted": accepted,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
