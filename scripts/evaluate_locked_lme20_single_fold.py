"""Evaluate the immutable LME20 backend on one leak-free OOF fold.

This script is deliberately *not* a tuning utility.  It rebuilds the exact
three-fold split, extracts train-fold embeddings with checkpoint/SHA-aware
caching, and applies the backend that preceded the current best leaderboard
submission without searching any aggregation or decision parameter:

* set score: log-mean-exp over enrollment exemplars, beta=20;
* head/set fusion alpha=0.15;
* set softmax kappa=16;
* cosine rejection tau=0 (disabled);
* collapsed unknown probability multiplier=0.75.

The output keeps Raw and LME20 Macro-F1, Known Accuracy, OOD-F1, rescued and
introduced errors, split identity, and artifact hashes together.  A strong
single-fold result is diagnostic only; it never authorizes a submission or a
new fold by itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_control_oof_centroid_crossfit import (  # noqa: E402
    NUM_FOLDS,
    build_or_load_train_artifact,
    load_oof,
    metric_bundle,
    metric_delta,
    predict,
    rebuild_exact_splits,
    sha256_file,
)
from scripts.analyze_prototype_aggregation_crossfit import (  # noqa: E402
    fold_variants,
)


LOCKED_VARIANT = "logmeanexp_b20"
LOCKED_PARAMETERS = {
    "alpha": 0.15,
    "kappa": 16.0,
    "tau": 0.0,
    "lambda_unknown": 0.75,
}


def evaluate_locked_lme20(
    *, fold: int, artifact: dict[str, np.ndarray], oof: dict
) -> dict:
    """Apply only the immutable LME20 policy and return its full diagnostics."""

    variants, diagnostics = fold_variants(
        fold=fold, artifact=artifact, oof=oof
    )
    if LOCKED_VARIANT not in variants:
        raise RuntimeError(f"Missing required aggregation {LOCKED_VARIANT!r}")
    evidence = variants[LOCKED_VARIANT]
    predictions = predict(evidence, dict(LOCKED_PARAMETERS))
    baseline_predictions = evidence.baseline_predictions
    labels = evidence.labels
    baseline = metric_bundle(labels, baseline_predictions)
    candidate = metric_bundle(labels, predictions)
    baseline_correct = baseline_predictions == labels
    candidate_correct = predictions == labels
    rescued_mask = ~baseline_correct & candidate_correct
    introduced_mask = baseline_correct & ~candidate_correct
    return {
        "fold": int(fold),
        "rows": int(len(labels)),
        "variant": LOCKED_VARIANT,
        "parameters": dict(LOCKED_PARAMETERS),
        "baseline": baseline,
        "candidate": candidate,
        "delta": metric_delta(candidate, baseline),
        "rescued_errors": int(np.sum(rescued_mask)),
        "introduced_errors": int(np.sum(introduced_mask)),
        "baseline_errors": int(np.sum(~baseline_correct)),
        "rescue_rate": float(
            np.sum(rescued_mask) / max(int(np.sum(~baseline_correct)), 1)
        ),
        "rescued_files": evidence.files[rescued_mask].astype(str).tolist(),
        "introduced_files": evidence.files[introduced_mask].astype(str).tolist(),
        "aggregation_diagnostics": diagnostics[LOCKED_VARIANT],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--cluster-map", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument(
        "--labels", type=Path,
        default=ROOT / "data" / "processed" / "audio_wav_labels.csv",
    )
    parser.add_argument(
        "--audio-dir", type=Path,
        default=ROOT / "data" / "processed" / "audio_wav",
    )
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "reports" / "generated"
        / "locked_lme20_single_fold.json",
    )
    args = parser.parse_args()

    if not 0 <= args.fold < NUM_FOLDS:
        raise ValueError(f"fold must be in [0,{NUM_FOLDS - 1}]")
    checkpoint_sha = sha256_file(args.checkpoint)
    if (
        args.expected_checkpoint_sha256
        and checkpoint_sha != args.expected_checkpoint_sha256
    ):
        raise RuntimeError(
            "Checkpoint SHA mismatch: "
            f"expected={args.expected_checkpoint_sha256}, actual={checkpoint_sha}"
        )

    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    splits, cleaning = rebuild_exact_splits(args.labels, args.audio_dir)
    train_frame, validation_frame = splits[args.fold]
    expected_files = set(validation_frame["audio_file"].astype(str))
    oof = load_oof(args.oof, args.fold, expected_files)
    artifact, cache_metadata = build_or_load_train_artifact(
        fold=args.fold,
        train_frame=train_frame,
        checkpoint_path=args.checkpoint,
        cluster_map_path=args.cluster_map,
        cache_path=args.cache,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    evaluation = evaluate_locked_lme20(
        fold=args.fold, artifact=artifact, oof=oof
    )
    report = {
        "contract": {
            "selection": "none; immutable policy copied from the best stable three-fold LME20 audit",
            "target_fold_used_for_tuning": False,
            "submission_authorized": False,
            "variant": LOCKED_VARIANT,
            "parameters": dict(LOCKED_PARAMETERS),
        },
        "provenance": {
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "oof": str(args.oof),
            "oof_sha256": sha256_file(args.oof),
            "cluster_map": str(args.cluster_map),
            "cluster_map_sha256": sha256_file(args.cluster_map),
            "cache": str(args.cache),
            "cache_sha256": sha256_file(args.cache),
            "cache_metadata": cache_metadata,
            "split_cleaning": cleaning,
            "expected_validation_files": int(len(expected_files)),
        },
        "evaluation": evaluation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report["evaluation"], indent=2, ensure_ascii=False))
    print(f"Wrote immutable LME20 audit: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
