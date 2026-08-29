"""Prove that submission LME-20 scoring reproduces the locked OOF policy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_control_oof_centroid_crossfit import (  # noqa: E402
    NUM_FOLDS,
    evaluate_policy,
    l2norm_rows,
    predict,
)
from scripts.analyze_prototype_aggregation_crossfit import fold_variants  # noqa: E402
from scripts.analyze_unknown_cluster_hypotheses_crossfit import (  # noqa: E402
    load_fold_inputs,
)
from submission.inference import prototype_logmeanexp_probs  # noqa: E402


PARAMS = {
    "alpha": 0.15,
    "kappa": 16.0,
    "tau": 0.5,
    "lambda_unknown": 0.75,
}
EXPECTED_AGGREGATE_MACRO_F1 = 0.9633564052


def runtime_prediction(oof: dict, artifact: dict) -> np.ndarray:
    train_embeddings = l2norm_rows(artifact["train_embeddings"])
    competition_labels = artifact["competition_labels"].astype(np.int64)
    unknown_mask = competition_labels == 0
    speaker_ids = competition_labels.copy()
    speaker_ids[unknown_mask] = (
        447 + artifact["unknown_cluster_ids"][unknown_mask].astype(np.int64)
    )
    prototype_probs, max_score = prototype_logmeanexp_probs(
        l2norm_rows(oof["embeddings"]),
        train_embeddings,
        speaker_ids,
        447,
        beta=20.0,
        kappa=PARAMS["kappa"],
    )
    fused = (
        PARAMS["alpha"] * oof["competition_probs"].astype(np.float64)
        + (1.0 - PARAMS["alpha"]) * prototype_probs
    )
    fused[:, 0] *= PARAMS["lambda_unknown"]
    fused /= fused.sum(axis=1, keepdims=True) + 1e-12
    predictions = fused.argmax(axis=1).astype(np.int64)
    predictions[max_score < PARAMS["tau"]] = 0
    return predictions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "checkpoints")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "data" / "experiments" / "campp_control_centroid_crossfit",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "generated"
        / "campp_lme20_submission_equivalence.json",
    )
    args = parser.parse_args()

    oofs, artifacts, metadata = load_fold_inputs(args.checkpoint_root, args.cache_dir)
    evidence = []
    predictions = []
    rows = []
    for fold in range(NUM_FOLDS):
        variants, _ = fold_variants(fold=fold, artifact=artifacts[fold], oof=oofs[fold])
        offline_evidence = variants["logmeanexp_b20"]
        offline_prediction = predict(offline_evidence, PARAMS)
        submission_prediction = runtime_prediction(oofs[fold], artifacts[fold])
        mismatch = int(np.sum(offline_prediction != submission_prediction))
        if mismatch:
            raise RuntimeError(
                f"Fold {fold} submission/offline prediction mismatch: {mismatch} rows"
            )
        evidence.append(offline_evidence)
        predictions.append(submission_prediction)
        rows.append({
            "fold": fold,
            "rows": int(len(submission_prediction)),
            "prediction_mismatches": mismatch,
            "prototype_cache_sha256": metadata[fold]["artifact_sha256"],
        })

    evaluation = evaluate_policy(evidence, predictions)
    macro = float(evaluation["aggregate"]["candidate"]["macro_f1"])
    if abs(macro - EXPECTED_AGGREGATE_MACRO_F1) > 5e-10:
        raise RuntimeError(
            f"Locked OOF Macro-F1 changed: {macro} != {EXPECTED_AGGREGATE_MACRO_F1}"
        )
    report = {
        "status": "equivalent",
        "parameters": {**PARAMS, "beta": 20.0},
        "folds": rows,
        "evaluation": evaluation,
        "expected_aggregate_macro_f1": EXPECTED_AGGREGATE_MACRO_F1,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "output": str(args.output),
        "aggregate": evaluation["aggregate"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
