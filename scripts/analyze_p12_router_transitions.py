"""Forensically describe P12 prediction transitions without selecting a router.

This analysis is deliberately descriptive.  It verifies that the routed OOF is
the exact preregistered hard switch between P0 and P11, then records every
rescued and introduced error so a later hypothesis can address failure modes
without post-hoc tuning of the P12 duration cut-off.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as record:
        required = {"files", "labels", "competition_probs"}
        missing = required - set(record.files)
        if missing:
            raise RuntimeError(f"{path} lacks keys: {sorted(missing)}")
        return {key: record[key] for key in record.files}


def _probability_summary(probabilities: np.ndarray, row: int, truth: int) -> dict:
    order = np.argsort(probabilities[row])[::-1]
    top1 = int(order[0])
    top2 = int(order[1])
    return {
        "prediction": top1,
        "top1_probability": float(probabilities[row, top1]),
        "top2_label": top2,
        "top2_probability": float(probabilities[row, top2]),
        "top1_top2_margin": float(
            probabilities[row, top1] - probabilities[row, top2]
        ),
        "true_label_probability": float(probabilities[row, truth]),
    }


def analyze_transitions(
    baseline: dict[str, np.ndarray],
    specialist: dict[str, np.ndarray],
    router: dict[str, np.ndarray],
) -> dict:
    files = baseline["files"].astype(str)
    labels = baseline["labels"].astype(np.int64)
    baseline_probs = baseline["competition_probs"].astype(np.float64)
    specialist_probs = specialist["competition_probs"].astype(np.float64)
    router_probs = router["competition_probs"].astype(np.float64)

    for name, record in (("specialist", specialist), ("router", router)):
        if not np.array_equal(files, record["files"].astype(str)):
            raise RuntimeError(f"{name} file order differs from baseline")
        if not np.array_equal(labels, record["labels"].astype(np.int64)):
            raise RuntimeError(f"{name} labels differ from baseline")
    if baseline_probs.shape != specialist_probs.shape or baseline_probs.shape != router_probs.shape:
        raise RuntimeError("OOF probability shapes differ")

    durations = np.asarray(router["duration_seconds"], dtype=np.float64)
    short_mask = np.asarray(router["short_specialist_mask"], dtype=bool)
    if durations.shape != labels.shape or short_mask.shape != labels.shape:
        raise RuntimeError("P12 duration or route-mask shape differs from labels")
    expected = baseline_probs.copy()
    expected[short_mask] = specialist_probs[short_mask]
    if not np.allclose(router_probs, expected, rtol=0.0, atol=1e-7):
        raise RuntimeError("P12 probabilities are not the exact hard-routed vectors")

    baseline_pred = baseline_probs.argmax(axis=1).astype(np.int64)
    specialist_pred = specialist_probs.argmax(axis=1).astype(np.int64)
    router_pred = router_probs.argmax(axis=1).astype(np.int64)
    baseline_correct = baseline_pred == labels
    router_correct = router_pred == labels
    rescued_mask = (~baseline_correct) & router_correct
    introduced_mask = baseline_correct & (~router_correct)
    changed_mask = baseline_pred != router_pred

    def records(mask: np.ndarray) -> list[dict]:
        output: list[dict] = []
        for row in np.flatnonzero(mask):
            truth = int(labels[row])
            output.append(
                {
                    "row": int(row),
                    "file": str(files[row]),
                    "duration_seconds": float(durations[row]),
                    "true_label": truth,
                    "true_type": "unknown" if truth == 0 else "known",
                    "baseline": _probability_summary(baseline_probs, row, truth),
                    "specialist": _probability_summary(specialist_probs, row, truth),
                    "router": _probability_summary(router_probs, row, truth),
                }
            )
        return output

    introduced_known = introduced_mask & (labels != 0)
    introduced_unknown = introduced_mask & (labels == 0)
    rescued_known = rescued_mask & (labels != 0)
    rescued_unknown = rescued_mask & (labels == 0)
    specialist_is_ood_only = bool(
        rescued_mask.sum() > 0
        and rescued_unknown.sum() == rescued_mask.sum()
        and rescued_known.sum() == 0
    )
    return {
        "contract": {
            "analysis_type": "descriptive_p12_transition_forensics",
            "router_or_threshold_selected": False,
            "p12_cutoff_changed": False,
            "leaderboard_used": False,
            "allowed_use": (
                "Understand P12 failure modes and select only a genuinely "
                "different future hypothesis; do not tune P12 on Fold0."
            ),
        },
        "integrity": {
            "rows": int(len(files)),
            "unique_files": int(len(set(files.tolist()))),
            "probability_columns": int(router_probs.shape[1]),
            "short_rows": int(short_mask.sum()),
            "exact_hard_route_verified": True,
        },
        "summary": {
            "prediction_changes": int(changed_mask.sum()),
            "rescued_errors": int(rescued_mask.sum()),
            "introduced_errors": int(introduced_mask.sum()),
            "rescued_known": int(rescued_known.sum()),
            "rescued_unknown": int(rescued_unknown.sum()),
            "introduced_known": int(introduced_known.sum()),
            "introduced_unknown": int(introduced_unknown.sum()),
            "net_correct": int(rescued_mask.sum() - introduced_mask.sum()),
        },
        "scientific_interpretation": {
            "all_rescues_are_unknown": specialist_is_ood_only,
            "known_identity_rescues": int(rescued_known.sum()),
            "conclusion": (
                "P11 behaves as a short-file OOD specialist, not as a better "
                "known-speaker classifier. P12 remains closed; these Fold0 "
                "transitions must not tune its cutoff or confidence rules."
                if specialist_is_ood_only
                else "Mixed rescue modes; P12 still remains closed."
            ),
        },
        "introduced_errors": records(introduced_mask),
        "rescued_errors": records(rescued_mask),
        "all_prediction_changes": records(changed_mask),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--specialist", type=Path, required=True)
    parser.add_argument("--router", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "provenance": {
            "baseline_oof": str(args.baseline.resolve()),
            "baseline_oof_sha256": sha256_file(args.baseline),
            "specialist_oof": str(args.specialist.resolve()),
            "specialist_oof_sha256": sha256_file(args.specialist),
            "router_oof": str(args.router.resolve()),
            "router_oof_sha256": sha256_file(args.router),
        },
        "analysis": analyze_transitions(
            _load(args.baseline), _load(args.specialist), _load(args.router)
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
