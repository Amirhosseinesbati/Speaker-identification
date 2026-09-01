"""Describe aligned OOF errors by real WAV duration without selecting a router.

This audit is deliberately non-selective.  It uses fixed bins tied to the
2-second student target and the 8-second control window, reports both the
competition 447-class Macro-F1 and an observed-class diagnostic, and never
evaluates a duration-routing policy or threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import wave
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_control_oof_centroid_crossfit import (  # noqa: E402
    metric_bundle,
    metric_delta,
)
from scripts.evaluate_fixed_oof_pair import (  # noqa: E402
    class_coverage,
    collapse_labels,
    load_oof,
    observed_class_macro_f1,
    sha256_file,
    transition_bundle,
)


DURATION_BINS = (
    ("le_2s", 0.0, 2.0),
    ("gt_2_le_4s", 2.0, 4.0),
    ("gt_4_le_6s", 4.0, 6.0),
    ("gt_6_le_8s", 6.0, 8.0),
    ("gt_8_le_12s", 8.0, 12.0),
    ("gt_12s", 12.0, float("inf")),
)


def digest_names(names: np.ndarray) -> str:
    payload = "\n".join(sorted(map(str, names))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        if rate <= 0:
            raise RuntimeError(f"Invalid WAV sample rate in {path}: {rate}")
        return float(handle.getnframes() / rate)


def duration_bin_name(duration: float) -> str:
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError(f"Duration must be finite and positive: {duration}")
    for name, lower, upper in DURATION_BINS:
        if duration > lower and duration <= upper:
            return name
    raise AssertionError(f"No duration bin for {duration}")


def align_pair(
    baseline: dict[str, np.ndarray], candidate: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    baseline_files = baseline["files"].astype(str)
    candidate_files = candidate["files"].astype(str)
    if set(baseline_files.tolist()) != set(candidate_files.tolist()):
        raise RuntimeError("Baseline and candidate OOF file sets differ")
    candidate_index = {
        name: position for position, name in enumerate(candidate_files)
    }
    order = np.asarray(
        [candidate_index[name] for name in baseline_files], dtype=np.int64
    )
    labels = collapse_labels(baseline["labels"])
    candidate_labels = collapse_labels(candidate["labels"])[order]
    if not np.array_equal(labels, candidate_labels):
        raise RuntimeError("Baseline and candidate labels differ after alignment")
    split = {}
    for key in ("split_fold", "split_folds", "split_seed"):
        left = int(np.asarray(baseline[key]).reshape(-1)[0])
        right = int(np.asarray(candidate[key]).reshape(-1)[0])
        if left != right:
            raise RuntimeError(
                f"Baseline/candidate {key} mismatch: {left} != {right}"
            )
        split[key] = left
    baseline_probs = baseline["competition_probs"].astype(np.float64)
    candidate_probs = candidate["competition_probs"].astype(np.float64)[order]
    return baseline_files, labels, baseline_probs, candidate_probs, split


def describe_subset(
    labels: np.ndarray,
    baseline_predictions: np.ndarray,
    candidate_predictions: np.ndarray,
) -> dict:
    baseline_metrics = metric_bundle(labels, baseline_predictions)
    candidate_metrics = metric_bundle(labels, candidate_predictions)
    return {
        "coverage": class_coverage(labels),
        "metrics": {
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
            "candidate_delta": metric_delta(
                candidate_metrics, baseline_metrics
            ),
        },
        "observed_class_macro_f1_descriptive_only": {
            "baseline": observed_class_macro_f1(
                labels, baseline_predictions
            ),
            "candidate": observed_class_macro_f1(
                labels, candidate_predictions
            ),
            "warning": (
                "This excludes absent competition classes and is not the "
                "447-class competition Macro-F1."
            ),
        },
        "candidate_vs_baseline_transitions": transition_bundle(
            labels, baseline_predictions, candidate_predictions
        ),
        "prediction_disagreements": int(
            np.sum(baseline_predictions != candidate_predictions)
        ),
    }


def analyze(
    baseline: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    audio_dir: Path,
) -> dict:
    files, labels, baseline_probs, candidate_probs, split = align_pair(
        baseline, candidate
    )
    durations = np.asarray(
        [wav_duration_seconds(audio_dir / name) for name in files],
        dtype=np.float64,
    )
    baseline_predictions = baseline_probs.argmax(axis=1).astype(np.int64)
    candidate_predictions = candidate_probs.argmax(axis=1).astype(np.int64)
    bins = {}
    for name, lower, upper in DURATION_BINS:
        mask = (durations > lower) & (durations <= upper)
        indices = np.flatnonzero(mask)
        bins[name] = {
            "bounds_seconds": {
                "lower_exclusive": lower,
                "upper_inclusive": None if np.isinf(upper) else upper,
            },
            "duration_min": (
                float(durations[indices].min()) if len(indices) else None
            ),
            "duration_max": (
                float(durations[indices].max()) if len(indices) else None
            ),
            "evaluation": (
                describe_subset(
                    labels[indices],
                    baseline_predictions[indices],
                    candidate_predictions[indices],
                )
                if len(indices)
                else None
            ),
        }
    return {
        "contract": {
            "analysis_type": "descriptive_actual_duration_oof",
            "routing_policy_evaluated": False,
            "threshold_or_bin_selected": False,
            "duration_bins_fixed_before_reading_results": True,
            "leaderboard_used": False,
            "warning": (
                "No bin result may select a routing cutoff on this Fold. Any "
                "P12 cutoff must be preregistered or selected on other folds."
            ),
        },
        "integrity": {
            "rows": int(len(files)),
            "unique_files": int(len(set(files.tolist()))),
            "file_set_sha256": digest_names(files),
            "split": split,
        },
        "duration_seconds": {
            "min": float(durations.min()),
            "max": float(durations.max()),
            "mean": float(durations.mean()),
            "quantiles": {
                str(q): float(np.quantile(durations, q))
                for q in (0.1, 0.25, 0.5, 0.75, 0.9)
            },
        },
        "overall": describe_subset(
            labels, baseline_predictions, candidate_predictions
        ),
        "bins": bins,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        load_oof(args.baseline), load_oof(args.candidate), args.audio_dir
    )
    report = {
        "provenance": {
            "baseline": str(args.baseline.resolve()),
            "baseline_sha256": sha256_file(args.baseline),
            "candidate": str(args.candidate.resolve()),
            "candidate_sha256": sha256_file(args.candidate),
            "audio_dir": str(args.audio_dir.resolve()),
        },
        "analysis": result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
