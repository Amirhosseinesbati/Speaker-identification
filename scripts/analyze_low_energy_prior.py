"""Audit pre-registered low-energy OOF fallbacks without tuning a threshold.

The audio thresholds are fixed by the Control Fold 1 preregistration:

* exact silence: PCM peak == 0
* near silence: PCM RMS < 1e-4

The fallback label is the most frequent competition class in the training
partition of the same fold.  Validation labels are used only for evaluation;
they never influence the prior or either threshold.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_oof_disagreements import _audio_identity
from scripts.compare_oof_predictions import (
    _metrics_from_predictions,
    _scalar,
    _sha256,
    load_oof,
)


NUM_CLASSES = 447
NEAR_SILENCE_RMS = 1e-4


def _competition_label(row: dict[str, str]) -> int:
    if int(row["is_ood"]):
        return 0
    label = int(row["metric_label"])
    if not 1 <= label < NUM_CLASSES:
        raise ValueError(f"Known row has invalid metric_label={label}: {row}")
    return label


def _load_training_prior(
    labels_path: Path,
    validation_files: set[str],
    corrupted_files: set[str],
) -> tuple[np.ndarray, dict[str, int]]:
    with labels_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    filenames = [row["audio_file"] for row in rows]
    if len(filenames) != len(set(filenames)):
        raise ValueError(f"{labels_path} contains duplicate audio_file rows")
    missing = sorted(validation_files - set(filenames))
    if missing:
        raise ValueError(
            f"{len(missing)} OOF files are absent from {labels_path}; first={missing[0]}"
        )
    leaked_corrupt = sorted(validation_files & corrupted_files)
    if leaked_corrupt:
        raise ValueError(
            f"OOF validation contains corrupted files; first={leaked_corrupt[0]}"
        )

    clean_rows = [row for row in rows if row["audio_file"] not in corrupted_files]
    train_rows = [
        row for row in clean_rows if row["audio_file"] not in validation_files
    ]
    if len(train_rows) + len(validation_files) != len(clean_rows):
        raise ValueError("Clean labels do not partition exactly into train and OOF rows")
    counts = Counter(_competition_label(row) for row in train_rows)
    prior = np.asarray([counts.get(label, 0) for label in range(NUM_CLASSES)], dtype=float)
    if prior.sum() <= 0:
        raise ValueError("Training-only prior is empty")
    prior /= prior.sum()
    return prior, {
        "all_rows": len(rows),
        "clean_rows": len(clean_rows),
        "corrupted_rows_excluded": len(rows) - len(clean_rows),
        "train_rows": len(train_rows),
        "validation_rows": len(validation_files),
        "train_unknown_rows": counts.get(0, 0),
        "train_known_rows": len(train_rows) - counts.get(0, 0),
    }


def _subgroup(
    labels: np.ndarray,
    predictions: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    y_true = np.where(labels > NUM_CLASSES - 1, 0, labels)
    if not mask.any():
        return {
            "samples": 0,
            "unknown_samples": 0,
            "known_samples": 0,
            "correct": 0,
            "accuracy": None,
            "predicted_unknown": 0,
        }
    selected_true = y_true[mask]
    selected_pred = predictions[mask]
    return {
        "samples": int(mask.sum()),
        "unknown_samples": int((selected_true == 0).sum()),
        "known_samples": int((selected_true != 0).sum()),
        "correct": int((selected_true == selected_pred).sum()),
        "accuracy": float((selected_true == selected_pred).mean()),
        "predicted_unknown": int((selected_pred == 0).sum()),
    }


def _metric_delta(
    baseline: dict[str, float],
    candidate: dict[str, float],
) -> dict[str, float]:
    return {key: float(candidate[key] - baseline[key]) for key in baseline}


def audit(
    oof_path: Path,
    audio_dir: Path,
    labels_path: Path,
    split_report_path: Path,
) -> dict[str, Any]:
    record = load_oof(oof_path)
    files = record["files"].astype(str)
    labels = record["labels"].astype(np.int64)
    probabilities = record["competition_probs"]
    predictions = probabilities.argmax(axis=1)

    split_report = json.loads(split_report_path.read_text(encoding="utf-8"))
    corrupted_files = set(split_report["corrupted_files"]["files"])
    prior, partition = _load_training_prior(
        labels_path,
        set(files.tolist()),
        corrupted_files,
    )
    prior_label = int(prior.argmax())

    identities = [_audio_identity(audio_dir / filename) for filename in files]
    audio_errors = [
        {"audio_file": filename, "error": identity["audio_error"]}
        for filename, identity in zip(files, identities)
        if identity["audio_error"] is not None
    ]
    if audio_errors:
        raise RuntimeError(
            f"Failed to inspect {len(audio_errors)} OOF WAV files; first={audio_errors[0]}"
        )

    peaks = np.asarray([identity["pcm_peak"] for identity in identities], dtype=float)
    rms = np.asarray([identity["pcm_rms"] for identity in identities], dtype=float)
    exact_mask = peaks == 0.0
    low_energy_mask = rms < NEAR_SILENCE_RMS
    if not np.all(~exact_mask | low_energy_mask):
        raise AssertionError("Exact-silence samples must also satisfy the RMS gate")

    baseline_metrics = _metrics_from_predictions(predictions, labels, NUM_CLASSES)

    def fallback(mask: np.ndarray) -> dict[str, Any]:
        candidate_predictions = predictions.copy()
        candidate_predictions[mask] = prior_label
        metrics = _metrics_from_predictions(candidate_predictions, labels, NUM_CLASSES)
        changed = mask & (candidate_predictions != predictions)
        return {
            "replacement_label": prior_label,
            "changed_predictions": int(changed.sum()),
            "baseline_subgroup": _subgroup(labels, predictions, mask),
            "fallback_subgroup": _subgroup(labels, candidate_predictions, mask),
            "metrics": metrics,
            "delta": _metric_delta(baseline_metrics, metrics),
        }

    return {
        "schema_version": 1,
        "oof_path": str(oof_path),
        "oof_sha256": _sha256(oof_path),
        "split_report_path": str(split_report_path),
        "split_report_sha256": _sha256(split_report_path),
        "split": {
            "scheme": str(_scalar(record, "split_scheme")),
            "fold": int(_scalar(record, "split_fold")),
            "folds": int(_scalar(record, "split_folds")),
            "seed": int(_scalar(record, "split_seed")),
        },
        "thresholds": {
            "exact_silence_peak": 0.0,
            "near_silence_rms_lt": NEAR_SILENCE_RMS,
        },
        "partition": partition,
        "train_only_prior": {
            "argmax_label": prior_label,
            "argmax_probability": float(prior[prior_label]),
            "unknown_probability": float(prior[0]),
            "top5": [
                {"label": int(index), "probability": float(prior[index])}
                for index in np.argsort(prior)[-5:][::-1]
            ],
        },
        "baseline_metrics": baseline_metrics,
        "audio": {
            "files": len(files),
            "exact_silence_files": int(exact_mask.sum()),
            "near_silence_files": int(low_energy_mask.sum()),
            "rms_min": float(rms.min()),
            "rms_median": float(np.median(rms)),
            "rms_max": float(rms.max()),
        },
        "exact_silence_fallback": fallback(exact_mask),
        "near_silence_fallback": fallback(low_energy_mask),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--split-report", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = audit(args.oof, args.audio_dir, args.labels, args.split_report)
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
