"""Build the preregistered train-only known-hard sampling artifact.

The output is consumed by ``data.known_sampling.weights_path``.  Validation
audio never defines thresholds, weights or feature normalisation.  The script
hard-fails on a split mismatch, duplicate file, unreadable audio or leakage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio_windows import speech_activity_mask  # noqa: E402
from src.data_pipeline import (  # noqa: E402
    _sampling_rows_sha256,
    load_unknown_cluster_map,
    prepare_labels,
)
from src.experiment_config import load_profile  # noqa: E402


FEATURE_KEYS = ("duration_seconds", "pcm_rms", "active_fraction")
QUANTILE = 0.25
HARD_WEIGHT = 2.0


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def classify_known_features(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Apply the locked lower-quartile, two-of-three hard-file rule."""
    if not rows:
        raise ValueError("Known feature table is empty")
    files = [str(row.get("audio_file", "")) for row in rows]
    if not all(files) or len(files) != len(set(files)):
        raise ValueError("Known feature table audio_file values must be unique")

    thresholds: dict[str, float] = {}
    for key in FEATURE_KEYS:
        values = np.asarray([row.get(key) for row in rows], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"Feature {key} contains non-finite values")
        thresholds[key] = float(np.quantile(values, QUANTILE))

    classified: list[dict[str, Any]] = []
    for row in rows:
        flags = {key: bool(float(row[key]) <= thresholds[key]) for key in FEATURE_KEYS}
        hard_votes = sum(flags.values())
        classified.append({
            **row,
            "lower_quartile": flags,
            "hard_votes": int(hard_votes),
            "is_hard": bool(hard_votes >= 2),
            "sampling_weight": HARD_WEIGHT if hard_votes >= 2 else 1.0,
        })
    return thresholds, classified


def build_artifact_payload(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    rows: list[dict[str, Any]],
    *,
    profile: str,
    split_metadata: dict[str, Any],
    competition_known_count: int,
) -> dict[str, Any]:
    """Validate split isolation and assemble a deterministic provenance record."""
    labels = train_df["label"].to_numpy(dtype=np.int64)
    known_mask = (labels > 0) & (labels <= int(competition_known_count))
    known_train_files = set(
        train_df.loc[known_mask, "audio_file"].astype(str).tolist()
    )
    feature_files = {str(row["audio_file"]) for row in rows}
    if feature_files != known_train_files:
        missing = sorted(known_train_files - feature_files)[:5]
        extra = sorted(feature_files - known_train_files)[:5]
        raise ValueError(
            "Feature table must exactly equal the known training pool; "
            f"missing={missing}, extra={extra}"
        )
    leaked = sorted(feature_files & set(val_df["audio_file"].astype(str)))
    if leaked:
        raise ValueError(f"Validation leakage in feature table: {leaked[:5]}")

    thresholds, classified = classify_known_features(rows)
    classified = sorted(classified, key=lambda row: str(row["audio_file"]))
    weights = {
        str(row["audio_file"]): float(row["sampling_weight"])
        for row in classified
    }
    hard_by_speaker = Counter(
        str(row["speaker_id"]) for row in classified if row["is_hard"]
    )
    hard_count = sum(bool(row["is_hard"]) for row in classified)

    return {
        "schema_version": 1,
        "profile": profile,
        "split": split_metadata,
        "training_rows_sha256": _sampling_rows_sha256(train_df),
        "known_feature_table_sha256": _canonical_sha256(classified),
        "feature_contract": {
            "features": list(FEATURE_KEYS),
            "quantile": QUANTILE,
            "hard_if_lower_quartile_votes_at_least": 2,
            "hard_weight": HARD_WEIGHT,
            "normal_weight": 1.0,
            "threshold_source": "known training pool only",
        },
        "thresholds": thresholds,
        "known_training_file_count": len(classified),
        "hard_file_count": int(hard_count),
        "hard_file_fraction": float(hard_count / len(classified)),
        "hard_file_count_by_speaker": dict(sorted(hard_by_speaker.items())),
        "validation_overlap_count": 0,
        "weights": weights,
        "features": classified,
    }


def _extract_one(
    row: dict[str, Any],
    audio_dir: Path,
    relative_db: float,
) -> dict[str, Any]:
    path = audio_dir / str(row["audio_file"])
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if audio.size == 0 or int(sample_rate) <= 0:
        raise ValueError(f"Empty or invalid audio: {path}")
    mono = np.asarray(audio.mean(axis=1), dtype=np.float32)
    if not np.isfinite(mono).all():
        raise ValueError(f"Non-finite audio samples: {path}")
    waveform = torch.from_numpy(mono).unsqueeze(0)
    active, _ = speech_activity_mask(
        waveform, sample_rate=int(sample_rate), relative_db=float(relative_db),
    )
    rms = math.sqrt(float(np.mean(np.square(mono, dtype=np.float64))))
    return {
        "audio_file": str(row["audio_file"]),
        "speaker_id": str(row["speaker_id"]),
        "label": int(row["label"]),
        "sample_rate": int(sample_rate),
        "duration_seconds": float(len(mono) / sample_rate),
        "pcm_rms": float(rms),
        "active_fraction": float(active.float().mean().item()),
    }


def extract_known_features(
    train_df: pd.DataFrame,
    audio_dir: Path,
    *,
    competition_known_count: int,
    relative_db: float,
    workers: int,
) -> list[dict[str, Any]]:
    labels = train_df["label"].to_numpy(dtype=np.int64)
    known = train_df.loc[
        (labels > 0) & (labels <= int(competition_known_count)),
        ["audio_file", "speaker_id", "label"],
    ]
    records = known.to_dict("records")
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = {
            pool.submit(_extract_one, row, audio_dir, relative_db): row
            for row in records
        }
        for future in as_completed(futures):
            results.append(future.result())
    return sorted(results, key=lambda row: str(row["audio_file"]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_profile(args.profile)
    data_cfg = config["data"]
    audio_cfg = config["audio"]
    split_cfg = data_cfg.get("split", {}) or {}
    competition_known_count = int(
        config.get("model", {}).get("competition_num_known", 446)
    )
    audio_dir = _project_path(data_cfg["audio_dir"])

    train_df, val_df, _ = prepare_labels(
        labels_path=str(_project_path(data_cfg["labels_path"])),
        output_path=str(_project_path(data_cfg["processed_labels"])),
        val_per_known=1,
        unknown_val_ratio=0.2,
        audio_dir=str(audio_dir),
        min_valid_duration=float(audio_cfg.get("min_valid_duration", 0.0)),
        split_scheme=str(split_cfg.get("scheme", "single")),
        fold=int(split_cfg.get("fold", 0)),
        folds=int(split_cfg.get("folds", 3)),
        unknown_cluster_map=load_unknown_cluster_map(config),
        clean_duplicates=bool(data_cfg.get("clean_duplicates", False)),
    )
    rows = extract_known_features(
        train_df,
        audio_dir,
        competition_known_count=competition_known_count,
        relative_db=float(audio_cfg.get("speech_relative_db", 35.0)),
        workers=args.workers,
    )
    payload = build_artifact_payload(
        train_df,
        val_df,
        rows,
        profile=args.profile,
        split_metadata={
            "scheme": str(split_cfg.get("scheme", "single")),
            "fold": int(split_cfg.get("fold", 0)),
            "folds": int(split_cfg.get("folds", 3)),
            "seed": int(split_cfg.get("seed", 42)),
        },
        competition_known_count=competition_known_count,
    )

    output = _project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps({
        "output": str(output),
        "training_rows_sha256": payload["training_rows_sha256"],
        "known_feature_table_sha256": payload["known_feature_table_sha256"],
        "known_training_file_count": payload["known_training_file_count"],
        "hard_file_count": payload["hard_file_count"],
        "thresholds": payload["thresholds"],
        "validation_overlap_count": payload["validation_overlap_count"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
