"""Preregistered fixed-duration diagnostics for D-ALMFT.

This is deliberately not a model-selection or calibration tool.  It evaluates
the same Fold-0 file set, the same deterministic leading crop, and direct Raw
probability argmax for a baseline and treatment checkpoint at each fixed
duration.  No threshold, epoch, crop, or blend is searched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_pipeline import SpeakerDataset
from src.metrics import evaluate_competition_probs
from src.model_factory import create_model_from_config
from src.train import forward_multi_window_evaluation


DEFAULT_REFERENCE_OOF = (
    ROOT
    / "checkpoints/p0-campp-known446-ood-control-oof-f0"
    / "campp_best_bundle/oof_predictions.npz"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_reference(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as arrays:
        required = {"files", "labels", "split_scheme", "split_folds", "split_fold", "split_seed"}
        missing = sorted(required - set(arrays.files))
        if missing:
            raise ValueError(f"Reference OOF missing arrays: {missing}")
        files = arrays["files"].astype(str)
        labels = arrays["labels"].astype(np.int64)
        split = {
            "scheme": str(arrays["split_scheme"].item()),
            "folds": int(arrays["split_folds"].item()),
            "fold": int(arrays["split_fold"].item()),
            "seed": int(arrays["split_seed"].item()),
        }
    if len(files) != len(set(files.tolist())):
        raise ValueError("Reference OOF contains duplicate filenames")
    frame = pd.DataFrame({"audio_file": files, "label": labels})
    return frame, split


def load_checkpoint(path: Path) -> tuple[dict, dict, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config")
    class_map = checkpoint.get("class_map")
    if not isinstance(config, dict) or not isinstance(class_map, dict):
        raise ValueError(f"Checkpoint is not self-describing: {path}")
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint lacks model_state_dict: {path}")
    return checkpoint, config, class_map


def metric_bundle(labels: np.ndarray, probs: np.ndarray, known_count: int) -> dict:
    metrics = evaluate_competition_probs(
        torch.from_numpy(probs), torch.from_numpy(labels)
    )
    predictions = probs.argmax(axis=1)
    known = (labels > 0) & (labels <= known_count)
    unknown = ~known
    return {
        "macro_f1": float(metrics["macro_f1"]),
        "known_accuracy": float(metrics["known_acc"]),
        "ood_f1": float(metrics["ood_f1"]),
        "overall_accuracy": float(metrics["overall_acc"]),
        "known_to_unknown": int(np.sum(known & (predictions == 0))),
        "known_to_wrong_known": int(
            np.sum(known & (predictions > 0) & (predictions != labels))
        ),
        "unknown_to_known": int(np.sum(unknown & (predictions > 0))),
    }


@torch.inference_mode()
def score_checkpoint(
    *,
    checkpoint_path: Path,
    frame: pd.DataFrame,
    duration_seconds: float,
    audio_dir: Path,
    sample_rate: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[np.ndarray, dict]:
    checkpoint, config, class_map = load_checkpoint(checkpoint_path)
    known_count = int((config.get("model", {}) or {}).get("competition_num_known", 446))
    model = create_model_from_config(config, num_known_speakers=len(class_map) - 1)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()

    dataset = SpeakerDataset(
        df=frame,
        audio_dir=audio_dir,
        sample_rate=sample_rate,
        duration_seconds=float(duration_seconds),
        augment=False,
        num_train_windows=1,
        eval_hop_ratio=0.5,
        max_eval_windows=1,
        eval_speech_aware=False,
        short_audio_mode="pad",
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    chunks: list[np.ndarray] = []
    observed_labels: list[np.ndarray] = []
    for waveforms, labels in loader:
        waveforms = waveforms.to(device, non_blocking=True)
        _, _, probabilities = forward_multi_window_evaluation(model, waveforms)
        chunks.append(probabilities.float().cpu().numpy())
        observed_labels.append(labels.numpy())
    probs = np.concatenate(chunks, axis=0).astype(np.float32)
    labels = np.concatenate(observed_labels).astype(np.int64)
    expected_labels = frame["label"].to_numpy(dtype=np.int64)
    if not np.array_equal(labels, expected_labels):
        raise RuntimeError("DataLoader label order differs from reference OOF")
    if probs.shape != (len(frame), known_count + 1):
        raise RuntimeError(f"Unexpected probability shape: {probs.shape}")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return probs, metric_bundle(labels, probs, known_count)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--reference-oof", type=Path, default=DEFAULT_REFERENCE_OOF)
    parser.add_argument("--audio-dir", type=Path, default=ROOT / "data/processed/audio_wav")
    parser.add_argument("--durations", type=float, nargs="+", default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    args = parser.parse_args()

    frame, split = load_reference(args.reference_oof)
    baseline_ckpt, baseline_config, baseline_map = load_checkpoint(args.baseline)
    treatment_ckpt, treatment_config, treatment_map = load_checkpoint(args.treatment)
    del baseline_ckpt, treatment_ckpt
    if baseline_map != treatment_map:
        raise ValueError("Baseline and treatment class maps differ")
    expected_split = {"scheme": "kfold", "folds": 3, "fold": 0, "seed": 42}
    if split != expected_split:
        raise ValueError(f"Reference split mismatch: {split} != {expected_split}")
    for name, config in (("baseline", baseline_config), ("treatment", treatment_config)):
        actual = dict((config.get("data", {}) or {}).get("split", {}) or {})
        if actual != expected_split:
            raise ValueError(f"{name} checkpoint split mismatch: {actual}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result: dict[str, Any] = {
        "contract": {
            "decision_policy": "Raw probability-average direct argmax",
            "crop_policy": "one deterministic leading crop per OOF file",
            "selection_or_tuning_allowed": False,
            "split": split,
            "rows": int(len(frame)),
            "unique_files": int(frame["audio_file"].nunique()),
        },
        "inputs": {
            "baseline": {"path": str(args.baseline), "sha256": sha256_file(args.baseline)},
            "treatment": {"path": str(args.treatment), "sha256": sha256_file(args.treatment)},
            "reference_oof": {
                "path": str(args.reference_oof),
                "sha256": sha256_file(args.reference_oof),
            },
        },
        "durations": {},
    }
    arrays: dict[str, np.ndarray] = {
        "files": frame["audio_file"].astype(str).to_numpy(),
        "labels": frame["label"].to_numpy(dtype=np.int64),
    }
    sample_rate = int((treatment_config.get("audio", {}) or {}).get("sample_rate", 16000))
    for duration in args.durations:
        if not (0.0 < float(duration) <= 6.0):
            raise ValueError(f"Duration outside preregistered (0,6] range: {duration}")
        key = f"{float(duration):g}s"
        baseline_probs, baseline_metrics = score_checkpoint(
            checkpoint_path=args.baseline,
            frame=frame,
            duration_seconds=duration,
            audio_dir=args.audio_dir,
            sample_rate=sample_rate,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        treatment_probs, treatment_metrics = score_checkpoint(
            checkpoint_path=args.treatment,
            frame=frame,
            duration_seconds=duration,
            audio_dir=args.audio_dir,
            sample_rate=sample_rate,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        result["durations"][key] = {
            "baseline": baseline_metrics,
            "treatment": treatment_metrics,
            "delta": {
                name: float(treatment_metrics[name] - baseline_metrics[name])
                for name in ("macro_f1", "known_accuracy", "ood_f1", "overall_accuracy")
            },
        }
        arrays[f"baseline_probs_{key}"] = baseline_probs
        arrays[f"treatment_probs_{key}"] = treatment_probs

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    np.savez_compressed(args.output_npz, **arrays)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
