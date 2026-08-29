"""Leak-free three-fold audit of the 1000-centroid CAM++ decision rule.

The competition collapses 554 latent speakers into the single ``unknown``
label.  This script tests whether recovering those train-side pseudo identities
as embedding centroids generalises beyond the Fold-0 result that motivated the
current leaderboard package.

Scientific contract
-------------------
* Rebuild the exact speaker-aware folds and assert every validation filename
  against each checkpoint's OOF bundle.
* Build known and pseudo-unknown centroids from the matching training fold only.
* Select all decision parameters for a target fold using the other two folds.
* Keep the historical Fold-0 parameters as a fixed, untuned reference.
* Report Macro-F1, Known Accuracy, binary OOD-F1 and error topology together.

The expensive train embedding extraction is cached with checkpoint, cluster-map
and split hashes so later decision analyses do not repeat GPU inference.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_pipeline import (  # noqa: E402
    SpeakerDataset,
    clean_conflicting_labels,
    create_class_mapping,
    ensure_target_columns,
    find_corrupted_files,
    find_duplicate_groups,
    speaker_aware_kfold,
)
from src.metrics import macro_f1_score  # noqa: E402
from src.model_factory import create_model_from_config  # noqa: E402
from submission.inference import (  # noqa: E402
    _collapse_centroid_probs,
    centroid_probs_matrix,
)


NUM_FOLDS = 3
NUM_KNOWN = 446
NUM_UNKNOWN_CLUSTERS = 554
NUM_CLASSES = NUM_KNOWN + 1
MIN_VALID_DURATION = 1.0
EVAL_HOP_RATIO = 0.5
MAX_EVAL_WINDOWS = 8

ALPHAS = (0.0, 0.15, 0.30, 0.50)
KAPPAS = (16.0, 24.0, 32.0, 40.0)
TAUS = (0.0, 0.48, 0.50, 0.52, 0.54, 0.56)
UNKNOWN_WEIGHTS = (0.50, 0.75, 1.00)
HISTORICAL_PARAMS = {
    "alpha": 0.0,
    "kappa": 32.0,
    "tau": 0.52,
    "lambda_unknown": 0.50,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_names(names: Iterable[str]) -> str:
    payload = "\n".join(sorted(map(str, names))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def l2norm_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / (np.linalg.norm(values, axis=1, keepdims=True) + 1e-12)


def metric_bundle(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    known = labels > 0
    unknown = ~known
    true_ood = unknown
    predicted_ood = predictions == 0
    tp = int(np.sum(true_ood & predicted_ood))
    fp = int(np.sum(~true_ood & predicted_ood))
    fn = int(np.sum(true_ood & ~predicted_ood))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    ood_f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "macro_f1": float(macro_f1_score(labels, predictions, NUM_CLASSES)),
        "accuracy": float(np.mean(labels == predictions)),
        "known_accuracy": float(np.mean(labels[known] == predictions[known])),
        "ood_f1": float(ood_f1),
        "ood_precision": float(precision),
        "ood_recall": float(recall),
        "known_to_unknown": int(np.sum(known & (predictions == 0))),
        "known_to_wrong_known": int(np.sum(
            known & (predictions > 0) & (predictions != labels)
        )),
        "unknown_to_known": int(np.sum(unknown & (predictions > 0))),
    }


def metric_delta(candidate: dict, baseline: dict) -> dict[str, float]:
    return {
        key: float(candidate[key] - baseline[key])
        for key in ("macro_f1", "accuracy", "known_accuracy", "ood_f1")
    }


def parameter_grid() -> list[dict[str, float]]:
    return [
        {
            "alpha": float(alpha),
            "kappa": float(kappa),
            "tau": float(tau),
            "lambda_unknown": float(weight),
        }
        for alpha, kappa, tau, weight in itertools.product(
            ALPHAS, KAPPAS, TAUS, UNKNOWN_WEIGHTS
        )
    ]


@dataclass
class FoldEvidence:
    fold: int
    files: np.ndarray
    labels: np.ndarray
    head: np.ndarray
    embeddings: np.ndarray
    known_centroids: np.ndarray
    unknown_centroids: np.ndarray
    centroid_probabilities: dict[float, np.ndarray]
    max_cosines: dict[float, np.ndarray]

    @property
    def baseline_predictions(self) -> np.ndarray:
        return self.head.argmax(axis=1).astype(np.int64)


def rebuild_exact_splits(labels_path: Path, audio_dir: Path) -> tuple[list, dict]:
    """Recreate all folds once, preserving the training pipeline's cleaning."""
    frame = pd.read_csv(labels_path)
    frame.columns = frame.columns.str.strip()
    frame = frame.drop_duplicates().dropna(
        subset=["speaker_id", "audio_file"]
    ).reset_index(drop=True)
    frame = ensure_target_columns(frame)
    corrupted = find_corrupted_files(frame, str(audio_dir), MIN_VALID_DURATION)
    duplicate_groups = find_duplicate_groups(frame, str(audio_dir))
    frame, duplicate_stats = clean_conflicting_labels(frame, str(audio_dir))
    class_map = create_class_mapping(frame)
    frame["metric_label"] = frame["speaker_id"].map(class_map).astype(int)
    frame["label"] = frame["metric_label"]
    splits = speaker_aware_kfold(
        frame,
        folds=NUM_FOLDS,
        random_seed=42,
        duplicate_groups=duplicate_groups,
        corrupted_files=set(corrupted),
    )
    return splits, {
        "source_rows": int(len(frame)),
        "corrupted_files": sorted(corrupted),
        "duplicate_groups": int(len(duplicate_groups)),
        "duplicate_cleaning": duplicate_stats,
    }


def load_oof(path: Path, expected_fold: int, expected_files: set[str]) -> dict:
    with np.load(path) as data:
        values = {key: data[key].copy() for key in data.files}
    files = values["files"].astype(str)
    actual_files = set(files.tolist())
    if actual_files != expected_files or len(actual_files) != len(files):
        raise RuntimeError(
            f"Fold {expected_fold} OOF/split mismatch: "
            f"oof={len(actual_files)}, expected={len(expected_files)}, "
            f"missing={len(expected_files - actual_files)}, "
            f"extra={len(actual_files - expected_files)}"
        )
    if int(np.asarray(values["split_fold"]).reshape(-1)[0]) != expected_fold:
        raise RuntimeError(f"Fold marker mismatch in {path}")
    if int(np.asarray(values["split_folds"]).reshape(-1)[0]) != NUM_FOLDS:
        raise RuntimeError(f"Fold count mismatch in {path}")
    if int(np.asarray(values["split_seed"]).reshape(-1)[0]) != 42:
        raise RuntimeError(f"Split seed mismatch in {path}")
    if values["competition_probs"].shape != (len(files), NUM_CLASSES):
        raise RuntimeError(f"Unexpected probability shape in {path}")
    return values


def build_or_load_train_artifact(
    *,
    fold: int,
    train_frame: pd.DataFrame,
    checkpoint_path: Path,
    cluster_map_path: Path,
    cache_path: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[dict[str, np.ndarray], dict]:
    checkpoint_sha = sha256_file(checkpoint_path)
    cluster_sha = sha256_file(cluster_map_path)
    split_sha = digest_names(train_frame["audio_file"].astype(str))
    metadata_path = cache_path.with_suffix(".json")
    expected = {
        "fold": fold,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "cluster_map": str(cluster_map_path),
        "cluster_map_sha256": cluster_sha,
        "train_file_sha256": split_sha,
        "train_files": int(len(train_frame)),
        "max_eval_windows": MAX_EVAL_WINDOWS,
    }
    if cache_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if all(metadata.get(key) == value for key, value in expected.items()):
            with np.load(cache_path) as data:
                arrays = {key: data[key].copy() for key in data.files}
            return arrays, metadata

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    checkpoint_class_map = checkpoint["class_map"]
    model = create_model_from_config(
        config, num_known_speakers=len(checkpoint_class_map) - 1
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    frame = train_frame.copy().reset_index(drop=True)
    frame["label"] = frame["speaker_id"].map(checkpoint_class_map)
    if frame["label"].isna().any():
        missing = sorted(frame.loc[frame["label"].isna(), "speaker_id"].unique())
        raise RuntimeError(f"Checkpoint class map misses training labels: {missing[:3]}")
    frame["label"] = frame["label"].astype(int)

    cluster_map = {
        str(name): int(cluster_id)
        for name, cluster_id in json.loads(
            cluster_map_path.read_text(encoding="utf-8")
        ).items()
    }
    unknown_files = set(
        frame.loc[frame["speaker_id"] == "unknown", "audio_file"].astype(str)
    )
    if set(cluster_map) != unknown_files:
        raise RuntimeError(
            f"Fold {fold} cluster-map/train mismatch: "
            f"missing={len(unknown_files - set(cluster_map))}, "
            f"extra={len(set(cluster_map) - unknown_files)}"
        )
    cluster_ids = np.full(len(frame), -1, dtype=np.int64)
    unknown_mask = frame["speaker_id"].eq("unknown").to_numpy()
    cluster_ids[unknown_mask] = frame.loc[
        unknown_mask, "audio_file"
    ].map(cluster_map).to_numpy(np.int64)
    if set(cluster_ids[unknown_mask].tolist()) != set(range(NUM_UNKNOWN_CLUSTERS)):
        raise RuntimeError(f"Fold {fold} does not contain dense 0..553 clusters")

    audio = config["audio"]
    dataset = SpeakerDataset(
        frame,
        config["data"]["audio_dir"],
        sample_rate=audio["sample_rate"],
        duration_seconds=audio["duration_seconds"],
        augment=False,
        min_valid_duration=audio.get("min_valid_duration", MIN_VALID_DURATION),
        num_train_windows=audio.get("num_train_windows", 1),
        eval_hop_ratio=audio.get("eval_hop_ratio", EVAL_HOP_RATIO),
        max_eval_windows=audio.get("max_eval_windows", MAX_EVAL_WINDOWS),
        eval_speech_aware=audio.get("eval_speech_aware", False),
        speech_relative_db=audio.get("speech_relative_db", 35.0),
        short_audio_mode=audio.get("short_audio_mode", "pad"),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    embedding_batches = []
    with torch.inference_mode():
        for windows, _ in tqdm(loader, desc=f"Fold {fold} train embeddings"):
            embedding_batches.append(model.embed(windows.to(device)).cpu().numpy())
    embeddings = l2norm_rows(np.concatenate(embedding_batches, axis=0))

    known_centroids = np.zeros((NUM_KNOWN, embeddings.shape[1]), np.float32)
    known_sizes = np.zeros(NUM_KNOWN, np.int64)
    labels = frame["label"].to_numpy(np.int64)
    for speaker_id in range(1, NUM_KNOWN + 1):
        mask = labels == speaker_id
        if not np.any(mask):
            raise RuntimeError(f"Fold {fold} has no train file for known {speaker_id}")
        known_sizes[speaker_id - 1] = int(mask.sum())
        known_centroids[speaker_id - 1] = embeddings[mask].mean(axis=0)

    unknown_centroids = np.zeros(
        (NUM_UNKNOWN_CLUSTERS, embeddings.shape[1]), np.float32
    )
    unknown_sizes = np.zeros(NUM_UNKNOWN_CLUSTERS, np.int64)
    for cluster_id in range(NUM_UNKNOWN_CLUSTERS):
        mask = cluster_ids == cluster_id
        unknown_sizes[cluster_id] = int(mask.sum())
        unknown_centroids[cluster_id] = embeddings[mask].mean(axis=0)

    arrays = {
        "train_files": frame["audio_file"].astype(str).to_numpy(),
        "train_embeddings": embeddings.astype(np.float32),
        "competition_labels": np.where(unknown_mask, 0, labels).astype(np.int64),
        "unknown_cluster_ids": cluster_ids,
        "known_centroids": l2norm_rows(known_centroids),
        "known_sizes": known_sizes,
        "unknown_centroids": l2norm_rows(unknown_centroids),
        "unknown_sizes": unknown_sizes,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **arrays)
    metadata = {
        **expected,
        "embedding_dim": int(embeddings.shape[1]),
        "known_train_files": int((~unknown_mask).sum()),
        "unknown_train_files": int(unknown_mask.sum()),
        "known_size_min": int(known_sizes.min()),
        "known_size_max": int(known_sizes.max()),
        "unknown_size_min": int(unknown_sizes.min()),
        "unknown_size_max": int(unknown_sizes.max()),
        "artifact_sha256": sha256_file(cache_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return arrays, metadata


def prepare_fold_evidence(
    *, fold: int, oof: dict, artifact: dict[str, np.ndarray]
) -> FoldEvidence:
    embeddings = l2norm_rows(oof["embeddings"])
    known = l2norm_rows(artifact["known_centroids"])
    unknown = l2norm_rows(artifact["unknown_centroids"])
    all_centroids = np.vstack([known, unknown])
    speaker_ids = np.arange(1, 1 + len(all_centroids), dtype=np.int64)
    centroid_probabilities = {}
    max_cosines = {}
    for kappa in KAPPAS:
        probabilities, max_cosine = centroid_probs_matrix(
            embeddings, all_centroids, speaker_ids, 1 + len(all_centroids), kappa
        )
        centroid_probabilities[kappa] = _collapse_centroid_probs(
            probabilities, NUM_CLASSES
        ).astype(np.float64)
        max_cosines[kappa] = max_cosine.astype(np.float64)
    return FoldEvidence(
        fold=fold,
        files=oof["files"].astype(str),
        labels=oof["labels"].astype(np.int64),
        head=oof["competition_probs"].astype(np.float64),
        embeddings=embeddings,
        known_centroids=known,
        unknown_centroids=unknown,
        centroid_probabilities=centroid_probabilities,
        max_cosines=max_cosines,
    )


def predict(evidence: FoldEvidence, params: dict[str, float]) -> np.ndarray:
    centroid = evidence.centroid_probabilities[float(params["kappa"])]
    fused = (
        float(params["alpha"]) * evidence.head
        + (1.0 - float(params["alpha"])) * centroid
    )
    fused = fused.copy()
    fused[:, 0] *= float(params["lambda_unknown"])
    fused /= fused.sum(axis=1, keepdims=True) + 1e-12
    predictions = fused.argmax(axis=1).astype(np.int64)
    predictions[
        evidence.max_cosines[float(params["kappa"])] < float(params["tau"])
    ] = 0
    return predictions


def select_on_folds(
    evidence: list[FoldEvidence], calibration_folds: tuple[int, int]
) -> tuple[dict[str, float], dict]:
    baselines = {
        fold: metric_bundle(
            evidence[fold].labels, evidence[fold].baseline_predictions
        )
        for fold in calibration_folds
    }
    ranked = []
    for params in parameter_grid():
        per_fold = {}
        gains = []
        for fold in calibration_folds:
            candidate = metric_bundle(
                evidence[fold].labels, predict(evidence[fold], params)
            )
            per_fold[fold] = candidate
            gains.append(candidate["macro_f1"] - baselines[fold]["macro_f1"])
        # Max-min selection makes direction consistency primary. Mean gain and
        # closeness to the already shipped historical rule are deterministic
        # tie breakers, not extra target-fold tuning knobs.
        distance = (
            abs(params["alpha"] - HISTORICAL_PARAMS["alpha"])
            + abs(params["kappa"] - HISTORICAL_PARAMS["kappa"]) / 32.0
            + abs(params["tau"] - HISTORICAL_PARAMS["tau"])
            + abs(params["lambda_unknown"] - HISTORICAL_PARAMS["lambda_unknown"])
        )
        rank = (min(gains), float(np.mean(gains)), -distance)
        ranked.append((rank, params, per_fold, gains))
    rank, params, per_fold, gains = max(ranked, key=lambda item: item[0])
    return params, {
        "selection_objective": "maximise minimum calibration-fold Macro-F1 gain",
        "calibration_folds": list(calibration_folds),
        "minimum_gain": float(min(gains)),
        "mean_gain": float(np.mean(gains)),
        "per_fold_metrics": {str(key): value for key, value in per_fold.items()},
        "rank_tuple": [float(value) for value in rank],
    }


def aggregate_predictions(
    evidence: list[FoldEvidence], per_fold_predictions: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    files = np.concatenate([fold.files for fold in evidence])
    if len(set(files.tolist())) != len(files):
        raise RuntimeError("OOF files overlap across folds")
    labels = np.concatenate([fold.labels for fold in evidence])
    predictions = np.concatenate(per_fold_predictions)
    return files, labels, predictions


def evaluate_policy(
    evidence: list[FoldEvidence], per_fold_predictions: list[np.ndarray]
) -> dict:
    fold_rows = []
    for fold, predictions in enumerate(per_fold_predictions):
        baseline = metric_bundle(
            evidence[fold].labels, evidence[fold].baseline_predictions
        )
        candidate = metric_bundle(evidence[fold].labels, predictions)
        fold_rows.append({
            "fold": fold,
            "baseline": baseline,
            "candidate": candidate,
            "delta": metric_delta(candidate, baseline),
        })
    _, labels, predictions = aggregate_predictions(evidence, per_fold_predictions)
    baseline_predictions = np.concatenate(
        [fold.baseline_predictions for fold in evidence]
    )
    baseline = metric_bundle(labels, baseline_predictions)
    candidate = metric_bundle(labels, predictions)
    baseline_correct = baseline_predictions == labels
    candidate_correct = predictions == labels
    return {
        "folds": fold_rows,
        "aggregate": {
            "baseline": baseline,
            "candidate": candidate,
            "delta": metric_delta(candidate, baseline),
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
        "--cluster-root", type=Path, default=ROOT / "data" / "processed"
    )
    parser.add_argument(
        "--labels", type=Path,
        default=ROOT / "data" / "processed" / "audio_wav_labels.csv",
    )
    parser.add_argument(
        "--audio-dir", type=Path,
        default=ROOT / "data" / "processed" / "audio_wav",
    )
    parser.add_argument(
        "--cache-dir", type=Path,
        default=ROOT / "data" / "experiments" / "campp_control_centroid_crossfit",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "reports" / "generated"
        / "campp_control_1000centroid_crossfit.json",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    splits, cleaning = rebuild_exact_splits(args.labels, args.audio_dir)
    evidence = []
    artifacts = []
    checkpoint_shas = []
    for fold in range(NUM_FOLDS):
        profile = f"p0-campp-known446-ood-control-oof-f{fold}"
        checkpoint_dir = args.checkpoint_root / profile
        checkpoint_path = checkpoint_dir / "campp_best_raw.pt"
        oof_path = checkpoint_dir / "campp_best_bundle" / "oof_predictions.npz"
        cluster_path = args.cluster_root / f"unknown_clusters_oof_f{fold}.json"
        train_frame, validation_frame = splits[fold]
        oof = load_oof(
            oof_path,
            expected_fold=fold,
            expected_files=set(validation_frame["audio_file"].astype(str)),
        )
        arrays, metadata = build_or_load_train_artifact(
            fold=fold,
            train_frame=train_frame,
            checkpoint_path=checkpoint_path,
            cluster_map_path=cluster_path,
            cache_path=args.cache_dir / f"fold{fold}_train_embeddings_centroids.npz",
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        evidence.append(prepare_fold_evidence(fold=fold, oof=oof, artifact=arrays))
        artifacts.append(metadata)
        checkpoint_shas.append(metadata["checkpoint_sha256"])

    crossfit_predictions = []
    selections = []
    for target in range(NUM_FOLDS):
        calibration = tuple(fold for fold in range(NUM_FOLDS) if fold != target)
        params, selection = select_on_folds(evidence, calibration)  # type: ignore[arg-type]
        prediction = predict(evidence[target], params)
        target_baseline = metric_bundle(
            evidence[target].labels, evidence[target].baseline_predictions
        )
        target_candidate = metric_bundle(evidence[target].labels, prediction)
        selections.append({
            "target_fold": target,
            "parameters": params,
            "calibration": selection,
            "held_out": {
                "baseline": target_baseline,
                "candidate": target_candidate,
                "delta": metric_delta(target_candidate, target_baseline),
            },
        })
        crossfit_predictions.append(prediction)

    historical_predictions = [
        predict(fold, HISTORICAL_PARAMS) for fold in evidence
    ]
    crossfit_evaluation = evaluate_policy(evidence, crossfit_predictions)
    historical_evaluation = evaluate_policy(evidence, historical_predictions)
    crossfit_delta = crossfit_evaluation["aggregate"]["delta"]
    fold_macro_deltas = [
        row["delta"]["macro_f1"] for row in crossfit_evaluation["folds"]
    ]
    gate = {
        "minimum_aggregate_macro_gain": 0.002,
        "minimum_positive_folds": 2,
        "minimum_worst_fold_gain": -0.002,
        "minimum_aggregate_known_accuracy_delta": -0.006,
        "minimum_aggregate_ood_f1_delta": -0.002,
    }
    gate_checks = {
        "aggregate_macro_gain": crossfit_delta["macro_f1"] >= 0.002,
        "positive_folds": int(sum(delta > 0 for delta in fold_macro_deltas)) >= 2,
        "worst_fold_gain": min(fold_macro_deltas) >= -0.002,
        "known_accuracy": crossfit_delta["known_accuracy"] >= -0.006,
        "ood_f1": crossfit_delta["ood_f1"] >= -0.002,
    }
    report = {
        "contract": {
            "scope": "CAM++ Control Fold0/1/2; train-only 446+554 centroids",
            "selection": "leave-one-fold-out; target fold never selects parameters",
            "parameter_grid": {
                "alpha": ALPHAS,
                "kappa": KAPPAS,
                "tau": TAUS,
                "lambda_unknown": UNKNOWN_WEIGHTS,
                "candidates": len(parameter_grid()),
            },
            "historical_fixed_parameters": HISTORICAL_PARAMS,
            "decision_metric": "447-class Macro-F1",
        },
        "provenance": {
            "cleaning": cleaning,
            "checkpoint_sha256": checkpoint_shas,
            "artifacts": artifacts,
            "oof_rows": [int(len(fold.files)) for fold in evidence],
            "oof_unique": int(len(set(np.concatenate(
                [fold.files for fold in evidence]
            ).tolist()))),
        },
        "crossfit": {
            "selections": selections,
            "evaluation": crossfit_evaluation,
            "gate": gate,
            "gate_checks": gate_checks,
            "accepted": bool(all(gate_checks.values())),
        },
        "historical_fixed": historical_evaluation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "crossfit_aggregate": crossfit_evaluation["aggregate"],
        "crossfit_fold_macro_deltas": fold_macro_deltas,
        "crossfit_parameters": [row["parameters"] for row in selections],
        "gate_checks": gate_checks,
        "accepted": bool(all(gate_checks.values())),
        "historical_fixed_aggregate": historical_evaluation["aggregate"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
