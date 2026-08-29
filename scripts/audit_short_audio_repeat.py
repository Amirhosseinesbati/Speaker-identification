"""Three-fold, single-variable audit of repeat-padding for short audio.

The locked CAM++ + LME20 policy was evaluated with zero padding for audio
shorter than the eight-second inference window.  Residual-error analysis shows
that short, low-energy known files are over-represented among the remaining
errors.  This audit changes exactly one thing: ``short_audio_mode`` is set to
``repeat`` for both enrollment and validation inference.  Model weights,
folds, window count, LME aggregation and every decision parameter stay fixed.

There is no parameter selection in this script.  Each validation fold is a
held-out evaluation of the same predeclared transformation.  Long-file output
equivalence and the historical locked OOF score are hard provenance checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_control_oof_centroid_crossfit import (  # noqa: E402
    NUM_FOLDS,
    l2norm_rows,
    metric_bundle,
    metric_delta,
    sha256_file,
)
from scripts.analyze_lme20_asnorm_crossfit import (  # noqa: E402
    LOCKED_ALPHA,
    LOCKED_RAW_KAPPA,
    LOCKED_TAU,
    LOCKED_UNKNOWN_WEIGHT,
    decision_predictions,
    logmeanexp_group_scores,
)
from scripts.analyze_prototype_aggregation_crossfit import (  # noqa: E402
    group_indices,
)
from scripts.analyze_unknown_cluster_hypotheses_crossfit import (  # noqa: E402
    load_fold_inputs,
)
from src.data_pipeline import SpeakerDataset  # noqa: E402
from src.model_factory import create_model_from_config  # noqa: E402


LOCKED_LME_BETA = 20.0
LOCKED_BASELINE_MACRO_F1 = 0.9633564052154656
LONG_EQUIVALENCE_ATOL = 2e-5
MIN_AGGREGATE_MACRO_GAIN = 0.001
MAX_GUARDRAIL_DROP = 0.001


def digest_names(names: Iterable[str]) -> str:
    payload = "\n".join(map(str, names)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def atomic_savez(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def audio_short_mask(
    files: np.ndarray,
    audio_dir: Path,
    target_sample_rate: int,
    target_length: int,
) -> np.ndarray:
    """Classify files that remain shorter than one inference window.

    The processed competition corpus is WAV.  Reading headers is enough to
    derive post-resample duration without decoding the waveform a second time.
    """
    result = np.zeros(len(files), dtype=bool)
    for index, name in enumerate(files.astype(str)):
        path = audio_dir / name
        info = sf.info(str(path))
        estimated_resampled_frames = int(round(
            info.frames * float(target_sample_rate) / float(info.samplerate)
        ))
        result[index] = estimated_resampled_frames < int(target_length)
    return result


def make_eval_dataset(
    files: np.ndarray,
    labels: np.ndarray,
    config: dict,
    short_audio_mode: str,
) -> SpeakerDataset:
    frame = pd.DataFrame({
        "audio_file": files.astype(str),
        "label": labels.astype(np.int64),
    })
    audio = config["audio"]
    return SpeakerDataset(
        frame,
        config["data"]["audio_dir"],
        sample_rate=int(audio["sample_rate"]),
        duration_seconds=float(audio["duration_seconds"]),
        augment=False,
        min_valid_duration=float(audio.get("min_valid_duration", 1.0)),
        num_train_windows=int(audio.get("num_train_windows", 1)),
        eval_hop_ratio=float(audio.get("eval_hop_ratio", 0.5)),
        max_eval_windows=int(audio.get("max_eval_windows", 8)),
        eval_speech_aware=bool(audio.get("eval_speech_aware", False)),
        speech_relative_db=float(audio.get("speech_relative_db", 35.0)),
        short_audio_mode=short_audio_mode,
    )


def extract_repeat_outputs(
    *,
    model: torch.nn.Module,
    train_dataset: SpeakerDataset,
    validation_dataset: SpeakerDataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    fold: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader_args = {
        "batch_size": int(batch_size),
        "shuffle": False,
        "num_workers": int(num_workers),
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, **loader_args)
    validation_loader = DataLoader(validation_dataset, **loader_args)

    train_chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for windows, _ in tqdm(
            train_loader, desc=f"Fold {fold} repeat enrollment"
        ):
            train_chunks.append(
                model.embed(windows.to(device, non_blocking=True)).cpu().numpy()
            )
    train_embeddings = l2norm_rows(np.concatenate(train_chunks, axis=0))

    validation_probabilities: list[np.ndarray] = []
    validation_embeddings: list[np.ndarray] = []
    with torch.inference_mode():
        for windows, _ in tqdm(
            validation_loader, desc=f"Fold {fold} repeat validation"
        ):
            windows = windows.to(device, non_blocking=True)
            # This is the exact probability-average inference path used by
            # OOF generation and the leaderboard package.  Calling the method
            # once per file avoids accidentally averaging across batch rows.
            for sample_windows in windows:
                probabilities, embedding = model.predict_proba_and_embed(
                    sample_windows, temperature=1.0
                )
                validation_probabilities.append(probabilities.cpu().numpy())
                validation_embeddings.append(embedding.cpu().numpy())
    return (
        train_embeddings.astype(np.float32),
        np.stack(validation_probabilities).astype(np.float32),
        l2norm_rows(np.stack(validation_embeddings)).astype(np.float32),
    )


def validate_repeat_arrays(
    arrays: dict[str, np.ndarray], pad_artifact: dict, pad_oof: dict
) -> None:
    expected = {
        "train_files": (len(pad_artifact["train_files"]),),
        "train_embeddings": pad_artifact["train_embeddings"].shape,
        "competition_labels": (len(pad_artifact["train_files"]),),
        "unknown_cluster_ids": (len(pad_artifact["train_files"]),),
        "validation_files": (len(pad_oof["files"]),),
        "validation_labels": (len(pad_oof["files"]),),
        "validation_probabilities": pad_oof["competition_probs"].shape,
        "validation_embeddings": pad_oof["embeddings"].shape,
        "train_short_mask": (len(pad_artifact["train_files"]),),
        "validation_short_mask": (len(pad_oof["files"]),),
    }
    for key, shape in expected.items():
        if key not in arrays or arrays[key].shape != shape:
            actual = None if key not in arrays else arrays[key].shape
            raise RuntimeError(f"{key} shape {actual} != {shape}")
    if not np.array_equal(
        arrays["train_files"].astype(str), pad_artifact["train_files"].astype(str)
    ):
        raise RuntimeError("Repeat enrollment order differs from pad artifact")
    if not np.array_equal(
        arrays["validation_files"].astype(str), pad_oof["files"].astype(str)
    ):
        raise RuntimeError("Repeat validation order differs from OOF bundle")
    for key in (
        "train_embeddings",
        "validation_probabilities",
        "validation_embeddings",
    ):
        if not np.all(np.isfinite(arrays[key])):
            raise RuntimeError(f"{key} contains non-finite values")


def maximum_absolute_difference(
    candidate: np.ndarray, reference: np.ndarray, mask: np.ndarray
) -> float:
    if not np.any(mask):
        return 0.0
    return float(np.max(np.abs(candidate[mask] - reference[mask])))


def build_or_load_fold_cache(
    *,
    fold: int,
    pad_artifact: dict[str, np.ndarray],
    pad_metadata: dict,
    pad_oof: dict[str, np.ndarray],
    checkpoint_root: Path,
    cache_dir: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    short_audio_mode: str,
) -> tuple[dict[str, np.ndarray], dict]:
    profile = f"p0-campp-known446-ood-control-oof-f{fold}"
    checkpoint_path = checkpoint_root / profile / "campp_best_raw.pt"
    oof_path = checkpoint_root / profile / "campp_best_bundle" / "oof_predictions.npz"
    cache_path = cache_dir / f"fold{fold}_{short_audio_mode}_inference.npz"
    metadata_path = cache_path.with_suffix(".json")

    train_files = pad_artifact["train_files"].astype(str)
    validation_files = pad_oof["files"].astype(str)
    expected = {
        "schema_version": 1,
        "fold": int(fold),
        "short_audio_mode": short_audio_mode,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "pad_artifact_sha256": str(pad_metadata["artifact_sha256"]),
        "oof_sha256": sha256_file(oof_path),
        "train_file_sha256": digest_names(train_files),
        "validation_file_sha256": digest_names(validation_files),
        "train_files": int(len(train_files)),
        "validation_files": int(len(validation_files)),
    }
    if cache_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        reusable = (
            all(metadata.get(key) == value for key, value in expected.items())
            and metadata.get("artifact_sha256") == sha256_file(cache_path)
        )
        if reusable:
            with np.load(cache_path) as data:
                arrays = {key: data[key].copy() for key in data.files}
            validate_repeat_arrays(arrays, pad_artifact, pad_oof)
            return arrays, {**metadata, "cache_status": "reused"}

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    config = deepcopy(checkpoint["config"])
    if str(config["audio"].get("short_audio_mode", "pad")) != "pad":
        raise RuntimeError(f"Fold {fold} baseline checkpoint is not pad-mode")
    config["audio"]["short_audio_mode"] = short_audio_mode
    class_map = checkpoint["class_map"]
    model = create_model_from_config(
        config, num_known_speakers=len(class_map) - 1
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    train_labels = pad_artifact["competition_labels"].astype(np.int64)
    validation_labels = pad_oof["labels"].astype(np.int64)
    train_dataset = make_eval_dataset(
        train_files, train_labels, config, short_audio_mode
    )
    validation_dataset = make_eval_dataset(
        validation_files, validation_labels, config, short_audio_mode
    )
    train_embeddings, validation_probabilities, validation_embeddings = (
        extract_repeat_outputs(
            model=model,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            fold=fold,
        )
    )

    audio = config["audio"]
    audio_dir = resolve_repo_path(config["data"]["audio_dir"])
    target_sample_rate = int(audio["sample_rate"])
    target_length = int(target_sample_rate * float(audio["duration_seconds"]))
    train_short = audio_short_mask(
        train_files, audio_dir, target_sample_rate, target_length
    )
    validation_short = audio_short_mask(
        validation_files, audio_dir, target_sample_rate, target_length
    )
    arrays = {
        "train_files": train_files.astype(str),
        "train_embeddings": train_embeddings,
        "competition_labels": train_labels,
        "unknown_cluster_ids": pad_artifact["unknown_cluster_ids"].astype(np.int64),
        "validation_files": validation_files.astype(str),
        "validation_labels": validation_labels,
        "validation_probabilities": validation_probabilities,
        "validation_embeddings": validation_embeddings,
        "train_short_mask": train_short,
        "validation_short_mask": validation_short,
    }
    validate_repeat_arrays(arrays, pad_artifact, pad_oof)

    train_long = ~train_short
    validation_long = ~validation_short
    equivalence = {
        "tolerance": LONG_EQUIVALENCE_ATOL,
        "train_long_files": int(train_long.sum()),
        "validation_long_files": int(validation_long.sum()),
        "train_embedding_max_abs_diff": maximum_absolute_difference(
            train_embeddings,
            pad_artifact["train_embeddings"].astype(np.float32),
            train_long,
        ),
        "validation_probability_max_abs_diff": maximum_absolute_difference(
            validation_probabilities,
            pad_oof["competition_probs"].astype(np.float32),
            validation_long,
        ),
        "validation_embedding_max_abs_diff": maximum_absolute_difference(
            validation_embeddings,
            pad_oof["embeddings"].astype(np.float32),
            validation_long,
        ),
    }
    if max(
        equivalence["train_embedding_max_abs_diff"],
        equivalence["validation_probability_max_abs_diff"],
        equivalence["validation_embedding_max_abs_diff"],
    ) > LONG_EQUIVALENCE_ATOL:
        raise RuntimeError(f"Fold {fold} long-file equivalence failed: {equivalence}")

    atomic_savez(cache_path, arrays)
    metadata = {
        **expected,
        "artifact_sha256": sha256_file(cache_path),
        "audio_dir": str(audio_dir),
        "sample_rate": target_sample_rate,
        "duration_seconds": float(audio["duration_seconds"]),
        "max_eval_windows": int(audio.get("max_eval_windows", 8)),
        "eval_speech_aware": bool(audio.get("eval_speech_aware", False)),
        "train_short_files": int(train_short.sum()),
        "validation_short_files": int(validation_short.sum()),
        "long_file_equivalence": equivalence,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return arrays, {**metadata, "cache_status": "built"}


def locked_lme_predictions(
    artifact: dict[str, np.ndarray],
    validation_embeddings: np.ndarray,
    head_probabilities: np.ndarray,
) -> np.ndarray:
    groups = group_indices(artifact)
    scores = logmeanexp_group_scores(
        validation_embeddings,
        artifact["train_embeddings"],
        groups,
        beta=LOCKED_LME_BETA,
    )
    return decision_predictions(
        head=head_probabilities,
        scores=scores,
        probability_kappa=LOCKED_RAW_KAPPA,
        raw_max_scores=scores.max(axis=1),
    )


def short_subset_summary(
    labels: np.ndarray,
    baseline_predictions: np.ndarray,
    candidate_predictions: np.ndarray,
    short_mask: np.ndarray,
) -> dict[str, int | float]:
    selected = np.asarray(short_mask, dtype=bool)
    baseline_correct = baseline_predictions == labels
    candidate_correct = candidate_predictions == labels
    count = int(selected.sum())
    return {
        "files": count,
        "baseline_correct": int(np.sum(selected & baseline_correct)),
        "candidate_correct": int(np.sum(selected & candidate_correct)),
        "baseline_accuracy": float(np.mean(baseline_correct[selected])) if count else 0.0,
        "candidate_accuracy": float(np.mean(candidate_correct[selected])) if count else 0.0,
        "rescued_errors": int(np.sum(selected & ~baseline_correct & candidate_correct)),
        "introduced_errors": int(np.sum(selected & baseline_correct & ~candidate_correct)),
        "changed_predictions": int(np.sum(
            selected & (baseline_predictions != candidate_predictions)
        )),
    }


def acceptance_gate(fold_rows: list[dict], aggregate: dict) -> dict:
    conditions = {
        "all_three_folds_macro_positive": all(
            row["delta"]["macro_f1"] > 0.0 for row in fold_rows
        ),
        "aggregate_macro_gain_at_least_0_001": (
            aggregate["delta"]["macro_f1"] >= MIN_AGGREGATE_MACRO_GAIN
        ),
        "all_fold_known_guardrails": all(
            row["delta"]["known_accuracy"] >= -MAX_GUARDRAIL_DROP
            for row in fold_rows
        ),
        "all_fold_ood_guardrails": all(
            row["delta"]["ood_f1"] >= -MAX_GUARDRAIL_DROP
            for row in fold_rows
        ),
        "aggregate_known_guardrail": (
            aggregate["delta"]["known_accuracy"] >= -MAX_GUARDRAIL_DROP
        ),
        "aggregate_ood_guardrail": (
            aggregate["delta"]["ood_f1"] >= -MAX_GUARDRAIL_DROP
        ),
    }
    return {
        "passed": bool(all(conditions.values())),
        "conditions": conditions,
        "thresholds": {
            "minimum_aggregate_macro_gain": MIN_AGGREGATE_MACRO_GAIN,
            "maximum_known_or_ood_drop": MAX_GUARDRAIL_DROP,
            "fold_direction": "strictly positive Macro-F1 in all three folds",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-root", type=Path, default=ROOT / "checkpoints"
    )
    parser.add_argument(
        "--pad-cache-dir",
        type=Path,
        default=ROOT / "data" / "experiments" / "campp_control_centroid_crossfit",
    )
    parser.add_argument(
        "--repeat-cache-dir",
        type=Path,
        default=ROOT / "data" / "experiments" / "campp_short_audio_repeat",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "generated" / "campp_lme20_short_audio_repeat.json",
    )
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--short-audio-mode", default="repeat")
    args = parser.parse_args()
    if args.short_audio_mode not in {"repeat", "tile", "tile_speech"}:
        raise ValueError("The candidate must use explicit short-audio repetition")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    pad_oofs, pad_artifacts, pad_metadata = load_fold_inputs(
        args.checkpoint_root, args.pad_cache_dir
    )
    repeat_arrays = []
    repeat_metadata = []
    for fold in range(NUM_FOLDS):
        arrays, metadata = build_or_load_fold_cache(
            fold=fold,
            pad_artifact=pad_artifacts[fold],
            pad_metadata=pad_metadata[fold],
            pad_oof=pad_oofs[fold],
            checkpoint_root=args.checkpoint_root,
            cache_dir=args.repeat_cache_dir,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            short_audio_mode=args.short_audio_mode,
        )
        repeat_arrays.append(arrays)
        repeat_metadata.append(metadata)

    fold_rows = []
    all_files = []
    all_labels = []
    all_pad_predictions = []
    all_repeat_predictions = []
    all_short_masks = []
    for fold in range(NUM_FOLDS):
        pad_oof = pad_oofs[fold]
        pad_artifact = pad_artifacts[fold]
        repeat = repeat_arrays[fold]
        repeat_artifact = {
            "train_embeddings": repeat["train_embeddings"],
            "competition_labels": repeat["competition_labels"],
            "unknown_cluster_ids": repeat["unknown_cluster_ids"],
        }
        pad_predictions = locked_lme_predictions(
            pad_artifact,
            pad_oof["embeddings"],
            pad_oof["competition_probs"],
        )
        repeat_predictions = locked_lme_predictions(
            repeat_artifact,
            repeat["validation_embeddings"],
            repeat["validation_probabilities"],
        )
        labels = pad_oof["labels"].astype(np.int64)
        baseline = metric_bundle(labels, pad_predictions)
        candidate = metric_bundle(labels, repeat_predictions)
        baseline_correct = pad_predictions == labels
        candidate_correct = repeat_predictions == labels
        fold_rows.append({
            "fold": fold,
            "baseline": baseline,
            "candidate": candidate,
            "delta": metric_delta(candidate, baseline),
            "rescued_errors": int(np.sum(~baseline_correct & candidate_correct)),
            "introduced_errors": int(np.sum(baseline_correct & ~candidate_correct)),
            "changed_predictions": int(np.sum(pad_predictions != repeat_predictions)),
            "short_audio": short_subset_summary(
                labels,
                pad_predictions,
                repeat_predictions,
                repeat["validation_short_mask"],
            ),
        })
        all_files.append(pad_oof["files"].astype(str))
        all_labels.append(labels)
        all_pad_predictions.append(pad_predictions)
        all_repeat_predictions.append(repeat_predictions)
        all_short_masks.append(repeat["validation_short_mask"].astype(bool))

    files = np.concatenate(all_files)
    if len(set(files.tolist())) != len(files):
        raise RuntimeError("OOF validation files overlap across folds")
    labels = np.concatenate(all_labels)
    pad_predictions = np.concatenate(all_pad_predictions)
    repeat_predictions = np.concatenate(all_repeat_predictions)
    short_mask = np.concatenate(all_short_masks)
    baseline = metric_bundle(labels, pad_predictions)
    candidate = metric_bundle(labels, repeat_predictions)
    aggregate = {
        "baseline": baseline,
        "candidate": candidate,
        "delta": metric_delta(candidate, baseline),
        "rescued_errors": int(np.sum(
            (pad_predictions != labels) & (repeat_predictions == labels)
        )),
        "introduced_errors": int(np.sum(
            (pad_predictions == labels) & (repeat_predictions != labels)
        )),
        "changed_predictions": int(np.sum(pad_predictions != repeat_predictions)),
        "short_audio": short_subset_summary(
            labels, pad_predictions, repeat_predictions, short_mask
        ),
    }
    if abs(baseline["macro_f1"] - LOCKED_BASELINE_MACRO_F1) > 1e-10:
        raise RuntimeError(
            "Locked pad baseline reproduction failed: "
            f"{baseline['macro_f1']} != {LOCKED_BASELINE_MACRO_F1}"
        )

    gate = acceptance_gate(fold_rows, aggregate)
    report = {
        "contract": {
            "hypothesis": (
                "repeat short audio to eight seconds instead of zero-padding"
            ),
            "single_changed_variable": "audio.short_audio_mode: pad -> repeat",
            "weights": "fixed selected Raw CAM++ Control Fold0/1/2 checkpoints",
            "folds": "fixed kfold/folds3/seed42 OOF",
            "enrollment_and_query_preprocessing_match": True,
            "lme_beta": LOCKED_LME_BETA,
            "alpha": LOCKED_ALPHA,
            "kappa": LOCKED_RAW_KAPPA,
            "tau": LOCKED_TAU,
            "lambda_unknown": LOCKED_UNKNOWN_WEIGHT,
            "parameter_selection": "none",
            "leaderboard_tuning": False,
        },
        "provenance": {
            "pad_cache_metadata": pad_metadata,
            "repeat_cache_metadata": repeat_metadata,
            "unique_oof_files": int(len(files)),
            "oof_file_sha256": digest_names(files),
        },
        "folds": fold_rows,
        "aggregate": aggregate,
        "acceptance_gate": gate,
        "decision": "accept" if gate["passed"] else "reject",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({
        "output": str(args.output),
        "decision": report["decision"],
        "fold_deltas": [row["delta"] for row in fold_rows],
        "aggregate": aggregate,
        "gate": gate,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
