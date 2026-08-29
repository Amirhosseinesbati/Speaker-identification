"""Three-fold audit of native-duration CAM++ inference for short audio.

This is a fixed, representation-level candidate.  Files at least eight seconds
long reuse their locked pad-mode outputs bit-for-bit.  Short files are decoded
normally but are passed to CAM++ at their true length, with no zero padding and
no periodic repetition.  Enrollment and validation are transformed together;
the locked LME20 decision rule is unchanged and there is no parameter search.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_control_oof_centroid_crossfit import (  # noqa: E402
    NUM_FOLDS,
    metric_bundle,
    metric_delta,
    sha256_file,
)
from scripts.analyze_lme20_asnorm_crossfit import (  # noqa: E402
    LOCKED_ALPHA,
    LOCKED_RAW_KAPPA,
    LOCKED_TAU,
    LOCKED_UNKNOWN_WEIGHT,
)
from scripts.analyze_unknown_cluster_hypotheses_crossfit import (  # noqa: E402
    load_fold_inputs,
)
from scripts.audit_short_audio_repeat import (  # noqa: E402
    LOCKED_BASELINE_MACRO_F1,
    LOCKED_LME_BETA,
    LONG_EQUIVALENCE_ATOL,
    acceptance_gate,
    atomic_savez,
    audio_short_mask,
    digest_names,
    locked_lme_predictions,
    make_eval_dataset,
    resolve_repo_path,
    short_subset_summary,
    validate_repeat_arrays,
)
from src.model_factory import create_model_from_config  # noqa: E402


NATIVE_MODE = "native_length"


def infer_native_short_rows(
    *,
    model: torch.nn.Module,
    dataset,
    files: np.ndarray,
    short_mask: np.ndarray,
    device: torch.device,
    base_embeddings: np.ndarray,
    base_probabilities: np.ndarray | None,
    description: str,
) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """Replace only short rows, preserving every long row bit-for-bit."""
    embeddings = np.asarray(base_embeddings, dtype=np.float32).copy()
    probabilities = (
        None
        if base_probabilities is None
        else np.asarray(base_probabilities, dtype=np.float32).copy()
    )
    indices = np.flatnonzero(np.asarray(short_mask, dtype=bool))
    observed_lengths: list[int] = []
    with torch.inference_mode():
        for index in tqdm(indices, desc=description):
            waveform = dataset._load_audio(dataset.audio_dir / str(files[index]))
            length = int(waveform.size(-1))
            if length <= 0 or length >= int(dataset.target_length):
                raise RuntimeError(
                    f"Short-mask mismatch for {files[index]}: {length} samples"
                )
            observed_lengths.append(length)
            native_batch = waveform.unsqueeze(0).to(device)  # (1, 1, N)
            if probabilities is None:
                embedding = model.embed(native_batch)[0]
            else:
                probability, embedding = model.predict_proba_and_embed(
                    native_batch, temperature=1.0
                )
                probabilities[index] = probability.cpu().numpy()
            embeddings[index] = embedding.cpu().numpy()
    short_norms = np.linalg.norm(embeddings[indices], axis=1)
    if len(indices) and not np.allclose(short_norms, 1.0, atol=2e-5):
        raise RuntimeError("Native-length model returned non-unit embeddings")
    diagnostics = {
        "short_rows": int(len(indices)),
        "minimum_samples": int(min(observed_lengths)) if observed_lengths else 0,
        "median_samples": (
            float(np.median(observed_lengths)) if observed_lengths else 0.0
        ),
        "maximum_samples": int(max(observed_lengths)) if observed_lengths else 0,
    }
    return embeddings.astype(np.float32), probabilities, diagnostics


def build_or_load_native_fold_cache(
    *,
    fold: int,
    pad_artifact: dict[str, np.ndarray],
    pad_metadata: dict,
    pad_oof: dict[str, np.ndarray],
    checkpoint_root: Path,
    cache_dir: Path,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict]:
    profile = f"p0-campp-known446-ood-control-oof-f{fold}"
    checkpoint_path = checkpoint_root / profile / "campp_best_raw.pt"
    oof_path = checkpoint_root / profile / "campp_best_bundle" / "oof_predictions.npz"
    cache_path = cache_dir / f"fold{fold}_{NATIVE_MODE}_inference.npz"
    metadata_path = cache_path.with_suffix(".json")
    train_files = pad_artifact["train_files"].astype(str)
    validation_files = pad_oof["files"].astype(str)
    expected = {
        "schema_version": 1,
        "fold": int(fold),
        "short_audio_mode": NATIVE_MODE,
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
    class_map = checkpoint["class_map"]
    model = create_model_from_config(
        config, num_known_speakers=len(class_map) - 1
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    train_labels = pad_artifact["competition_labels"].astype(np.int64)
    validation_labels = pad_oof["labels"].astype(np.int64)
    train_dataset = make_eval_dataset(train_files, train_labels, config, "pad")
    validation_dataset = make_eval_dataset(
        validation_files, validation_labels, config, "pad"
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

    train_embeddings, _, train_diagnostics = infer_native_short_rows(
        model=model,
        dataset=train_dataset,
        files=train_files,
        short_mask=train_short,
        device=device,
        base_embeddings=pad_artifact["train_embeddings"],
        base_probabilities=None,
        description=f"Fold {fold} native enrollment",
    )
    validation_embeddings, validation_probabilities, validation_diagnostics = (
        infer_native_short_rows(
            model=model,
            dataset=validation_dataset,
            files=validation_files,
            short_mask=validation_short,
            device=device,
            base_embeddings=pad_oof["embeddings"],
            base_probabilities=pad_oof["competition_probs"],
            description=f"Fold {fold} native validation",
        )
    )
    assert validation_probabilities is not None
    arrays = {
        "train_files": train_files.astype(str),
        "train_embeddings": train_embeddings,
        "competition_labels": train_labels,
        "unknown_cluster_ids": pad_artifact["unknown_cluster_ids"].astype(np.int64),
        "validation_files": validation_files.astype(str),
        "validation_labels": validation_labels,
        "validation_probabilities": validation_probabilities.astype(np.float32),
        "validation_embeddings": validation_embeddings,
        "train_short_mask": train_short,
        "validation_short_mask": validation_short,
    }
    validate_repeat_arrays(arrays, pad_artifact, pad_oof)

    train_long = ~train_short
    validation_long = ~validation_short
    equivalence = {
        "tolerance": 0.0,
        "train_long_files": int(train_long.sum()),
        "validation_long_files": int(validation_long.sum()),
        "train_embedding_max_abs_diff": float(np.max(np.abs(
            arrays["train_embeddings"][train_long]
            - pad_artifact["train_embeddings"][train_long]
        ))),
        "validation_probability_max_abs_diff": float(np.max(np.abs(
            arrays["validation_probabilities"][validation_long]
            - pad_oof["competition_probs"][validation_long]
        ))),
        "validation_embedding_max_abs_diff": float(np.max(np.abs(
            arrays["validation_embeddings"][validation_long]
            - pad_oof["embeddings"][validation_long]
        ))),
    }
    if max(
        equivalence["train_embedding_max_abs_diff"],
        equivalence["validation_probability_max_abs_diff"],
        equivalence["validation_embedding_max_abs_diff"],
    ) != 0.0:
        raise RuntimeError(f"Fold {fold} copied long rows are not bitwise equal")

    atomic_savez(cache_path, arrays)
    metadata = {
        **expected,
        "artifact_sha256": sha256_file(cache_path),
        "audio_dir": str(audio_dir),
        "sample_rate": target_sample_rate,
        "duration_seconds": float(audio["duration_seconds"]),
        "train_short_files": int(train_short.sum()),
        "validation_short_files": int(validation_short.sum()),
        "train_native_length": train_diagnostics,
        "validation_native_length": validation_diagnostics,
        "long_file_equivalence": equivalence,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return arrays, {**metadata, "cache_status": "built"}


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
        "--native-cache-dir",
        type=Path,
        default=ROOT / "data" / "experiments" / "campp_short_audio_native",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "generated" / "campp_lme20_short_audio_native.json",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    pad_oofs, pad_artifacts, pad_metadata = load_fold_inputs(
        args.checkpoint_root, args.pad_cache_dir
    )
    candidates = []
    candidate_metadata = []
    for fold in range(NUM_FOLDS):
        arrays, metadata = build_or_load_native_fold_cache(
            fold=fold,
            pad_artifact=pad_artifacts[fold],
            pad_metadata=pad_metadata[fold],
            pad_oof=pad_oofs[fold],
            checkpoint_root=args.checkpoint_root,
            cache_dir=args.native_cache_dir,
            device=device,
        )
        candidates.append(arrays)
        candidate_metadata.append(metadata)

    fold_rows = []
    all_files = []
    all_labels = []
    all_pad_predictions = []
    all_native_predictions = []
    all_short_masks = []
    for fold in range(NUM_FOLDS):
        pad_oof = pad_oofs[fold]
        pad_artifact = pad_artifacts[fold]
        native = candidates[fold]
        native_artifact = {
            "train_embeddings": native["train_embeddings"],
            "competition_labels": native["competition_labels"],
            "unknown_cluster_ids": native["unknown_cluster_ids"],
        }
        pad_predictions = locked_lme_predictions(
            pad_artifact,
            pad_oof["embeddings"],
            pad_oof["competition_probs"],
        )
        native_predictions = locked_lme_predictions(
            native_artifact,
            native["validation_embeddings"],
            native["validation_probabilities"],
        )
        labels = pad_oof["labels"].astype(np.int64)
        baseline = metric_bundle(labels, pad_predictions)
        candidate = metric_bundle(labels, native_predictions)
        baseline_correct = pad_predictions == labels
        candidate_correct = native_predictions == labels
        fold_rows.append({
            "fold": fold,
            "baseline": baseline,
            "candidate": candidate,
            "delta": metric_delta(candidate, baseline),
            "rescued_errors": int(np.sum(~baseline_correct & candidate_correct)),
            "introduced_errors": int(np.sum(baseline_correct & ~candidate_correct)),
            "changed_predictions": int(np.sum(
                pad_predictions != native_predictions
            )),
            "short_audio": short_subset_summary(
                labels,
                pad_predictions,
                native_predictions,
                native["validation_short_mask"],
            ),
        })
        all_files.append(pad_oof["files"].astype(str))
        all_labels.append(labels)
        all_pad_predictions.append(pad_predictions)
        all_native_predictions.append(native_predictions)
        all_short_masks.append(native["validation_short_mask"].astype(bool))

    files = np.concatenate(all_files)
    if len(set(files.tolist())) != len(files):
        raise RuntimeError("OOF validation files overlap across folds")
    labels = np.concatenate(all_labels)
    pad_predictions = np.concatenate(all_pad_predictions)
    native_predictions = np.concatenate(all_native_predictions)
    short_mask = np.concatenate(all_short_masks)
    baseline = metric_bundle(labels, pad_predictions)
    candidate = metric_bundle(labels, native_predictions)
    aggregate = {
        "baseline": baseline,
        "candidate": candidate,
        "delta": metric_delta(candidate, baseline),
        "rescued_errors": int(np.sum(
            (pad_predictions != labels) & (native_predictions == labels)
        )),
        "introduced_errors": int(np.sum(
            (pad_predictions == labels) & (native_predictions != labels)
        )),
        "changed_predictions": int(np.sum(
            pad_predictions != native_predictions
        )),
        "short_audio": short_subset_summary(
            labels, pad_predictions, native_predictions, short_mask
        ),
    }
    if abs(baseline["macro_f1"] - LOCKED_BASELINE_MACRO_F1) > 1e-10:
        raise RuntimeError("Locked pad baseline reproduction failed")
    gate = acceptance_gate(fold_rows, aggregate)
    report = {
        "contract": {
            "hypothesis": (
                "use true waveform duration for audio shorter than eight seconds"
            ),
            "single_changed_variable": "short audio pad -> native length",
            "long_files": "copied bit-for-bit from locked pad artifacts",
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
            "native_cache_metadata": candidate_metadata,
            "unique_oof_files": int(len(files)),
            "oof_file_sha256": digest_names(files),
            "long_file_tolerance": LONG_EQUIVALENCE_ATOL,
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
