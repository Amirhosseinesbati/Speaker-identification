"""Fixed three-fold Raw/EMA snapshot ensemble audit for CAM++ LME20.

The externally validated baseline uses the selected Raw CAM++ checkpoint from
each fold, the fixed LME20 prototype backend, and the locked decision policy.
This audit changes one variable: the Raw head/prototype/max-score evidence is
averaged 50/50 with evidence extracted from the matching best EMA snapshot.

There is no parameter search.  The target OOF fold never selects a weight,
threshold, epoch, or checkpoint.  The EMA enrollment and validation evidence
is extracted with the production batching paths and cached with complete
checkpoint, input-artifact, file-order, and payload hashes.
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
from scipy.special import softmax
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_control_oof_centroid_crossfit import (  # noqa: E402
    NUM_CLASSES,
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
    logmeanexp_group_scores,
)
from scripts.analyze_prototype_aggregation_crossfit import (  # noqa: E402
    group_indices,
)
from scripts.analyze_unknown_cluster_hypotheses_crossfit import (  # noqa: E402
    load_fold_inputs,
)
from scripts.audit_short_audio_repeat import (  # noqa: E402
    LOCKED_BASELINE_MACRO_F1,
    LOCKED_LME_BETA,
    acceptance_gate,
    atomic_savez,
    digest_names,
    make_eval_dataset,
)
from src.model_factory import create_model_from_config  # noqa: E402
from submission.inference import _collapse_centroid_probs  # noqa: E402


SNAPSHOT_WEIGHT = 0.5
CACHE_SCHEMA = 1
EQUIVALENCE_ATOL = 1e-8


def probability_evidence(
    artifact: dict[str, np.ndarray],
    validation_embeddings: np.ndarray,
    head_probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return locked head, prototype probability, and raw max-score evidence."""
    groups = group_indices(artifact)
    scores = logmeanexp_group_scores(
        validation_embeddings,
        artifact["train_embeddings"],
        groups,
        beta=LOCKED_LME_BETA,
    )
    internal = np.zeros((len(scores), 1 + scores.shape[1]), dtype=np.float64)
    internal[:, 1:] = softmax(float(LOCKED_RAW_KAPPA) * scores, axis=1)
    prototype = _collapse_centroid_probs(internal, NUM_CLASSES)
    head = np.asarray(head_probabilities, dtype=np.float64)
    if head.shape != prototype.shape:
        raise RuntimeError(
            f"Head/prototype shape mismatch: {head.shape} != {prototype.shape}"
        )
    return head, prototype, scores.max(axis=1).astype(np.float64)


def final_decision(
    head: np.ndarray,
    prototype: np.ndarray,
    max_score: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the immutable LME20 head/prototype fusion and hard OOD gate."""
    head = np.asarray(head, dtype=np.float64)
    prototype = np.asarray(prototype, dtype=np.float64)
    max_score = np.asarray(max_score, dtype=np.float64)
    if head.shape != prototype.shape or max_score.shape != (len(head),):
        raise RuntimeError("Invalid snapshot decision evidence shapes")
    fused = LOCKED_ALPHA * head + (1.0 - LOCKED_ALPHA) * prototype
    fused[:, 0] *= LOCKED_UNKNOWN_WEIGHT
    fused /= fused.sum(axis=1, keepdims=True) + 1e-12
    predictions = fused.argmax(axis=1).astype(np.int64)
    predictions[max_score < LOCKED_TAU] = 0
    return fused, predictions


def fixed_raw_ema_decision(
    raw: tuple[np.ndarray, np.ndarray, np.ndarray],
    ema: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Average Raw/EMA evidence with the preregistered equal weight."""
    for raw_item, ema_item in zip(raw, ema):
        if raw_item.shape != ema_item.shape:
            raise RuntimeError(
                f"Raw/EMA evidence shape mismatch: {raw_item.shape} != {ema_item.shape}"
            )
    head = SNAPSHOT_WEIGHT * raw[0] + SNAPSHOT_WEIGHT * ema[0]
    prototype = SNAPSHOT_WEIGHT * raw[1] + SNAPSHOT_WEIGHT * ema[1]
    max_score = SNAPSHOT_WEIGHT * raw[2] + SNAPSHOT_WEIGHT * ema[2]
    return final_decision(head, prototype, max_score)


def validate_cache(
    arrays: dict[str, np.ndarray], raw_artifact: dict, raw_oof: dict
) -> None:
    expected = {
        "train_files": (len(raw_artifact["train_files"]),),
        "train_embeddings": raw_artifact["train_embeddings"].shape,
        "competition_labels": (len(raw_artifact["train_files"]),),
        "unknown_cluster_ids": (len(raw_artifact["train_files"]),),
        "validation_files": (len(raw_oof["files"]),),
        "validation_labels": (len(raw_oof["files"]),),
        "validation_probabilities": raw_oof["competition_probs"].shape,
        "validation_embeddings": raw_oof["embeddings"].shape,
    }
    for key, shape in expected.items():
        if key not in arrays or arrays[key].shape != shape:
            actual = None if key not in arrays else arrays[key].shape
            raise RuntimeError(f"{key} shape {actual} != {shape}")
    if not np.array_equal(
        arrays["train_files"].astype(str), raw_artifact["train_files"].astype(str)
    ):
        raise RuntimeError("EMA enrollment order differs from Raw artifact")
    if not np.array_equal(
        arrays["validation_files"].astype(str), raw_oof["files"].astype(str)
    ):
        raise RuntimeError("EMA validation order differs from Raw OOF")
    if not np.array_equal(
        arrays["competition_labels"].astype(np.int64),
        raw_artifact["competition_labels"].astype(np.int64),
    ):
        raise RuntimeError("EMA competition labels differ from Raw artifact")
    if not np.array_equal(
        arrays["unknown_cluster_ids"].astype(np.int64),
        raw_artifact["unknown_cluster_ids"].astype(np.int64),
    ):
        raise RuntimeError("EMA unknown-cluster ids differ from Raw artifact")
    for key in (
        "train_embeddings",
        "validation_probabilities",
        "validation_embeddings",
    ):
        if not np.all(np.isfinite(arrays[key])):
            raise RuntimeError(f"{key} contains non-finite values")


def extract_outputs(
    *,
    model: torch.nn.Module,
    train_dataset,
    validation_dataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    fold: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use the exact enrollment and OOF batching paths from production."""
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
        for windows, _ in tqdm(train_loader, desc=f"Fold {fold} EMA enrollment"):
            train_chunks.append(
                model.embed(windows.to(device, non_blocking=True)).cpu().numpy()
            )
    train_embeddings = l2norm_rows(np.concatenate(train_chunks, axis=0))

    probability_chunks: list[np.ndarray] = []
    embedding_chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for windows, _ in tqdm(
            validation_loader, desc=f"Fold {fold} EMA validation"
        ):
            windows = windows.to(device, non_blocking=True)
            for file_windows in windows:
                probability, embedding = model.predict_proba_and_embed(
                    file_windows, temperature=1.0
                )
                probability_chunks.append(probability.cpu().numpy())
                embedding_chunks.append(embedding.cpu().numpy())
    return (
        train_embeddings.astype(np.float32),
        np.stack(probability_chunks).astype(np.float32),
        l2norm_rows(np.stack(embedding_chunks)).astype(np.float32),
    )


def build_or_load_ema_cache(
    *,
    fold: int,
    raw_artifact: dict[str, np.ndarray],
    raw_metadata: dict,
    raw_oof: dict[str, np.ndarray],
    checkpoint_root: Path,
    cache_dir: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[dict[str, np.ndarray], dict]:
    profile = f"p0-campp-known446-ood-control-oof-f{fold}"
    raw_checkpoint_path = checkpoint_root / profile / "campp_best_raw.pt"
    ema_checkpoint_path = checkpoint_root / profile / "campp_best_ema.pt"
    raw_oof_path = (
        checkpoint_root / profile / "campp_best_bundle" / "oof_predictions.npz"
    )
    cache_path = cache_dir / f"fold{fold}_ema_inference.npz"
    metadata_path = cache_path.with_suffix(".json")
    train_files = raw_artifact["train_files"].astype(str)
    validation_files = raw_oof["files"].astype(str)
    expected = {
        "schema_version": CACHE_SCHEMA,
        "fold": int(fold),
        "variant": "best_ema",
        "raw_checkpoint_sha256": sha256_file(raw_checkpoint_path),
        "ema_checkpoint_sha256": sha256_file(ema_checkpoint_path),
        "raw_artifact_sha256": str(raw_metadata["artifact_sha256"]),
        "raw_oof_sha256": sha256_file(raw_oof_path),
        "train_file_sha256": digest_names(train_files),
        "validation_file_sha256": digest_names(validation_files),
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
            validate_cache(arrays, raw_artifact, raw_oof)
            return arrays, {**metadata, "cache_status": "reused"}

    raw_checkpoint = torch.load(
        raw_checkpoint_path, map_location="cpu", weights_only=False
    )
    ema_checkpoint = torch.load(
        ema_checkpoint_path, map_location="cpu", weights_only=False
    )
    if raw_checkpoint["config"] != ema_checkpoint["config"]:
        raise RuntimeError(f"Fold {fold} Raw/EMA configs differ")
    if raw_checkpoint["class_map"] != ema_checkpoint["class_map"]:
        raise RuntimeError(f"Fold {fold} Raw/EMA class maps differ")
    config = deepcopy(ema_checkpoint["config"])
    class_map = ema_checkpoint["class_map"]
    model = create_model_from_config(
        config, num_known_speakers=len(class_map) - 1
    )
    model.load_state_dict(ema_checkpoint["model_state_dict"])
    model.to(device).eval()

    train_labels = raw_artifact["competition_labels"].astype(np.int64)
    validation_labels = raw_oof["labels"].astype(np.int64)
    train_dataset = make_eval_dataset(train_files, train_labels, config, "pad")
    validation_dataset = make_eval_dataset(
        validation_files, validation_labels, config, "pad"
    )
    train_embeddings, validation_probabilities, validation_embeddings = (
        extract_outputs(
            model=model,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            fold=fold,
        )
    )
    arrays = {
        "train_files": train_files,
        "train_embeddings": train_embeddings,
        "competition_labels": train_labels,
        "unknown_cluster_ids": raw_artifact["unknown_cluster_ids"].astype(
            np.int64
        ),
        "validation_files": validation_files,
        "validation_labels": validation_labels,
        "validation_probabilities": validation_probabilities,
        "validation_embeddings": validation_embeddings,
    }
    validate_cache(arrays, raw_artifact, raw_oof)
    atomic_savez(cache_path, arrays)
    metadata = {
        **expected,
        "artifact_sha256": sha256_file(cache_path),
        "train_files": int(len(train_files)),
        "validation_files": int(len(validation_files)),
        "embedding_batching": {
            "enrollment": "model.embed window-major production cache path",
            "validation": "per-file predict_proba_and_embed production path",
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return arrays, {**metadata, "cache_status": "built"}


def decision_summary(
    labels: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, int]:
    baseline_correct = baseline == labels
    candidate_correct = candidate == labels
    return {
        "rescued_errors": int(np.sum(~baseline_correct & candidate_correct)),
        "introduced_errors": int(np.sum(baseline_correct & ~candidate_correct)),
        "changed_predictions": int(np.sum(baseline != candidate)),
        "both_wrong": int(np.sum(~baseline_correct & ~candidate_correct)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-root", type=Path, default=ROOT / "checkpoints"
    )
    parser.add_argument(
        "--raw-cache-dir",
        type=Path,
        default=ROOT / "data" / "experiments" / "campp_control_centroid_crossfit",
    )
    parser.add_argument(
        "--ema-cache-dir",
        type=Path,
        default=ROOT / "data" / "experiments" / "campp_raw_ema_lme20",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "generated" / "campp_raw_ema_lme20.json",
    )
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--num-workers", type=int, default=8)
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

    raw_oofs, raw_artifacts, raw_metadata = load_fold_inputs(
        args.checkpoint_root, args.raw_cache_dir
    )
    ema_arrays = []
    ema_metadata = []
    for fold in range(NUM_FOLDS):
        arrays, metadata = build_or_load_ema_cache(
            fold=fold,
            raw_artifact=raw_artifacts[fold],
            raw_metadata=raw_metadata[fold],
            raw_oof=raw_oofs[fold],
            checkpoint_root=args.checkpoint_root,
            cache_dir=args.ema_cache_dir,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        ema_arrays.append(arrays)
        ema_metadata.append(metadata)

    fold_rows = []
    all_files = []
    all_labels = []
    all_raw_predictions = []
    all_ema_predictions = []
    all_ensemble_predictions = []
    for fold in range(NUM_FOLDS):
        raw_oof = raw_oofs[fold]
        raw_artifact = raw_artifacts[fold]
        ema = ema_arrays[fold]
        ema_artifact = {
            "train_embeddings": ema["train_embeddings"],
            "competition_labels": ema["competition_labels"],
            "unknown_cluster_ids": ema["unknown_cluster_ids"],
        }
        raw_evidence = probability_evidence(
            raw_artifact,
            raw_oof["embeddings"],
            raw_oof["competition_probs"],
        )
        ema_evidence = probability_evidence(
            ema_artifact,
            ema["validation_embeddings"],
            ema["validation_probabilities"],
        )
        raw_probabilities, raw_predictions = final_decision(*raw_evidence)
        ema_probabilities, ema_predictions = final_decision(*ema_evidence)
        ensemble_probabilities, ensemble_predictions = fixed_raw_ema_decision(
            raw_evidence, ema_evidence
        )
        # A self-ensemble must be bitwise decision-equivalent to the baseline.
        self_probabilities, self_predictions = fixed_raw_ema_decision(
            raw_evidence, raw_evidence
        )
        if not np.array_equal(self_predictions, raw_predictions):
            raise RuntimeError(f"Fold {fold} self-ensemble prediction mismatch")
        if float(np.max(np.abs(self_probabilities - raw_probabilities))) > EQUIVALENCE_ATOL:
            raise RuntimeError(f"Fold {fold} self-ensemble probability mismatch")

        labels = raw_oof["labels"].astype(np.int64)
        raw_metrics = metric_bundle(labels, raw_predictions)
        ema_metrics = metric_bundle(labels, ema_predictions)
        ensemble_metrics = metric_bundle(labels, ensemble_predictions)
        fold_rows.append({
            "fold": fold,
            "raw": raw_metrics,
            "ema_diagnostic": ema_metrics,
            "ensemble": ensemble_metrics,
            "delta": metric_delta(ensemble_metrics, raw_metrics),
            "ema_delta": metric_delta(ema_metrics, raw_metrics),
            **decision_summary(labels, raw_predictions, ensemble_predictions),
            "raw_ema_probability_max_abs_diff": float(np.max(np.abs(
                raw_probabilities - ema_probabilities
            ))),
            "raw_ema_prediction_disagreements": int(np.sum(
                raw_predictions != ema_predictions
            )),
        })
        all_files.append(raw_oof["files"].astype(str))
        all_labels.append(labels)
        all_raw_predictions.append(raw_predictions)
        all_ema_predictions.append(ema_predictions)
        all_ensemble_predictions.append(ensemble_predictions)

    files = np.concatenate(all_files)
    if len(set(files.tolist())) != len(files):
        raise RuntimeError("OOF validation files overlap across folds")
    labels = np.concatenate(all_labels)
    raw_predictions = np.concatenate(all_raw_predictions)
    ema_predictions = np.concatenate(all_ema_predictions)
    ensemble_predictions = np.concatenate(all_ensemble_predictions)
    raw_metrics = metric_bundle(labels, raw_predictions)
    ema_metrics = metric_bundle(labels, ema_predictions)
    ensemble_metrics = metric_bundle(labels, ensemble_predictions)
    if abs(raw_metrics["macro_f1"] - LOCKED_BASELINE_MACRO_F1) > 1e-10:
        raise RuntimeError(
            "Locked Raw LME20 baseline mismatch: "
            f"{raw_metrics['macro_f1']:.16f} != {LOCKED_BASELINE_MACRO_F1:.16f}"
        )
    aggregate = {
        "raw": raw_metrics,
        "ema_diagnostic": ema_metrics,
        "ensemble": ensemble_metrics,
        "delta": metric_delta(ensemble_metrics, raw_metrics),
        "ema_delta": metric_delta(ema_metrics, raw_metrics),
        **decision_summary(labels, raw_predictions, ensemble_predictions),
        "raw_ema_prediction_disagreements": int(np.sum(
            raw_predictions != ema_predictions
        )),
    }
    gate = acceptance_gate(fold_rows, aggregate)
    report = {
        "contract": {
            "hypothesis": (
                "the best EMA snapshot has complementary errors to the selected "
                "Raw snapshot despite lower standalone validation Macro-F1"
            ),
            "single_changed_variable": (
                "Raw-only head/prototype/max-score evidence -> fixed 50/50 "
                "Raw/EMA evidence average"
            ),
            "snapshot_weight": SNAPSHOT_WEIGHT,
            "weights_selected": False,
            "epochs_selected": False,
            "folds": "fixed kfold/folds3/seed42 OOF",
            "lme_beta": LOCKED_LME_BETA,
            "alpha": LOCKED_ALPHA,
            "kappa": LOCKED_RAW_KAPPA,
            "tau": LOCKED_TAU,
            "lambda_unknown": LOCKED_UNKNOWN_WEIGHT,
            "leaderboard_tuning": False,
            "latest_snapshot": "provenance diagnostic only; not evaluated",
        },
        "provenance": {
            "raw_cache_metadata": raw_metadata,
            "ema_cache_metadata": ema_metadata,
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
