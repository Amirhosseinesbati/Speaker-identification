"""Leak-free three-Fold audit of frozen frontend evidence under LME20.

The primary candidate is deliberately fixed before inference: preserve the
locked CAM++ competition head, average CAM++ and frozen ERes2NetV2 LME20
prototype evidence 50/50, and retain every deployed decision parameter.  The
script also records two mechanism-only ERes diagnostics, but they cannot be
promoted when the primary candidate fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.special import softmax
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_control_oof_centroid_crossfit import (  # noqa: E402
    NUM_CLASSES,
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
    logmeanexp_group_scores,
)
from scripts.analyze_prototype_aggregation_crossfit import (  # noqa: E402
    group_indices,
)
from scripts.analyze_unknown_cluster_hypotheses_crossfit import (  # noqa: E402
    load_fold_inputs,
)
from src.data_pipeline import SpeakerDataset  # noqa: E402
from src.encoders import ERes2NetV2Encoder  # noqa: E402
from submission.inference import _collapse_centroid_probs  # noqa: E402


PRIMARY_VARIANT = "equal_campp_eres_prototype_evidence"
CACHE_SCHEMA = 1
AUDIO_KEYS = (
    "sample_rate",
    "duration_seconds",
    "min_valid_duration",
    "eval_hop_ratio",
    "max_eval_windows",
    "eval_speech_aware",
    "speech_relative_db",
    "short_audio_mode",
)


def digest_names(names: Iterable[str]) -> str:
    payload = "\n".join(sorted(map(str, names))).encode("utf-8")
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


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def encode_multiwindow(
    encoder: torch.nn.Module, windows: torch.Tensor
) -> torch.Tensor:
    """Average raw frontend embeddings over windows, then L2-normalise."""
    if windows.ndim != 4:
        raise RuntimeError(f"Expected (B,W,1,T), got {tuple(windows.shape)}")
    chunks = []
    for window_index in range(windows.shape[1]):
        hidden, _ = encoder(windows[:, window_index])
        if hidden.ndim != 3 or hidden.shape[1] != 1:
            raise RuntimeError(
                f"Unexpected frontend output shape {tuple(hidden.shape)}"
            )
        chunks.append(hidden[:, 0])
    return F.normalize(torch.stack(chunks, dim=0).mean(dim=0), p=2, dim=1)


def prototype_evidence(
    scores: np.ndarray, kappa: float
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64)
    internal = np.zeros((len(scores), 1 + scores.shape[1]), dtype=np.float64)
    internal[:, 1:] = softmax(float(kappa) * scores, axis=1)
    prototype = _collapse_centroid_probs(internal, NUM_CLASSES)
    return prototype, scores.max(axis=1).astype(np.float64)


def final_decision(
    head: np.ndarray,
    prototype: np.ndarray,
    max_score: np.ndarray,
    *,
    alpha: float,
    tau: float,
    unknown_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    head = np.asarray(head, dtype=np.float64)
    prototype = np.asarray(prototype, dtype=np.float64)
    max_score = np.asarray(max_score, dtype=np.float64)
    if head.shape != prototype.shape or max_score.shape != (len(head),):
        raise RuntimeError("Invalid decision evidence shapes")
    fused = float(alpha) * head + (1.0 - float(alpha)) * prototype
    fused[:, 0] *= float(unknown_weight)
    fused /= fused.sum(axis=1, keepdims=True) + 1e-12
    predictions = fused.argmax(axis=1).astype(np.int64)
    predictions[max_score < float(tau)] = 0
    return fused, predictions


def evidence_variants(
    *,
    campp_head: np.ndarray,
    campp_prototype: np.ndarray,
    campp_max_score: np.ndarray,
    frontend_prototype: np.ndarray,
    frontend_max_score: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return immutable evidence triples; no result-dependent selection."""
    return {
        "locked_campp_lme20": (
            campp_head,
            campp_prototype,
            campp_max_score,
        ),
        "eres_prototype_only": (
            frontend_prototype,
            frontend_prototype,
            frontend_max_score,
        ),
        "campp_head_eres_prototype": (
            campp_head,
            frontend_prototype,
            frontend_max_score,
        ),
        PRIMARY_VARIANT: (
            campp_head,
            0.5 * campp_prototype + 0.5 * frontend_prototype,
            0.5 * campp_max_score + 0.5 * frontend_max_score,
        ),
    }


def evaluate_gate(
    *,
    aggregate_delta: dict[str, float],
    fold_macro_deltas: list[float],
    rescued_errors: int,
    introduced_errors: int,
    baseline_errors: int,
    gate: dict,
) -> dict:
    rescue_rate = rescued_errors / max(baseline_errors, 1)
    checks = {
        "aggregate_macro_gain": aggregate_delta["macro_f1"]
        >= float(gate["minimum_aggregate_macro_gain"]),
        "aggregate_known_guardrail": aggregate_delta["known_accuracy"]
        >= -float(gate["maximum_aggregate_known_accuracy_drop"]),
        "aggregate_ood_guardrail": aggregate_delta["ood_f1"]
        >= -float(gate["maximum_aggregate_ood_f1_drop"]),
        "positive_fold_count": sum(value > 0.0 for value in fold_macro_deltas)
        >= int(gate["minimum_positive_folds"]),
        "worst_fold_guardrail": min(fold_macro_deltas)
        >= float(gate["minimum_fold_macro_delta"]),
        "rescue_rate": rescue_rate
        >= float(gate["minimum_baseline_error_rescue_rate"]),
        "rescued_outnumber_introduced": (
            rescued_errors > introduced_errors
            if gate.get("require_more_rescued_than_introduced", True)
            else True
        ),
    }
    return {
        "checks": checks,
        "passed": bool(all(checks.values())),
        "rescue_rate": float(rescue_rate),
        "rescued_errors": int(rescued_errors),
        "introduced_errors": int(introduced_errors),
        "baseline_errors": int(baseline_errors),
    }


def control_audio_contract(checkpoint_root: Path) -> tuple[dict, list[dict]]:
    contracts = []
    provenance = []
    for fold in range(NUM_FOLDS):
        profile = f"p0-campp-known446-ood-control-oof-f{fold}"
        checkpoint_path = checkpoint_root / profile / "campp_best_raw.pt"
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        config = checkpoint["config"]
        audio = config["audio"]
        contract = {key: audio.get(key) for key in AUDIO_KEYS}
        contract["audio_dir"] = config["data"]["audio_dir"]
        contracts.append(contract)
        provenance.append({
            "fold": fold,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "audio_contract": contract,
        })
    if any(contract != contracts[0] for contract in contracts[1:]):
        raise RuntimeError("Control Fold audio/window contracts differ")
    return contracts[0], provenance


def make_dataset(files: np.ndarray, audio_contract: dict) -> SpeakerDataset:
    frame = pd.DataFrame({
        "audio_file": files.astype(str),
        "label": np.zeros(len(files), dtype=np.int64),
    })
    return SpeakerDataset(
        frame,
        str(resolve_repo_path(audio_contract["audio_dir"])),
        sample_rate=int(audio_contract["sample_rate"]),
        duration_seconds=float(audio_contract["duration_seconds"]),
        augment=False,
        min_valid_duration=float(audio_contract["min_valid_duration"] or 1.0),
        num_train_windows=1,
        eval_hop_ratio=float(audio_contract["eval_hop_ratio"] or 0.5),
        max_eval_windows=int(audio_contract["max_eval_windows"] or 8),
        eval_speech_aware=bool(audio_contract["eval_speech_aware"]),
        speech_relative_db=float(audio_contract["speech_relative_db"] or 35.0),
        short_audio_mode=str(audio_contract["short_audio_mode"] or "pad"),
    )


def build_or_load_global_cache(
    *,
    files: np.ndarray,
    audio_contract: dict,
    analysis_config: dict,
    config_sha256: str,
    cache_path: Path,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict]:
    encoder_cfg = analysis_config["encoder"]
    checkpoint_path = resolve_repo_path(encoder_cfg["checkpoint"])
    if sha256_file(checkpoint_path) != encoder_cfg["checkpoint_sha256"]:
        raise RuntimeError("ERes2NetV2 checkpoint hash mismatch")
    metadata_path = cache_path.with_suffix(".json")
    contract = {
        "schema": CACHE_SCHEMA,
        "analysis_config_sha256": config_sha256,
        "encoder_checkpoint": str(checkpoint_path),
        "encoder_checkpoint_sha256": encoder_cfg["checkpoint_sha256"],
        "file_count": int(len(files)),
        "file_set_sha256": digest_names(files.astype(str)),
        "audio_contract": audio_contract,
    }
    if cache_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("contract") == contract
            and metadata.get("cache_sha256") == sha256_file(cache_path)
        ):
            with np.load(cache_path) as data:
                arrays = {key: data[key].copy() for key in data.files}
            if (
                np.array_equal(arrays["files"].astype(str), files.astype(str))
                and arrays["embeddings"].shape
                == (len(files), int(encoder_cfg["embedding_dim"]))
                and np.all(np.isfinite(arrays["embeddings"]))
            ):
                return arrays, metadata

    encoder = ERes2NetV2Encoder(
        local_path=str(resolve_repo_path(encoder_cfg["local_path"])),
        allow_hub_download=False,
        freeze_encoder=True,
    ).to(device).eval()
    dataset = make_dataset(files, audio_contract)
    runtime = analysis_config["runtime"]
    loader = DataLoader(
        dataset,
        batch_size=int(runtime["batch_size"]),
        shuffle=False,
        num_workers=int(runtime["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    chunks = []
    with torch.inference_mode():
        for windows, _ in tqdm(loader, desc="Frozen ERes2NetV2 embeddings"):
            chunks.append(
                encode_multiwindow(
                    encoder, windows.to(device, non_blocking=True)
                ).cpu().numpy()
            )
    embeddings = np.concatenate(chunks, axis=0).astype(np.float32)
    if embeddings.shape != (len(files), int(encoder_cfg["embedding_dim"])):
        raise RuntimeError(f"Unexpected embedding shape {embeddings.shape}")
    if not np.all(np.isfinite(embeddings)):
        raise RuntimeError("Frozen frontend embeddings contain non-finite values")
    arrays = {"files": files.astype(str), "embeddings": embeddings}
    atomic_savez(cache_path, arrays)
    metadata = {
        "contract": contract,
        "cache_sha256": sha256_file(cache_path),
        "embedding_norm_min": float(np.linalg.norm(embeddings, axis=1).min()),
        "embedding_norm_max": float(np.linalg.norm(embeddings, axis=1).max()),
    }
    atomic_write_json(metadata_path, metadata)
    return arrays, metadata


def aggregate_variant(
    folds: list[dict], variant: str
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    files = np.concatenate([row["files"] for row in folds])
    if len(set(files.astype(str).tolist())) != len(files):
        raise RuntimeError("OOF files overlap across Folds")
    labels = np.concatenate([row["labels"] for row in folds])
    predictions = np.concatenate([row["predictions"][variant] for row in folds])
    baseline_predictions = np.concatenate([
        row["predictions"]["locked_campp_lme20"] for row in folds
    ])
    return metric_bundle(labels, predictions), files, labels, baseline_predictions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-config", type=Path,
        default=ROOT / "configs" / "analyses"
        / "frozen-eres2netv2-lme20-threefold.json",
    )
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "checkpoints")
    parser.add_argument(
        "--cache-dir", type=Path,
        default=ROOT / "data" / "experiments" / "frozen_eres2netv2_lme20",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "reports" / "generated"
        / "frozen_eres2netv2_lme20_threefold.json",
    )
    args = parser.parse_args()

    config = json.loads(args.analysis_config.read_text(encoding="utf-8"))
    config_sha = sha256_file(args.analysis_config)
    if config["locked_decision"]["primary_variant"] != PRIMARY_VARIANT:
        raise RuntimeError("Primary variant contract mismatch")
    runtime = config["runtime"]
    device = torch.device(runtime["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the preregistered analysis")

    baseline_cache_dir = resolve_repo_path(config["data"]["baseline_cache_dir"])
    oofs, artifacts, baseline_metadata = load_fold_inputs(
        args.checkpoint_root, baseline_cache_dir
    )
    expected_count = int(config["data"]["expected_unique_oof_files"])
    all_files = np.asarray(sorted({
        name for oof in oofs for name in oof["files"].astype(str).tolist()
    }), dtype=str)
    if len(all_files) != expected_count:
        raise RuntimeError(f"Expected {expected_count} OOF files, got {len(all_files)}")

    audio_contract, checkpoint_provenance = control_audio_contract(
        args.checkpoint_root
    )
    cache_path = args.cache_dir / "frozen_eres2netv2_embeddings.npz"
    cache, cache_metadata = build_or_load_global_cache(
        files=all_files,
        audio_contract=audio_contract,
        analysis_config=config,
        config_sha256=config_sha,
        cache_path=cache_path,
        device=device,
    )
    file_index = {
        name: index for index, name in enumerate(cache["files"].astype(str))
    }

    decision = config["locked_decision"]
    beta = float(decision["lme_beta"])
    kappa = float(decision["prototype_kappa"])
    alpha = float(decision["alpha_head"])
    tau = float(decision["ood_tau"])
    unknown_weight = float(decision["unknown_weight"])
    fold_rows = []
    for fold in range(NUM_FOLDS):
        oof = oofs[fold]
        artifact = artifacts[fold]
        train_names = artifact["train_files"].astype(str)
        validation_names = oof["files"].astype(str)
        if any(name not in file_index for name in train_names):
            raise RuntimeError(f"Fold {fold} train file missing from global cache")
        train_frontend = cache["embeddings"][[file_index[name] for name in train_names]]
        validation_frontend = cache["embeddings"][[
            file_index[name] for name in validation_names
        ]]
        groups = group_indices(artifact)
        campp_scores = logmeanexp_group_scores(
            oof["embeddings"], artifact["train_embeddings"], groups, beta=beta
        )
        frontend_scores = logmeanexp_group_scores(
            validation_frontend, train_frontend, groups, beta=beta
        )
        campp_prototype, campp_max = prototype_evidence(campp_scores, kappa)
        frontend_prototype, frontend_max = prototype_evidence(
            frontend_scores, kappa
        )
        evidence = evidence_variants(
            campp_head=oof["competition_probs"],
            campp_prototype=campp_prototype,
            campp_max_score=campp_max,
            frontend_prototype=frontend_prototype,
            frontend_max_score=frontend_max,
        )
        predictions = {}
        metrics = {}
        for name, (head, prototype, max_score) in evidence.items():
            _, prediction = final_decision(
                head, prototype, max_score,
                alpha=alpha, tau=tau, unknown_weight=unknown_weight,
            )
            predictions[name] = prediction
            metrics[name] = metric_bundle(oof["labels"], prediction)
        baseline = metrics["locked_campp_lme20"]
        fold_rows.append({
            "fold": fold,
            "files": validation_names,
            "labels": oof["labels"].astype(np.int64),
            "predictions": predictions,
            "metrics": metrics,
            "delta_vs_baseline": {
                name: metric_delta(value, baseline)
                for name, value in metrics.items() if name != "locked_campp_lme20"
            },
            "score_diagnostics": {
                "campp_max_mean": float(campp_max.mean()),
                "frontend_max_mean": float(frontend_max.mean()),
            },
        })

    aggregate = {}
    aggregate_predictions = {}
    all_files_order = all_labels = baseline_predictions = None
    for variant in fold_rows[0]["predictions"]:
        metrics, files, labels, baseline_pred = aggregate_variant(fold_rows, variant)
        aggregate[variant] = metrics
        aggregate_predictions[variant] = np.concatenate([
            row["predictions"][variant] for row in fold_rows
        ])
        all_files_order, all_labels, baseline_predictions = files, labels, baseline_pred
    assert all_files_order is not None and all_labels is not None
    assert baseline_predictions is not None
    locked_macro = float(config["baseline"]["oof_macro_f1"])
    actual_macro = aggregate["locked_campp_lme20"]["macro_f1"]
    if abs(actual_macro - locked_macro) > 1e-12:
        raise RuntimeError(
            f"Locked LME20 reproduction mismatch: {actual_macro} != {locked_macro}"
        )

    primary_predictions = aggregate_predictions[PRIMARY_VARIANT]
    baseline_correct = all_labels == baseline_predictions
    primary_correct = all_labels == primary_predictions
    rescued = int(np.sum(~baseline_correct & primary_correct))
    introduced = int(np.sum(baseline_correct & ~primary_correct))
    baseline_errors = int(np.sum(~baseline_correct))
    primary_delta = metric_delta(
        aggregate[PRIMARY_VARIANT], aggregate["locked_campp_lme20"]
    )
    fold_macro_deltas = [
        row["delta_vs_baseline"][PRIMARY_VARIANT]["macro_f1"]
        for row in fold_rows
    ]
    gate = evaluate_gate(
        aggregate_delta=primary_delta,
        fold_macro_deltas=fold_macro_deltas,
        rescued_errors=rescued,
        introduced_errors=introduced,
        baseline_errors=baseline_errors,
        gate=config["gate"],
    )

    oof_path = args.cache_dir / "frozen_eres2netv2_lme20_oof.npz"
    oof_arrays = {
        "files": all_files_order.astype(str),
        "labels": all_labels.astype(np.int64),
        "baseline_predictions": baseline_predictions.astype(np.int64),
    }
    for variant, values in aggregate_predictions.items():
        oof_arrays[f"predictions__{variant}"] = values.astype(np.int64)
    atomic_savez(oof_path, oof_arrays)

    report = {
        "contract": config,
        "provenance": {
            "analysis_config": str(args.analysis_config),
            "analysis_config_sha256": config_sha,
            "control_checkpoints": checkpoint_provenance,
            "baseline_cache_metadata": baseline_metadata,
            "frontend_cache": cache_metadata,
            "oof_file_count": int(len(all_files_order)),
            "oof_file_set_sha256": digest_names(all_files_order.astype(str)),
            "leaderboard_used_for_selection": False,
        },
        "folds": [{
            key: value for key, value in row.items()
            if key not in {"files", "labels", "predictions"}
        } for row in fold_rows],
        "aggregate": {
            "metrics": aggregate,
            "delta_vs_baseline": {
                name: metric_delta(value, aggregate["locked_campp_lme20"])
                for name, value in aggregate.items()
                if name != "locked_campp_lme20"
            },
            "primary_variant": PRIMARY_VARIANT,
            "primary_gate": gate,
        },
        "decision": (
            "eligible_for_packaging_analysis"
            if gate["passed"] else "reject_frozen_eres2netv2_and_advance_backlog"
        ),
    }
    atomic_write_json(args.output, report)

    receipt_path = args.cache_dir / "frozen_eres2netv2_lme20_receipt.json"
    artifacts_for_receipt = [
        args.analysis_config,
        resolve_repo_path(config["encoder"]["checkpoint"]),
        cache_path,
        cache_path.with_suffix(".json"),
        oof_path,
        args.output,
    ]
    receipt = {
        "analysis_id": config["analysis_id"],
        "artifacts": [{
            "path": str(path),
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        } for path in artifacts_for_receipt],
        "decision": report["decision"],
        "gate_passed": bool(gate["passed"]),
    }
    atomic_write_json(receipt_path, receipt)
    print(json.dumps({
        "output": str(args.output),
        "receipt": str(receipt_path),
        "baseline": aggregate["locked_campp_lme20"],
        "primary": aggregate[PRIMARY_VARIANT],
        "delta": primary_delta,
        "gate": gate,
        "decision": report["decision"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
