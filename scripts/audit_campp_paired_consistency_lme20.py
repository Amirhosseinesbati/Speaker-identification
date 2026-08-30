"""Terminal Fold-0 audit for a preregistered paired-consistency A/B horizon.

The evaluator performs no parameter search.  It compares selected Raw LME20
decisions for the matched control and fixed-weight consistency branch, then
evaluates the one preregistered 50/50 probability-evidence fusion with the
externally validated CAM++ Control Fold 0.  Both the historical 80-epoch pair
and the superseding 120-epoch pair have immutable config hashes.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_control_oof_centroid_crossfit import (  # noqa: E402
    NUM_CLASSES,
    build_or_load_train_artifact,
    load_oof,
    metric_bundle,
    metric_delta,
    rebuild_exact_splits,
    sha256_file,
)
from scripts.analyze_unknown_cluster_hypotheses_crossfit import (  # noqa: E402
    load_fold_inputs,
)
from scripts.audit_campp_channelrobust_lme20 import (  # noqa: E402
    align_oof,
    error_changes,
    validate_raw_bundle_binding,
)
from scripts.audit_raw_ema_lme20 import (  # noqa: E402
    final_decision,
    fixed_raw_ema_decision,
    probability_evidence,
)
from scripts.audit_short_audio_repeat import digest_names  # noqa: E402


EXTERNAL_CONTROL_PROFILE = "p0-campp-known446-ood-control-oof-f0"
MATCHED_CONTROL_PROFILE = (
    "p4-campp-known446-ood-channelrobust-paired-control-long80-oof-f0"
)
TREATMENT_PROFILE = (
    "p4-campp-known446-ood-channelrobust-consistency-c01-long80-oof-f0"
)
SOURCE_PROFILE = "p3-campp-known446-ood-channelrobust-oof-f0"
SOURCE_CHECKPOINT_SHA256 = (
    "a46715e603173201a35bf20d9b43f6ad27f0352561b4c834ce7a2b3a3ae67a06"
)
MATCHED_CONFIG_SHA256 = (
    "29fad79221ef180cdd7eb35102dc75cd488e8505f4de0d8e44eb20d3cd144562"
)
TREATMENT_CONFIG_SHA256 = (
    "1d0625d1c4311dbe0544775cfeb8db91c2f07d0335eba748a7c87240ef4ba860"
)
LONG120_MATCHED_CONTROL_PROFILE = (
    "p4-campp-known446-ood-channelrobust-paired-control-long120-oof-f0"
)
LONG120_TREATMENT_PROFILE = (
    "p4-campp-known446-ood-channelrobust-consistency-c01-long120-oof-f0"
)
LONG120_MATCHED_CONFIG_SHA256 = (
    "88eed2d8f3ab1a4e37f72ae1955ded78887e84332308fc965c66b777cae0b5e1"
)
LONG120_TREATMENT_CONFIG_SHA256 = (
    "823891d4aa396b02d21563efc487acbe71f3bcff84572b96eb8a2d1554826f77"
)
HORIZON_SPECS = {
    80: {
        "matched_profile": MATCHED_CONTROL_PROFILE,
        "treatment_profile": TREATMENT_PROFILE,
        "matched_config_sha256": MATCHED_CONFIG_SHA256,
        "treatment_config_sha256": TREATMENT_CONFIG_SHA256,
        "milestones": (40,),
    },
    120: {
        "matched_profile": LONG120_MATCHED_CONTROL_PROFILE,
        "treatment_profile": LONG120_TREATMENT_PROFILE,
        "matched_config_sha256": LONG120_MATCHED_CONFIG_SHA256,
        "treatment_config_sha256": LONG120_TREATMENT_CONFIG_SHA256,
        "milestones": (40, 80),
    },
}
EXTERNAL_CONTROL_LME20_MACRO_F1 = 0.9611456662793696
MINIMUM_MATCHED_MACRO_GAIN = 0.002
MINIMUM_FUSION_MACRO_GAIN = 0.002
MAXIMUM_KNOWN_DROP = 0.001
MAXIMUM_OOD_DROP = 0.001
MINIMUM_RESCUE_RATE = 0.20
MINIMUM_EMBEDDING_SPREAD_RATIO = 0.95
TAIL_WINDOW_EPOCHS = 10
MINIMUM_TAIL_MEAN_GAIN = 0.0005
MINIMUM_RELATIVE_GAP_GAIN = 0.0005
MAXIMUM_TAIL_GUARDRAIL_DROP = 0.002
RANDOMIZATION_REPLICATES = 20_000
RANDOMIZATION_SEED = 20_260_830


def _normalise_identity(config: dict) -> dict:
    result = copy.deepcopy(config)
    result.pop("_meta", None)
    result.pop("experiment", None)
    result.pop("logging", None)
    result.pop("mlops", None)
    hardware = result.get("hardware", {}) or {}
    for profile in (hardware.get("profiles", {}) or {}).values():
        if isinstance(profile, dict):
            profile.pop("description", None)
    return result


def assert_paired_single_objective_contract(
    matched_config: dict,
    treatment_config: dict,
    *,
    expected_epochs: int = 80,
    expected_milestones: tuple[int, ...] = (40,),
) -> None:
    """Require identical paired science except the enabled consistency term."""
    matched = _normalise_identity(matched_config)
    treatment = _normalise_identity(treatment_config)
    for name, config in (("matched", matched), ("treatment", treatment)):
        training = config.get("training", {}) or {}
        if int(training.get("epochs", -1)) != expected_epochs:
            raise RuntimeError(
                f"{name} is not an {expected_epochs}-epoch recipe"
            )
        if tuple(training.get("milestone_epochs", [])) != expected_milestones:
            raise RuntimeError(
                f"{name} does not preserve milestones {expected_milestones}"
            )
        if int(training.get("early_stopping_patience", -1)) != 0:
            raise RuntimeError(f"{name} enables metric early stopping")
        if str(training.get("selection_variant", "")).lower() != "raw":
            raise RuntimeError(f"{name} does not lock canonical Raw selection")
        expected_source = (
            f"checkpoints/{SOURCE_PROFILE}/campp_best_raw.pt"
        )
        configured_source = str(training.get("warm_start_checkpoint", ""))
        if not configured_source.replace("\\", "/").endswith(expected_source):
            raise RuntimeError(f"{name} warm start is not the locked source Raw")

    matched_consistency = matched["training"]["loss"]["consistency"]
    treatment_consistency = treatment["training"]["loss"]["consistency"]
    if matched_consistency != {
        "enabled": False,
        "type": "cosine",
        "weight": 0.1,
    }:
        raise RuntimeError("Unexpected matched-control consistency config")
    if treatment_consistency != {
        "enabled": True,
        "type": "cosine",
        "weight": 0.1,
    }:
        raise RuntimeError("Unexpected treatment consistency config")
    treatment_consistency["enabled"] = False
    if treatment != matched:
        changed = sorted(
            key for key in set(matched) | set(treatment)
            if matched.get(key) != treatment.get(key)
        )
        raise RuntimeError(
            "Paired branches differ outside consistency.enabled: "
            + ", ".join(changed)
        )


def embedding_spread(artifact: dict[str, np.ndarray]) -> float:
    """Mean coordinate standard deviation over deterministic train embeddings."""
    embeddings = np.asarray(artifact["train_embeddings"], dtype=np.float64)
    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        raise RuntimeError("Unexpected train embedding matrix")
    spread = float(np.std(embeddings, axis=0, ddof=0).mean())
    if not np.isfinite(spread) or spread <= 0.0:
        raise RuntimeError(f"Non-finite or collapsed embedding spread: {spread}")
    return spread


def _fast_macro_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Competition Macro-F1 without repeated sklearn allocation.

    The fixed full label set is retained, including zero-valued F1 for absent
    classes, so this is numerically equivalent to the canonical metric helper.
    """
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if labels.ndim != 1 or predictions.shape != labels.shape:
        raise ValueError("labels and predictions must be aligned 1-D arrays")
    if not (
        np.all((0 <= labels) & (labels < NUM_CLASSES))
        and np.all((0 <= predictions) & (predictions < NUM_CLASSES))
    ):
        raise ValueError("labels and predictions must use competition class ids")
    true_count = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float64)
    pred_count = np.bincount(predictions, minlength=NUM_CLASSES).astype(np.float64)
    true_positive = np.bincount(
        labels[labels == predictions], minlength=NUM_CLASSES
    ).astype(np.float64)
    denominator = true_count + pred_count
    per_class = np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros(NUM_CLASSES, dtype=np.float64),
        where=denominator > 0.0,
    )
    return float(per_class.mean())


def paired_randomization_diagnostic(
    labels: np.ndarray,
    baseline_predictions: np.ndarray,
    candidate_predictions: np.ndarray,
    *,
    replicates: int = RANDOMIZATION_REPLICATES,
    seed: int = RANDOMIZATION_SEED,
) -> dict:
    """Fixed-seed paired prediction-swap test for the primary Macro-F1 delta.

    This quantifies how unusual the observed paired delta is under prediction
    exchangeability.  It is deliberately descriptive: it cannot override the
    preregistered effect-size and Known/OOD guardrails.
    """
    labels = np.asarray(labels, dtype=np.int64)
    baseline = np.asarray(baseline_predictions, dtype=np.int64)
    candidate = np.asarray(candidate_predictions, dtype=np.int64)
    if (
        labels.ndim != 1
        or baseline.shape != labels.shape
        or candidate.shape != labels.shape
    ):
        raise ValueError("paired randomization inputs must be aligned 1-D arrays")
    if int(replicates) <= 0:
        raise ValueError("replicates must be positive")

    observed = _fast_macro_f1(labels, candidate) - _fast_macro_f1(labels, baseline)
    rng = np.random.default_rng(int(seed))
    null_deltas = np.empty(int(replicates), dtype=np.float64)
    for index in range(int(replicates)):
        swap = rng.integers(0, 2, size=len(labels), dtype=np.int8).astype(bool)
        permuted_candidate = np.where(swap, baseline, candidate)
        permuted_baseline = np.where(swap, candidate, baseline)
        null_deltas[index] = (
            _fast_macro_f1(labels, permuted_candidate)
            - _fast_macro_f1(labels, permuted_baseline)
        )

    tolerance = 1e-15
    one_sided = (
        1.0 + float(np.sum(null_deltas >= observed - tolerance))
    ) / (float(replicates) + 1.0)
    two_sided = (
        1.0 + float(np.sum(np.abs(null_deltas) >= abs(observed) - tolerance))
    ) / (float(replicates) + 1.0)
    baseline_correct = baseline == labels
    candidate_correct = candidate == labels
    return {
        "method": "paired Monte Carlo prediction-swap randomization",
        "primary_metric": "447-class Macro-F1",
        "observed_delta": float(observed),
        "replicates": int(replicates),
        "seed": int(seed),
        "one_sided_improvement_p_value": float(one_sided),
        "two_sided_p_value": float(two_sided),
        "null_delta_quantiles": {
            "q025": float(np.quantile(null_deltas, 0.025)),
            "q500": float(np.quantile(null_deltas, 0.500)),
            "q975": float(np.quantile(null_deltas, 0.975)),
        },
        "prediction_disagreements": int(np.sum(baseline != candidate)),
        "candidate_only_correct": int(np.sum(candidate_correct & ~baseline_correct)),
        "baseline_only_correct": int(np.sum(baseline_correct & ~candidate_correct)),
        "decision_role": "descriptive_only_cannot_override_locked_gate",
    }


def milestone_diagnostic(
    path: Path,
    expected_profile: str,
    expected_epoch: int = 40,
) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if int(checkpoint.get("epoch", -1)) != expected_epoch:
        raise RuntimeError(f"Milestone is not epoch {expected_epoch}: {path}")
    config = checkpoint.get("config", {}) or {}
    checkpoint_dir = str((config.get("logging", {}) or {}).get("checkpoint_dir", ""))
    if not checkpoint_dir.replace("\\", "/").endswith(expected_profile):
        raise RuntimeError(f"Milestone profile mismatch: {path}")
    history = checkpoint.get("training_history", checkpoint.get("history", []))
    if not isinstance(history, list) or not history:
        raise RuntimeError(f"Milestone history missing: {path}")
    epochs = np.asarray(
        [int(item.get("epoch", -1)) for item in history], dtype=np.int64,
    )
    expected_epochs = np.arange(1, expected_epoch + 1, dtype=np.int64)
    if not np.array_equal(epochs, expected_epochs):
        raise RuntimeError(
            f"Milestone history is not contiguous 1..{expected_epoch}: {path}"
        )
    row = history[-1]
    if int(row.get("epoch", -1)) != expected_epoch:
        raise RuntimeError(
            f"Milestone history does not end at epoch {expected_epoch}: {path}"
        )
    keys = (
        "val_macro_f1",
        "val_logit_avg_macro_f1",
        "val_known_acc",
        "val_ood_f1",
        "val_ema_macro_f1",
        "train_loss",
        "train_loss_consistency",
        "train_loss_consistency_weighted",
        "train_pair_cosine",
        "train_embedding_std_augmented",
        "train_embedding_std_clean",
    )
    metrics = {
        key: (float(row[key]) if row.get(key) is not None else None)
        for key in keys
    }
    finite = [value for value in metrics.values() if value is not None]
    if not all(np.isfinite(value) for value in finite):
        raise RuntimeError(f"Milestone contains non-finite metrics: {path}")

    trajectory_keys = (
        "val_macro_f1",
        "val_logit_avg_macro_f1",
        "val_ema_macro_f1",
        "val_known_acc",
        "val_ood_f1",
        "train_loss",
        "val_loss",
    )
    curves = {
        key: np.asarray([float(item[key]) for item in history], dtype=np.float64)
        for key in trajectory_keys
    }
    if not all(np.isfinite(values).all() for values in curves.values()):
        raise RuntimeError(
            f"Milestone history contains non-finite trajectory metrics: {path}"
        )
    window = min(TAIL_WINDOW_EPOCHS, expected_epoch // 2)
    if window < 1:
        raise RuntimeError(f"Milestone history is too short for windows: {path}")
    previous_slice = slice(-2 * window, -window)
    tail_slice = slice(-window, None)
    previous_means = {
        key: float(values[previous_slice].mean())
        for key, values in curves.items()
    }
    tail_means = {
        key: float(values[tail_slice].mean())
        for key, values in curves.items()
    }
    slope_width = min(2 * TAIL_WINDOW_EPOCHS, expected_epoch)
    slope_x = np.arange(slope_width, dtype=np.float64)
    slopes = {
        key: float(np.polyfit(slope_x, values[-slope_width:], 1)[0])
        for key, values in curves.items()
    }
    best_index = int(np.argmax(curves["val_macro_f1"]))
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "epoch": expected_epoch,
        "metrics": metrics,
        "history_length": len(history),
        "history_contiguous": True,
        "trajectory": {
            "window_epochs": window,
            "previous_window": [
                expected_epoch - 2 * window + 1,
                expected_epoch - window,
            ],
            "tail_window": [expected_epoch - window + 1, expected_epoch],
            "previous_means": previous_means,
            "tail_means": tail_means,
            "tail_minus_previous": {
                key: float(tail_means[key] - previous_means[key])
                for key in trajectory_keys
            },
            "slopes_last_20": slopes,
            "best_raw_epoch": int(epochs[best_index]),
            "best_raw_macro_f1": float(curves["val_macro_f1"][best_index]),
            "best_raw_known_accuracy": float(
                curves["val_known_acc"][best_index]
            ),
            "best_raw_ood_f1": float(curves["val_ood_f1"][best_index]),
        },
    }


def terminal_curve_diagnostic(
    path: Path,
    expected_profile: str,
    *,
    expected_epoch: int = 80,
) -> dict:
    """Validate the terminal checkpoint and summarize its locked tail windows."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if int(checkpoint.get("epoch", -1)) != expected_epoch:
        raise RuntimeError(f"Latest checkpoint is not epoch {expected_epoch}: {path}")
    config = checkpoint.get("config", {}) or {}
    checkpoint_dir = str((config.get("logging", {}) or {}).get("checkpoint_dir", ""))
    if not checkpoint_dir.replace("\\", "/").endswith(expected_profile):
        raise RuntimeError(f"Latest checkpoint profile mismatch: {path}")
    history = checkpoint.get("training_history", checkpoint.get("history", []))
    if not isinstance(history, list) or len(history) < 2 * TAIL_WINDOW_EPOCHS:
        raise RuntimeError(f"Terminal history is too short: {path}")
    epochs = np.asarray([int(row.get("epoch", -1)) for row in history], dtype=np.int64)
    expected = np.arange(1, expected_epoch + 1, dtype=np.int64)
    if not np.array_equal(epochs, expected):
        raise RuntimeError(f"Terminal history is not contiguous 1..{expected_epoch}: {path}")

    keys = ("val_macro_f1", "val_known_acc", "val_ood_f1")
    curves = {
        key: np.asarray([float(row[key]) for row in history], dtype=np.float64)
        for key in keys
    }
    if not all(np.isfinite(values).all() for values in curves.values()):
        raise RuntimeError(f"Terminal history contains non-finite metrics: {path}")

    previous_slice = slice(-2 * TAIL_WINDOW_EPOCHS, -TAIL_WINDOW_EPOCHS)
    tail_slice = slice(-TAIL_WINDOW_EPOCHS, None)
    previous_means = {
        key: float(values[previous_slice].mean()) for key, values in curves.items()
    }
    tail_means = {
        key: float(values[tail_slice].mean()) for key, values in curves.items()
    }
    tail_x = np.arange(2 * TAIL_WINDOW_EPOCHS, dtype=np.float64)
    macro_tail = curves["val_macro_f1"][-2 * TAIL_WINDOW_EPOCHS :]
    best_index = int(np.argmax(curves["val_macro_f1"]))
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "terminal_epoch": expected_epoch,
        "window_epochs": TAIL_WINDOW_EPOCHS,
        "previous_window": [expected_epoch - 2 * TAIL_WINDOW_EPOCHS + 1,
                            expected_epoch - TAIL_WINDOW_EPOCHS],
        "tail_window": [expected_epoch - TAIL_WINDOW_EPOCHS + 1, expected_epoch],
        "previous_means": previous_means,
        "tail_means": tail_means,
        "tail_minus_previous": {
            key: float(tail_means[key] - previous_means[key]) for key in keys
        },
        "macro_slope_last_20": float(np.polyfit(tail_x, macro_tail, 1)[0]),
        "best_raw_epoch": int(epochs[best_index]),
        "best_raw_macro_f1": float(curves["val_macro_f1"][best_index]),
    }


def matched_extension_diagnostic(
    matched_curve: dict,
    treatment_curve: dict,
    *,
    spread_ratio: float,
) -> dict:
    """Predeclared evidence gate for a separate, still-matched extension.

    This does not accept the treatment or authorize an asymmetric continuation.
    It only identifies the specific under-training pattern that warrants a new
    preregistered matched extension after the configured paired horizon.
    """
    matched_gap = (
        matched_curve["tail_means"]["val_macro_f1"]
        - matched_curve["previous_means"]["val_macro_f1"]
    )
    treatment_gap = (
        treatment_curve["tail_means"]["val_macro_f1"]
        - treatment_curve["previous_means"]["val_macro_f1"]
    )
    relative_gap_gain = float(treatment_gap - matched_gap)
    checks = {
        "treatment_tail_mean_gain": (
            treatment_gap >= MINIMUM_TAIL_MEAN_GAIN
        ),
        "treatment_tail_positive_slope": (
            treatment_curve["macro_slope_last_20"] > 0.0
        ),
        "treatment_best_in_final_window": (
            treatment_curve["best_raw_epoch"]
            > treatment_curve["terminal_epoch"] - TAIL_WINDOW_EPOCHS
        ),
        "relative_gap_is_still_improving": (
            relative_gap_gain >= MINIMUM_RELATIVE_GAP_GAIN
        ),
        "tail_known_guardrail": (
            treatment_curve["tail_minus_previous"]["val_known_acc"]
            >= -MAXIMUM_TAIL_GUARDRAIL_DROP
        ),
        "tail_ood_guardrail": (
            treatment_curve["tail_minus_previous"]["val_ood_f1"]
            >= -MAXIMUM_TAIL_GUARDRAIL_DROP
        ),
        "embedding_spread_guardrail": (
            spread_ratio >= MINIMUM_EMBEDDING_SPREAD_RATIO
        ),
    }
    return {
        "eligible_for_separate_matched_extension": bool(all(checks.values())),
        "checks": checks,
        "matched_tail_mean_gain": float(matched_gap),
        "treatment_tail_mean_gain": float(treatment_gap),
        "relative_gap_gain": relative_gap_gain,
        "thresholds": {
            "tail_window_epochs": TAIL_WINDOW_EPOCHS,
            "minimum_treatment_tail_mean_gain": MINIMUM_TAIL_MEAN_GAIN,
            "minimum_relative_gap_gain": MINIMUM_RELATIVE_GAP_GAIN,
            "maximum_tail_known_or_ood_drop": MAXIMUM_TAIL_GUARDRAIL_DROP,
            "minimum_embedding_spread_ratio": MINIMUM_EMBEDDING_SPREAD_RATIO,
        },
        "effect": (
            "preregister a new matched extension; never continue treatment alone"
        ),
    }


def acceptance_gate(
    *,
    matched_delta: dict[str, float],
    fusion_delta: dict[str, float],
    rescue_rate: float,
    spread_ratio: float,
) -> dict:
    checks = {
        "matched_lme20_macro_gain": (
            matched_delta["macro_f1"] >= MINIMUM_MATCHED_MACRO_GAIN
        ),
        "fixed_fusion_macro_gain": (
            fusion_delta["macro_f1"] >= MINIMUM_FUSION_MACRO_GAIN
        ),
        "fixed_fusion_known_guardrail": (
            fusion_delta["known_accuracy"] >= -MAXIMUM_KNOWN_DROP
        ),
        "fixed_fusion_ood_guardrail": (
            fusion_delta["ood_f1"] >= -MAXIMUM_OOD_DROP
        ),
        "external_control_rescue_rate": rescue_rate >= MINIMUM_RESCUE_RATE,
        "embedding_spread_ratio": (
            spread_ratio >= MINIMUM_EMBEDDING_SPREAD_RATIO
        ),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "thresholds": {
            "minimum_matched_lme20_macro_gain": MINIMUM_MATCHED_MACRO_GAIN,
            "minimum_fixed_fusion_macro_gain": MINIMUM_FUSION_MACRO_GAIN,
            "maximum_known_accuracy_drop": MAXIMUM_KNOWN_DROP,
            "maximum_ood_f1_drop": MAXIMUM_OOD_DROP,
            "minimum_external_control_rescue_rate": MINIMUM_RESCUE_RATE,
            "minimum_embedding_spread_ratio": MINIMUM_EMBEDDING_SPREAD_RATIO,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "checkpoints")
    parser.add_argument(
        "--control-cache-dir",
        type=Path,
        default=ROOT / "data" / "experiments" / "campp_control_centroid_crossfit",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "data" / "experiments" / "campp_paired_consistency_lme20",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=ROOT / "data" / "processed" / "audio_wav_labels.csv",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "audio_wav",
    )
    parser.add_argument(
        "--cluster-map",
        type=Path,
        default=ROOT / "data" / "processed" / "unknown_clusters_oof_f0.json",
    )
    parser.add_argument("--horizon", type=int, choices=sorted(HORIZON_SPECS), default=120)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    spec = HORIZON_SPECS[args.horizon]
    matched_profile = str(spec["matched_profile"])
    treatment_profile = str(spec["treatment_profile"])
    milestones = tuple(int(value) for value in spec["milestones"])
    if args.output is None:
        args.output = (
            ROOT / "reports" / "generated"
            / f"campp_paired_consistency_long{args.horizon}_fold0.json"
        )

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    external_oofs, external_artifacts, external_metadata = load_fold_inputs(
        args.checkpoint_root, args.control_cache_dir
    )
    external_oof = external_oofs[0]
    external_artifact = external_artifacts[0]
    splits, cleaning = rebuild_exact_splits(args.labels, args.audio_dir)
    train_frame, validation_frame = splits[0]
    expected_validation = set(validation_frame["audio_file"].astype(str))

    profiles = (matched_profile, treatment_profile)
    locked_config_hashes = {
        matched_profile: str(spec["matched_config_sha256"]),
        treatment_profile: str(spec["treatment_config_sha256"]),
    }
    for profile, expected_sha256 in locked_config_hashes.items():
        config_path = ROOT / "configs" / "experiments" / f"{profile}.yaml"
        if sha256_file(config_path) != expected_sha256:
            raise RuntimeError(
                f"Locked long{args.horizon} config SHA changed: {profile}"
            )
    loaded: dict[str, dict] = {}
    for profile in profiles:
        checkpoint_dir = args.checkpoint_root / profile
        checkpoint_path = checkpoint_dir / "campp_best_raw.pt"
        latest_path = checkpoint_dir / "campp_latest.pt"
        oof_path = checkpoint_dir / "campp_best_bundle" / "oof_predictions.npz"
        binding = validate_raw_bundle_binding(
            checkpoint_dir, checkpoint_path, oof_path
        )
        oof = align_oof(
            external_oof,
            load_oof(oof_path, 0, expected_validation),
        )
        artifact, metadata = build_or_load_train_artifact(
            fold=0,
            train_frame=train_frame,
            checkpoint_path=checkpoint_path,
            cluster_map_path=args.cluster_map,
            cache_path=args.cache_dir / profile / "fold0_train_embeddings_centroids.npz",
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        milestone_receipts = {
            str(epoch): milestone_diagnostic(
                checkpoint_dir / f"campp_milestone_epoch{epoch:03d}_raw.pt",
                profile,
                epoch,
            )
            for epoch in milestones
        }
        loaded[profile] = {
            "checkpoint_path": checkpoint_path,
            "latest_path": latest_path,
            "oof_path": oof_path,
            "binding": binding,
            "oof": oof,
            "artifact": artifact,
            "artifact_metadata": metadata,
            "checkpoint": checkpoint,
            "milestones": milestone_receipts,
            "terminal_curve": terminal_curve_diagnostic(
                latest_path, profile, expected_epoch=args.horizon
            ),
        }

    matched = loaded[matched_profile]
    treatment = loaded[treatment_profile]
    if matched["checkpoint"]["class_map"] != treatment["checkpoint"]["class_map"]:
        raise RuntimeError(f"Long{args.horizon} class maps differ")
    if (
        matched["checkpoint"]["config"]["data"]["split"]
        != treatment["checkpoint"]["config"]["data"]["split"]
    ):
        raise RuntimeError(f"Long{args.horizon} split provenance differs")
    assert_paired_single_objective_contract(
        matched["checkpoint"]["config"],
        treatment["checkpoint"]["config"],
        expected_epochs=args.horizon,
        expected_milestones=milestones,
    )
    source_checkpoint = (
        args.checkpoint_root / SOURCE_PROFILE / "campp_best_raw.pt"
    )
    if sha256_file(source_checkpoint) != SOURCE_CHECKPOINT_SHA256:
        raise RuntimeError("Locked Channel-Robust source checkpoint SHA changed")

    external_evidence = probability_evidence(
        external_artifact,
        external_oof["embeddings"],
        external_oof["competition_probs"],
    )
    matched_evidence = probability_evidence(
        matched["artifact"],
        matched["oof"]["embeddings"],
        matched["oof"]["competition_probs"],
    )
    treatment_evidence = probability_evidence(
        treatment["artifact"],
        treatment["oof"]["embeddings"],
        treatment["oof"]["competition_probs"],
    )
    _, external_predictions = final_decision(*external_evidence)
    _, matched_predictions = final_decision(*matched_evidence)
    _, treatment_predictions = final_decision(*treatment_evidence)
    _, fusion_predictions = fixed_raw_ema_decision(
        external_evidence, treatment_evidence
    )
    labels = external_oof["labels"].astype(np.int64)
    external_metrics = metric_bundle(labels, external_predictions)
    if abs(
        external_metrics["macro_f1"] - EXTERNAL_CONTROL_LME20_MACRO_F1
    ) > 1e-10:
        raise RuntimeError("Locked external Control LME20 reproduction failed")
    matched_metrics = metric_bundle(labels, matched_predictions)
    treatment_metrics = metric_bundle(labels, treatment_predictions)
    fusion_metrics = metric_bundle(labels, fusion_predictions)
    matched_delta = metric_delta(treatment_metrics, matched_metrics)
    external_delta = metric_delta(treatment_metrics, external_metrics)
    fusion_delta = metric_delta(fusion_metrics, external_metrics)
    treatment_changes = error_changes(
        labels, external_predictions, treatment_predictions
    )
    fusion_changes = error_changes(labels, external_predictions, fusion_predictions)
    randomization_diagnostics = {
        "treatment_vs_matched_control": paired_randomization_diagnostic(
            labels, matched_predictions, treatment_predictions
        ),
        "fixed_fusion_vs_external_control": paired_randomization_diagnostic(
            labels, external_predictions, fusion_predictions
        ),
    }
    matched_spread = embedding_spread(matched["artifact"])
    treatment_spread = embedding_spread(treatment["artifact"])
    spread_ratio = float(treatment_spread / matched_spread)
    extension_diagnostic = matched_extension_diagnostic(
        matched["terminal_curve"],
        treatment["terminal_curve"],
        spread_ratio=spread_ratio,
    )
    gate = acceptance_gate(
        matched_delta=matched_delta,
        fusion_delta=fusion_delta,
        rescue_rate=float(treatment_changes["rescue_rate"]),
        spread_ratio=spread_ratio,
    )

    report = {
        "contract": {
            "scope": f"Fold 0 long{args.horizon} matched A/B",
            "matched_control_profile": matched_profile,
            "treatment_profile": treatment_profile,
            "single_training_treatment": "fixed cosine consistency weight 0.1",
            "horizon": args.horizon,
            "diagnostic_milestones": list(milestones),
            "decision": "selected Raw probability-average LME20 direct argmax",
            "fusion": "fixed 50/50 probability-evidence average",
            "parameter_search": False,
            "leaderboard_tuning": False,
        },
        "provenance": {
            "cleaning": cleaning,
            "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
            "config_sha256": locked_config_hashes,
            "external_control_artifact": external_metadata[0],
            "validation_files": int(len(labels)),
            "validation_file_sha256": digest_names(
                external_oof["files"].astype(str)
            ),
            "matched": {
                "checkpoint_sha256": sha256_file(matched["checkpoint_path"]),
                "latest_sha256": sha256_file(matched["latest_path"]),
                "oof_sha256": sha256_file(matched["oof_path"]),
                "bundle_binding": matched["binding"],
                "train_artifact": matched["artifact_metadata"],
                "milestones": matched["milestones"],
                "terminal_curve": matched["terminal_curve"],
            },
            "treatment": {
                "checkpoint_sha256": sha256_file(treatment["checkpoint_path"]),
                "latest_sha256": sha256_file(treatment["latest_path"]),
                "oof_sha256": sha256_file(treatment["oof_path"]),
                "bundle_binding": treatment["binding"],
                "train_artifact": treatment["artifact_metadata"],
                "milestones": treatment["milestones"],
                "terminal_curve": treatment["terminal_curve"],
            },
        },
        "external_control": external_metrics,
        "matched_control": matched_metrics,
        "treatment": {
            "metrics": treatment_metrics,
            "delta_vs_matched": matched_delta,
            "delta_vs_external_control": external_delta,
            "error_changes_vs_external_control": treatment_changes,
        },
        "fixed_fusion": {
            "metrics": fusion_metrics,
            "delta_vs_external_control": fusion_delta,
            "error_changes_vs_external_control": fusion_changes,
        },
        "paired_randomization_diagnostics": randomization_diagnostics,
        "collapse_guard": {
            "statistic": "mean coordinate std of deterministic train embeddings",
            "matched_control": matched_spread,
            "treatment": treatment_spread,
            "treatment_over_matched": spread_ratio,
        },
        "acceptance_gate": gate,
        "matched_extension_diagnostic": extension_diagnostic,
        "decision": "accept" if gate["passed"] else "reject",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({
        "output": str(args.output),
        "decision": report["decision"],
        "external_control": external_metrics,
        "matched_control": matched_metrics,
        "treatment": report["treatment"],
        "fixed_fusion": report["fixed_fusion"],
        "paired_randomization_diagnostics": randomization_diagnostics,
        "collapse_guard": report["collapse_guard"],
        "gate": gate,
        "matched_extension_diagnostic": extension_diagnostic,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
