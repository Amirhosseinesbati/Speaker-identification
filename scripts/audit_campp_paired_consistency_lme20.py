"""Terminal Fold-0 audit for the preregistered long80 paired-consistency A/B.

The evaluator performs no parameter search.  It compares selected Raw LME20
decisions for the matched 80-epoch control and fixed-weight consistency branch,
then evaluates the one preregistered 50/50 probability-evidence fusion with the
externally validated CAM++ Control Fold 0.
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
EXTERNAL_CONTROL_LME20_MACRO_F1 = 0.9611456662793696
MINIMUM_MATCHED_MACRO_GAIN = 0.002
MINIMUM_FUSION_MACRO_GAIN = 0.002
MAXIMUM_KNOWN_DROP = 0.001
MAXIMUM_OOD_DROP = 0.001
MINIMUM_RESCUE_RATE = 0.20
MINIMUM_EMBEDDING_SPREAD_RATIO = 0.95


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
) -> None:
    """Require identical long80 science except the enabled consistency term."""
    matched = _normalise_identity(matched_config)
    treatment = _normalise_identity(treatment_config)
    for name, config in (("matched", matched), ("treatment", treatment)):
        training = config.get("training", {}) or {}
        if int(training.get("epochs", -1)) != 80:
            raise RuntimeError(f"{name} is not an 80-epoch recipe")
        if list(training.get("milestone_epochs", [])) != [40]:
            raise RuntimeError(f"{name} does not preserve epoch 40")
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
            "Long80 branches differ outside consistency.enabled: "
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


def milestone_diagnostic(path: Path, expected_profile: str) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if int(checkpoint.get("epoch", -1)) != 40:
        raise RuntimeError(f"Milestone is not epoch 40: {path}")
    config = checkpoint.get("config", {}) or {}
    checkpoint_dir = str((config.get("logging", {}) or {}).get("checkpoint_dir", ""))
    if not checkpoint_dir.replace("\\", "/").endswith(expected_profile):
        raise RuntimeError(f"Milestone profile mismatch: {path}")
    history = checkpoint.get("training_history", checkpoint.get("history", []))
    if not isinstance(history, list) or not history:
        raise RuntimeError(f"Milestone history missing: {path}")
    row = history[-1]
    if int(row.get("epoch", -1)) != 40:
        raise RuntimeError(f"Milestone history does not end at epoch 40: {path}")
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
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "epoch": 40,
        "metrics": metrics,
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
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT / "reports" / "generated"
            / "campp_paired_consistency_long80_fold0.json"
        ),
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

    profiles = (MATCHED_CONTROL_PROFILE, TREATMENT_PROFILE)
    locked_config_hashes = {
        MATCHED_CONTROL_PROFILE: MATCHED_CONFIG_SHA256,
        TREATMENT_PROFILE: TREATMENT_CONFIG_SHA256,
    }
    for profile, expected_sha256 in locked_config_hashes.items():
        config_path = ROOT / "configs" / "experiments" / f"{profile}.yaml"
        if sha256_file(config_path) != expected_sha256:
            raise RuntimeError(f"Locked long80 config SHA changed: {profile}")
    loaded: dict[str, dict] = {}
    for profile in profiles:
        checkpoint_dir = args.checkpoint_root / profile
        checkpoint_path = checkpoint_dir / "campp_best_raw.pt"
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
        milestone_path = (
            checkpoint_dir / "campp_milestone_epoch040_raw.pt"
        )
        loaded[profile] = {
            "checkpoint_path": checkpoint_path,
            "oof_path": oof_path,
            "binding": binding,
            "oof": oof,
            "artifact": artifact,
            "artifact_metadata": metadata,
            "checkpoint": checkpoint,
            "milestone": milestone_diagnostic(milestone_path, profile),
        }

    matched = loaded[MATCHED_CONTROL_PROFILE]
    treatment = loaded[TREATMENT_PROFILE]
    if matched["checkpoint"]["class_map"] != treatment["checkpoint"]["class_map"]:
        raise RuntimeError("Long80 class maps differ")
    if (
        matched["checkpoint"]["config"]["data"]["split"]
        != treatment["checkpoint"]["config"]["data"]["split"]
    ):
        raise RuntimeError("Long80 split provenance differs")
    assert_paired_single_objective_contract(
        matched["checkpoint"]["config"], treatment["checkpoint"]["config"]
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
    matched_spread = embedding_spread(matched["artifact"])
    treatment_spread = embedding_spread(treatment["artifact"])
    spread_ratio = float(treatment_spread / matched_spread)
    gate = acceptance_gate(
        matched_delta=matched_delta,
        fusion_delta=fusion_delta,
        rescue_rate=float(treatment_changes["rescue_rate"]),
        spread_ratio=spread_ratio,
    )

    report = {
        "contract": {
            "scope": "Fold 0 long80 matched A/B",
            "matched_control_profile": MATCHED_CONTROL_PROFILE,
            "treatment_profile": TREATMENT_PROFILE,
            "single_training_treatment": "fixed cosine consistency weight 0.1",
            "horizon": 80,
            "diagnostic_milestone": 40,
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
                "oof_sha256": sha256_file(matched["oof_path"]),
                "bundle_binding": matched["binding"],
                "train_artifact": matched["artifact_metadata"],
                "milestone": matched["milestone"],
            },
            "treatment": {
                "checkpoint_sha256": sha256_file(treatment["checkpoint_path"]),
                "oof_sha256": sha256_file(treatment["oof_path"]),
                "bundle_binding": treatment["binding"],
                "train_artifact": treatment["artifact_metadata"],
                "milestone": treatment["milestone"],
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
        "collapse_guard": {
            "statistic": "mean coordinate std of deterministic train embeddings",
            "matched_control": matched_spread,
            "treatment": treatment_spread,
            "treatment_over_matched": spread_ratio,
        },
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
        "external_control": external_metrics,
        "matched_control": matched_metrics,
        "treatment": report["treatment"],
        "fixed_fusion": report["fixed_fusion"],
        "collapse_guard": report["collapse_guard"],
        "gate": gate,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
