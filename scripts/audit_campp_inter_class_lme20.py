"""Terminal Fold-0 audit for the preregistered P6 inter-class A/B pair.

The evaluator performs no search.  It reproduces the immutable external CAM++
LME20 reference, the historical terminal P5 pair, and the same-commit P6
control/treatment pair.  It then applies the preregistered score, guardrail,
rescue, representation-spread, and class-weight-energy gates.
"""

from __future__ import annotations

import argparse
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
from scripts.audit_campp_paired_consistency_lme20 import (  # noqa: E402
    EXTERNAL_CONTROL_LME20_MACRO_F1,
    P5_CROSS_FILE_MATCHED_CONFIG_SHA256,
    P5_CROSS_FILE_MATCHED_PROFILE,
    P5_CROSS_FILE_TREATMENT_CONFIG_SHA256,
    P5_CROSS_FILE_TREATMENT_PROFILE,
    SOURCE_CHECKPOINT_SHA256,
    SOURCE_PROFILE,
    _normalise_identity,
    embedding_spread,
    paired_randomization_diagnostic,
    terminal_curve_diagnostic,
)
from scripts.audit_inter_class_angular_energy import audit_pair  # noqa: E402
from scripts.audit_raw_ema_lme20 import (  # noqa: E402
    final_decision,
    fixed_raw_ema_decision,
    probability_evidence,
)
from scripts.audit_short_audio_repeat import digest_names  # noqa: E402


P6_CONTROL_PROFILE = (
    "p6-campp-known446-ood-crossfile-consistency-interclass-control-es80p20-oof-f0"
)
P6_TREATMENT_PROFILE = (
    "p6-campp-known446-ood-crossfile-consistency-interclass-e01-es80p20-oof-f0"
)
P6_CONTROL_CONFIG_SHA256 = (
    "1f041abb3035af1f39de1714f389a6b9aadb04fb0a0c34e737c5ca363d92b3ab"
)
P6_TREATMENT_CONFIG_SHA256 = (
    "25dd1f1a52e4908c47e66b234de8e58671b82cb78fa528453a3865490f01ba5c"
)
PROFILE_HASHES = {
    P5_CROSS_FILE_MATCHED_PROFILE: P5_CROSS_FILE_MATCHED_CONFIG_SHA256,
    P5_CROSS_FILE_TREATMENT_PROFILE: P5_CROSS_FILE_TREATMENT_CONFIG_SHA256,
    P6_CONTROL_PROFILE: P6_CONTROL_CONFIG_SHA256,
    P6_TREATMENT_PROFILE: P6_TREATMENT_CONFIG_SHA256,
}
MINIMUM_MACRO_GAIN = 0.002
MAXIMUM_GUARDRAIL_DROP = 0.001
MINIMUM_RESCUE_RATE = 0.20
MINIMUM_EMBEDDING_SPREAD_RATIO = 0.95
MAXIMUM_ENERGY_RATIO = 0.95
P6_MAXIMUM_EPOCHS = 120
P6_EARLY_STOPPING_START_EPOCH = 80
P6_EARLY_STOPPING_PATIENCE = 20


def assert_p6_single_objective_contract(
    p5_treatment_config: dict,
    p6_control_config: dict,
    p6_treatment_config: dict,
) -> None:
    """Require source matching and exactly one P6 scientific difference."""
    p5_treatment = _normalise_identity(p5_treatment_config)
    p6_control = _normalise_identity(p6_control_config)
    p6_treatment = _normalise_identity(p6_treatment_config)
    for name, candidate in (
        ("control", p6_control), ("treatment", p6_treatment)
    ):
        training = candidate["training"]
        if int(training.get("epochs", -1)) != P6_MAXIMUM_EPOCHS:
            raise RuntimeError(f"P6 {name} maximum epoch changed")
        if int(training.get("early_stopping_start_epoch", -1)) != (
            P6_EARLY_STOPPING_START_EPOCH
        ):
            raise RuntimeError(f"P6 {name} delayed start changed")
        if int(training.get("early_stopping_patience", -1)) != (
            P6_EARLY_STOPPING_PATIENCE
        ):
            raise RuntimeError(f"P6 {name} patience changed")

    expected_control = {
        "enabled": False,
        "type": "exclusive_angular_energy",
        "weight": 0.01,
    }
    expected_treatment = {**expected_control, "enabled": True}
    control_inter = p6_control["training"]["loss"]["speaker"]["inter_class"]
    treatment_inter = p6_treatment["training"]["loss"]["speaker"][
        "inter_class"
    ]
    if control_inter != expected_control:
        raise RuntimeError("Unexpected P6 control inter-class configuration")
    if treatment_inter != expected_treatment:
        raise RuntimeError("Unexpected P6 treatment inter-class configuration")
    treatment_inter["enabled"] = False
    if p6_treatment != p6_control:
        raise RuntimeError(
            "P6 branches differ outside inter_class.enabled"
        )

    # The shared delayed-stopping amendment is the sole difference from the
    # historical P5 recipe.  Normalise it away before asserting scientific
    # source matching; the strict P6 A/B still differs only in inter-class use.
    reference_control = p6_control
    reference_training = reference_control["training"]
    reference_training.pop("early_stopping_start_epoch", None)
    reference_training["early_stopping_patience"] = 0
    if reference_control != p5_treatment:
        raise RuntimeError(
            "P6 source-matched control differs from P5 beyond the shared "
            "delayed-stopping amendment"
        )


def delayed_terminal_curve_diagnostic(path: Path, profile: str) -> dict:
    """Validate a complete max-120 or policy-compliant early-stop terminal."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    terminal_epoch = int(checkpoint.get("epoch", -1))
    if not 1 <= terminal_epoch <= P6_MAXIMUM_EPOCHS:
        raise RuntimeError(f"Invalid delayed-stop terminal epoch: {path}")
    config = checkpoint.get("config", {}) or {}
    training = config.get("training", {}) or {}
    if (
        int(training.get("epochs", -1)) != P6_MAXIMUM_EPOCHS
        or int(training.get("early_stopping_start_epoch", -1))
        != P6_EARLY_STOPPING_START_EPOCH
        or int(training.get("early_stopping_patience", -1))
        != P6_EARLY_STOPPING_PATIENCE
    ):
        raise RuntimeError(f"Delayed-stop policy mismatch: {path}")
    diagnostic = terminal_curve_diagnostic(
        path, profile, expected_epoch=terminal_epoch
    )
    history = checkpoint.get(
        "training_history", checkpoint.get("history", [])
    )
    best = -float("inf")
    stale = 0
    for row in history:
        epoch = int(row.get("epoch", -1))
        value = row.get("val_macro_f1")
        if not isinstance(value, (int, float)):
            continue
        if float(value) > best:
            best = float(value)
            stale = 0
        elif epoch >= P6_EARLY_STOPPING_START_EPOCH:
            stale += 1
        else:
            stale = 0
    if terminal_epoch < P6_MAXIMUM_EPOCHS and (
        terminal_epoch < P6_EARLY_STOPPING_START_EPOCH
        or stale < P6_EARLY_STOPPING_PATIENCE
    ):
        raise RuntimeError(f"Premature P6 terminal does not satisfy policy: {path}")
    diagnostic["delayed_early_stopping"] = {
        "maximum_epochs": P6_MAXIMUM_EPOCHS,
        "start_epoch": P6_EARLY_STOPPING_START_EPOCH,
        "patience": P6_EARLY_STOPPING_PATIENCE,
        "terminal_staleness": stale,
        "stopped_early": terminal_epoch < P6_MAXIMUM_EPOCHS,
    }
    return diagnostic


def _guardrail(candidate: dict, reference: dict, key: str) -> bool:
    return bool(
        float(candidate[key])
        >= float(reference[key]) - MAXIMUM_GUARDRAIL_DROP
    )


def acceptance_gate(
    *,
    p6_control_metrics: dict,
    p6_treatment_metrics: dict,
    p5_control_metrics: dict,
    p5_treatment_metrics: dict,
    external_metrics: dict,
    fusion_metrics: dict,
    rescue_rate: float,
    spread_ratio: float,
    energy_ratio: float,
) -> dict:
    """Apply every fixed P6 score and mechanism condition."""
    checks = {
        "source_matched_macro_gain": (
            p6_treatment_metrics["macro_f1"]
            - p6_control_metrics["macro_f1"]
            >= MINIMUM_MACRO_GAIN
        ),
        "historical_p5_control_macro_gain": (
            p6_treatment_metrics["macro_f1"]
            - p5_control_metrics["macro_f1"]
            >= MINIMUM_MACRO_GAIN
        ),
        "historical_p5_treatment_noninferiority": (
            p6_treatment_metrics["macro_f1"]
            >= p5_treatment_metrics["macro_f1"]
        ),
        "fixed_fusion_macro_gain": (
            fusion_metrics["macro_f1"] - external_metrics["macro_f1"]
            >= MINIMUM_MACRO_GAIN
        ),
        "p6_control_known_guardrail": _guardrail(
            p6_treatment_metrics, p6_control_metrics, "known_accuracy"
        ),
        "p6_control_ood_guardrail": _guardrail(
            p6_treatment_metrics, p6_control_metrics, "ood_f1"
        ),
        "historical_p5_control_known_guardrail": _guardrail(
            p6_treatment_metrics, p5_control_metrics, "known_accuracy"
        ),
        "historical_p5_control_ood_guardrail": _guardrail(
            p6_treatment_metrics, p5_control_metrics, "ood_f1"
        ),
        "fixed_fusion_known_guardrail": _guardrail(
            fusion_metrics, external_metrics, "known_accuracy"
        ),
        "fixed_fusion_ood_guardrail": _guardrail(
            fusion_metrics, external_metrics, "ood_f1"
        ),
        "minimum_rescue_rate": rescue_rate >= MINIMUM_RESCUE_RATE,
        "embedding_spread_ratio": (
            spread_ratio >= MINIMUM_EMBEDDING_SPREAD_RATIO
        ),
        "exclusive_energy_ratio": energy_ratio <= MAXIMUM_ENERGY_RATIO,
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "thresholds": {
            "minimum_macro_gain": MINIMUM_MACRO_GAIN,
            "maximum_known_or_ood_drop": MAXIMUM_GUARDRAIL_DROP,
            "minimum_rescue_rate": MINIMUM_RESCUE_RATE,
            "minimum_embedding_spread_ratio": MINIMUM_EMBEDDING_SPREAD_RATIO,
            "maximum_exclusive_energy_ratio": MAXIMUM_ENERGY_RATIO,
        },
    }


def _load_profile_bundle(
    *,
    profile: str,
    checkpoint_root: Path,
    external_oof: dict,
    expected_validation: set[str],
    train_frame,
    cluster_map_path: Path,
    cache_dir: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> dict:
    checkpoint_dir = checkpoint_root / profile
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
        cluster_map_path=cluster_map_path,
        cache_path=cache_dir / profile / "fold0_train_embeddings_centroids.npz",
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    return {
        "checkpoint_path": checkpoint_path,
        "latest_path": latest_path,
        "oof_path": oof_path,
        "binding": binding,
        "oof": oof,
        "artifact": artifact,
        "artifact_metadata": metadata,
        "checkpoint": checkpoint,
        "terminal_curve": delayed_terminal_curve_diagnostic(
            latest_path, profile
        ),
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
        "--p5-cache-dir",
        type=Path,
        default=ROOT / "data" / "experiments" / "campp_cross_file_consistency_lme20",
    )
    parser.add_argument(
        "--p6-cache-dir",
        type=Path,
        default=ROOT / "data" / "experiments" / "campp_inter_class_lme20",
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
        default=ROOT / "reports" / "generated" / "campp_inter_class_lme20_fold0.json",
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

    loaded: dict[str, dict] = {}
    profiles = tuple(PROFILE_HASHES)
    for profile in profiles:
        config_path = ROOT / "configs" / "experiments" / f"{profile}.yaml"
        if sha256_file(config_path) != PROFILE_HASHES[profile]:
            raise RuntimeError(f"Locked config SHA changed: {profile}")
        cache_dir = (
            args.p5_cache_dir if profile.startswith("p5-") else args.p6_cache_dir
        )
        loaded[profile] = _load_profile_bundle(
            profile=profile,
            checkpoint_root=args.checkpoint_root,
            external_oof=external_oof,
            expected_validation=expected_validation,
            train_frame=train_frame,
            cluster_map_path=args.cluster_map,
            cache_dir=cache_dir,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    p5_control = loaded[P5_CROSS_FILE_MATCHED_PROFILE]
    p5_treatment = loaded[P5_CROSS_FILE_TREATMENT_PROFILE]
    p6_control = loaded[P6_CONTROL_PROFILE]
    p6_treatment = loaded[P6_TREATMENT_PROFILE]
    assert_p6_single_objective_contract(
        p5_treatment["checkpoint"]["config"],
        p6_control["checkpoint"]["config"],
        p6_treatment["checkpoint"]["config"],
    )
    class_maps = [bundle["checkpoint"]["class_map"] for bundle in loaded.values()]
    if any(class_map != class_maps[0] for class_map in class_maps[1:]):
        raise RuntimeError("P5/P6 class maps differ")
    splits_seen = [
        bundle["checkpoint"]["config"]["data"]["split"]
        for bundle in loaded.values()
    ]
    if any(split != splits_seen[0] for split in splits_seen[1:]):
        raise RuntimeError("P5/P6 split provenance differs")
    source_checkpoint = args.checkpoint_root / SOURCE_PROFILE / "campp_best_raw.pt"
    if sha256_file(source_checkpoint) != SOURCE_CHECKPOINT_SHA256:
        raise RuntimeError("Locked source checkpoint SHA changed")

    bundles = {"external": {"artifact": external_artifact, "oof": external_oof}}
    bundles.update(loaded)
    predictions: dict[str, np.ndarray] = {}
    evidences: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, bundle in bundles.items():
        evidences[name] = probability_evidence(
            bundle["artifact"],
            bundle["oof"]["embeddings"],
            bundle["oof"]["competition_probs"],
        )
        _, predictions[name] = final_decision(*evidences[name])
    _, fusion_predictions = fixed_raw_ema_decision(
        evidences["external"], evidences[P6_TREATMENT_PROFILE]
    )
    labels = external_oof["labels"].astype(np.int64)
    metrics = {
        name: metric_bundle(labels, prediction)
        for name, prediction in predictions.items()
    }
    external_metrics = metrics["external"]
    if abs(
        external_metrics["macro_f1"] - EXTERNAL_CONTROL_LME20_MACRO_F1
    ) > 1e-10:
        raise RuntimeError("Locked external Control LME20 reproduction failed")
    fusion_metrics = metric_bundle(labels, fusion_predictions)
    treatment_changes = error_changes(
        labels,
        predictions[P6_CONTROL_PROFILE],
        predictions[P6_TREATMENT_PROFILE],
    )
    fusion_changes = error_changes(
        labels, predictions["external"], fusion_predictions
    )
    p6_control_spread = embedding_spread(p6_control["artifact"])
    p6_treatment_spread = embedding_spread(p6_treatment["artifact"])
    spread_ratio = float(p6_treatment_spread / p6_control_spread)
    energy = audit_pair(
        p6_control["checkpoint_path"],
        p6_treatment["checkpoint_path"],
        maximum_energy_ratio=MAXIMUM_ENERGY_RATIO,
    )
    energy_ratio = float(
        energy["mechanism_gate"]["observed_energy_ratio"]
    )
    gate = acceptance_gate(
        p6_control_metrics=metrics[P6_CONTROL_PROFILE],
        p6_treatment_metrics=metrics[P6_TREATMENT_PROFILE],
        p5_control_metrics=metrics[P5_CROSS_FILE_MATCHED_PROFILE],
        p5_treatment_metrics=metrics[P5_CROSS_FILE_TREATMENT_PROFILE],
        external_metrics=external_metrics,
        fusion_metrics=fusion_metrics,
        rescue_rate=float(treatment_changes["rescue_rate"]),
        spread_ratio=spread_ratio,
        energy_ratio=energy_ratio,
    )

    report = {
        "contract": {
            "scope": "Fold 0 source-matched P6 inter-class angular A/B",
            "parameter_search": False,
            "leaderboard_tuning": False,
            "decision": "selected Raw probability-average LME20 direct argmax",
            "fusion": "fixed 50/50 probability-evidence average",
            "inter_class_type": "exclusive_angular_energy",
            "inter_class_weight": 0.01,
            "adaptive_horizon": {
                "maximum_epochs": P6_MAXIMUM_EPOCHS,
                "early_stopping_start_epoch": P6_EARLY_STOPPING_START_EPOCH,
                "patience": P6_EARLY_STOPPING_PATIENCE,
            },
        },
        "provenance": {
            "cleaning": cleaning,
            "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
            "config_sha256": PROFILE_HASHES,
            "external_control_artifact": external_metadata[0],
            "validation_files": int(len(labels)),
            "validation_file_sha256": digest_names(
                external_oof["files"].astype(str)
            ),
            "profiles": {
                profile: {
                    "checkpoint_sha256": sha256_file(bundle["checkpoint_path"]),
                    "latest_sha256": sha256_file(bundle["latest_path"]),
                    "oof_sha256": sha256_file(bundle["oof_path"]),
                    "bundle_binding": bundle["binding"],
                    "train_artifact": bundle["artifact_metadata"],
                    "terminal_curve": bundle["terminal_curve"],
                }
                for profile, bundle in loaded.items()
            },
        },
        "metrics": metrics,
        "p6_treatment_deltas": {
            "vs_p6_control": metric_delta(
                metrics[P6_TREATMENT_PROFILE], metrics[P6_CONTROL_PROFILE]
            ),
            "vs_p5_control": metric_delta(
                metrics[P6_TREATMENT_PROFILE],
                metrics[P5_CROSS_FILE_MATCHED_PROFILE],
            ),
            "vs_p5_treatment": metric_delta(
                metrics[P6_TREATMENT_PROFILE],
                metrics[P5_CROSS_FILE_TREATMENT_PROFILE],
            ),
        },
        "p6_treatment_error_changes_vs_p6_control": treatment_changes,
        "fixed_fusion": {
            "metrics": fusion_metrics,
            "delta_vs_external_control": metric_delta(
                fusion_metrics, external_metrics
            ),
            "error_changes_vs_external_control": fusion_changes,
        },
        "paired_randomization_diagnostics": {
            "p6_treatment_vs_p6_control": paired_randomization_diagnostic(
                labels,
                predictions[P6_CONTROL_PROFILE],
                predictions[P6_TREATMENT_PROFILE],
            ),
            "fixed_fusion_vs_external_control": paired_randomization_diagnostic(
                labels, predictions["external"], fusion_predictions
            ),
        },
        "collapse_guard": {
            "p6_control": p6_control_spread,
            "p6_treatment": p6_treatment_spread,
            "treatment_over_control": spread_ratio,
        },
        "inter_class_energy": energy,
        "acceptance_gate": gate,
        "decision": "accept" if gate["passed"] else "reject",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.output)
    print(json.dumps({
        "output": str(args.output),
        "decision": report["decision"],
        "p6_treatment_deltas": report["p6_treatment_deltas"],
        "fixed_fusion": report["fixed_fusion"],
        "collapse_guard": report["collapse_guard"],
        "inter_class_energy": energy,
        "acceptance_gate": gate,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
