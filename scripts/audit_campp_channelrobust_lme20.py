"""Locked Fold-0 LME20 audit for the channel-robust CAM++ complement.

The training treatment changes only augmentation.  This evaluator is frozen
before the first completed treatment epoch and performs no parameter search:
it compares Control, the selected channel-robust candidate, and one fixed
50/50 probability-evidence fusion under the already shipped LME20 policy.
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
from scripts.audit_raw_ema_lme20 import (  # noqa: E402
    final_decision,
    fixed_raw_ema_decision,
    probability_evidence,
)
from scripts.audit_short_audio_repeat import digest_names  # noqa: E402


CONTROL_PROFILE = "p0-campp-known446-ood-control-oof-f0"
CANDIDATE_PROFILE = "p3-campp-known446-ood-channelrobust-oof-f0"
CONTINUATION_PROFILE = (
    "p3-campp-known446-ood-channelrobust-continuation-oof-f0"
)
CONTROL_FOLD0_LME20_MACRO_F1 = 0.9611456662793696
MINIMUM_FUSION_MACRO_GAIN = 0.002
MAXIMUM_KNOWN_DROP = 0.001
MAXIMUM_OOD_DROP = 0.001
MINIMUM_CANDIDATE_RESCUE_RATE = 0.20
MAXIMUM_STANDALONE_DEFICIT = 0.010


def assert_augmentation_only_contract(
    control_config: dict, candidate_config: dict
) -> None:
    """Require the terminal treatment to differ only in augmentation/identity."""
    control = copy.deepcopy(control_config)
    candidate = copy.deepcopy(candidate_config)
    for config in (control, candidate):
        config.pop("_meta", None)
        config.pop("experiment", None)
        config.pop("logging", None)
    if candidate.get("augmentation") == control.get("augmentation"):
        raise RuntimeError("Candidate augmentation treatment is missing")
    candidate["augmentation"] = copy.deepcopy(control.get("augmentation"))
    if candidate != control:
        changed = sorted(
            key for key in set(control) | set(candidate)
            if control.get(key) != candidate.get(key)
        )
        raise RuntimeError(
            "Candidate changed fields outside augmentation: " + ", ".join(changed)
        )


def assert_stateful_continuation_contract(
    source_config: dict,
    continuation_config: dict,
) -> None:
    """Prove that a continuation changed only resume identity and patience.

    The final scientific treatment remains the source augmentation policy.
    This check prevents the terminal evaluator from silently accepting a
    continuation that changed a loss, learning rate, split, architecture or
    any other treatment field.
    """
    source = copy.deepcopy(source_config)
    continuation = copy.deepcopy(continuation_config)
    for config in (source, continuation):
        config.pop("_meta", None)
        config.pop("experiment", None)
        config.pop("logging", None)

    source_training = source.get("training", {}) or {}
    continuation_training = continuation.get("training", {}) or {}
    if int(source_training.get("early_stopping_patience", -1)) != 20:
        raise RuntimeError("Unexpected source early-stopping patience")
    if int(continuation_training.get("early_stopping_patience", -1)) != 12:
        raise RuntimeError("Unexpected continuation early-stopping patience")

    resume_checkpoint = str(
        continuation_training.get("resume_checkpoint", "")
    )
    resume_history = str(
        continuation_training.get("resume_history_path", "")
    )
    expected_dir = f"checkpoints/{CANDIDATE_PROFILE}/"
    if not resume_checkpoint.replace("\\", "/").endswith(
        expected_dir + "campp_best_raw.pt"
    ):
        raise RuntimeError("Continuation resume checkpoint is not source Raw")
    if not resume_history.replace("\\", "/").endswith(
        expected_dir + "campp_latest.pt"
    ):
        raise RuntimeError("Continuation history is not the source latest checkpoint")

    continuation_training.pop("resume_checkpoint", None)
    continuation_training.pop("resume_history_path", None)
    continuation_training["early_stopping_patience"] = source_training[
        "early_stopping_patience"
    ]
    if continuation != source:
        changed = sorted(
            key for key in set(source) | set(continuation)
            if source.get(key) != continuation.get(key)
        )
        raise RuntimeError(
            "Continuation changed the scientific contract: "
            + ", ".join(changed)
        )


def validate_raw_bundle_binding(
    candidate_dir: Path,
    raw_checkpoint_path: Path,
    oof_path: Path,
) -> dict:
    """Bind the selected bundle and OOF to the exact selected Raw weights."""
    bundle_dir = oof_path.parent
    manifest_path = bundle_dir / "manifest.json"
    selected_checkpoint_path = candidate_dir / "campp_best.pt"
    if not manifest_path.is_file():
        raise RuntimeError(f"Candidate bundle manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if Path(str(manifest.get("checkpoint", ""))).name != selected_checkpoint_path.name:
        raise RuntimeError("Candidate bundle manifest points to an unexpected checkpoint")
    selected_sha = sha256_file(selected_checkpoint_path)
    raw_sha = sha256_file(raw_checkpoint_path)
    oof_sha = sha256_file(oof_path)
    if manifest.get("checkpoint_sha256") != selected_sha:
        raise RuntimeError("Candidate selected-checkpoint manifest SHA mismatch")
    if manifest.get("oof_predictions_sha256") != oof_sha:
        raise RuntimeError("Candidate OOF manifest SHA mismatch")

    selected = torch.load(
        selected_checkpoint_path, map_location="cpu", weights_only=False
    )
    raw = torch.load(raw_checkpoint_path, map_location="cpu", weights_only=False)
    if selected.get("weight_variant") != "raw" or raw.get("weight_variant") != "raw":
        raise RuntimeError("Candidate selected checkpoint is not Raw")
    if int(selected.get("epoch", -1)) != int(raw.get("epoch", -2)):
        raise RuntimeError("Candidate selected/Raw epochs differ")
    if not np.isclose(
        float(selected.get("val_macro_f1", np.nan)),
        float(raw.get("val_macro_f1", np.nan)),
        atol=1e-12,
        rtol=0.0,
    ):
        raise RuntimeError("Candidate selected/Raw metrics differ")
    selected_state = selected.get("model_state_dict") or {}
    raw_state = raw.get("model_state_dict") or {}
    if selected_state.keys() != raw_state.keys() or not all(
        torch.equal(selected_state[key], raw_state[key]) for key in selected_state
    ):
        raise RuntimeError("Candidate selected/Raw model states differ")
    return {
        "manifest_sha256": sha256_file(manifest_path),
        "selected_checkpoint_sha256": selected_sha,
        "raw_checkpoint_sha256": raw_sha,
        "oof_predictions_sha256": oof_sha,
        "selected_epoch": int(selected["epoch"]),
        "selected_val_macro_f1": float(selected["val_macro_f1"]),
        "weight_variant": "raw",
    }


def align_oof(reference: dict, candidate: dict) -> dict:
    """Align a candidate OOF payload to the immutable Control file order."""
    reference_files = np.asarray(reference["files"]).astype(str)
    candidate_files = np.asarray(candidate["files"]).astype(str)
    if len(set(reference_files.tolist())) != len(reference_files):
        raise RuntimeError("Control OOF filenames are not unique")
    if len(set(candidate_files.tolist())) != len(candidate_files):
        raise RuntimeError("Candidate OOF filenames are not unique")
    if set(reference_files.tolist()) != set(candidate_files.tolist()):
        raise RuntimeError("Candidate and Control OOF filename sets differ")
    candidate_index = {name: index for index, name in enumerate(candidate_files)}
    order = np.asarray([candidate_index[name] for name in reference_files], dtype=int)
    aligned = {}
    for key, values in candidate.items():
        array = np.asarray(values)
        aligned[key] = (
            array[order].copy()
            if array.ndim > 0 and array.shape[0] == len(candidate_files)
            else array.copy()
        )
    if not np.array_equal(aligned["files"].astype(str), reference_files):
        raise RuntimeError("Candidate OOF alignment failed")
    if not np.array_equal(
        aligned["labels"].astype(np.int64),
        np.asarray(reference["labels"]).astype(np.int64),
    ):
        raise RuntimeError("Candidate and Control OOF labels differ")
    for key in ("competition_probs", "embeddings"):
        if not np.all(np.isfinite(aligned[key])):
            raise RuntimeError(f"Candidate {key} contains non-finite values")
    probabilities = np.asarray(aligned["competition_probs"], dtype=np.float64)
    if np.any(probabilities < -1e-7) or np.any(probabilities > 1.0 + 1e-7):
        raise RuntimeError("Candidate competition_probs fall outside [0, 1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5, rtol=0.0):
        raise RuntimeError("Candidate competition_probs rows do not sum to one")
    return aligned


def error_changes(
    labels: np.ndarray, baseline: np.ndarray, candidate: np.ndarray
) -> dict[str, float | int]:
    baseline_correct = baseline == labels
    candidate_correct = candidate == labels
    baseline_errors = int(np.sum(~baseline_correct))
    rescued = int(np.sum(~baseline_correct & candidate_correct))
    return {
        "baseline_errors": baseline_errors,
        "rescued_errors": rescued,
        "introduced_errors": int(np.sum(baseline_correct & ~candidate_correct)),
        "changed_predictions": int(np.sum(baseline != candidate)),
        "rescue_rate": float(rescued / max(baseline_errors, 1)),
    }


def acceptance_gate(
    standalone_delta: dict[str, float],
    fusion_delta: dict[str, float],
    candidate_rescue_rate: float,
) -> dict:
    checks = {
        "standalone_lme20_deficit": (
            standalone_delta["macro_f1"] >= -MAXIMUM_STANDALONE_DEFICIT
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
        "candidate_rescue_rate": (
            candidate_rescue_rate >= MINIMUM_CANDIDATE_RESCUE_RATE
        ),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "thresholds": {
            "minimum_fixed_fusion_macro_gain": MINIMUM_FUSION_MACRO_GAIN,
            "maximum_known_accuracy_drop": MAXIMUM_KNOWN_DROP,
            "maximum_ood_f1_drop": MAXIMUM_OOD_DROP,
            "minimum_candidate_rescue_rate": MINIMUM_CANDIDATE_RESCUE_RATE,
            "maximum_standalone_lme20_deficit": MAXIMUM_STANDALONE_DEFICIT,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-root", type=Path, default=ROOT / "checkpoints"
    )
    parser.add_argument(
        "--control-cache-dir",
        type=Path,
        default=ROOT / "data" / "experiments" / "campp_control_centroid_crossfit",
    )
    parser.add_argument(
        "--candidate-profile",
        choices=[CANDIDATE_PROFILE, CONTINUATION_PROFILE],
        default=CANDIDATE_PROFILE,
    )
    parser.add_argument(
        "--candidate-cache-dir",
        type=Path,
        default=None,
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
        default=None,
    )
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    candidate_profile = str(args.candidate_profile)
    candidate_cache_dir = (
        args.candidate_cache_dir
        if args.candidate_cache_dir is not None
        else (
            ROOT
            / "data"
            / "experiments"
            / "campp_channelrobust_lme20"
            / candidate_profile
        )
    )
    output_path = (
        args.output
        if args.output is not None
        else (
            ROOT
            / "reports"
            / "generated"
            / f"campp_channelrobust_lme20_{candidate_profile}.json"
        )
    )

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    raw_oofs, raw_artifacts, raw_metadata = load_fold_inputs(
        args.checkpoint_root, args.control_cache_dir
    )
    raw_oof = raw_oofs[0]
    raw_artifact = raw_artifacts[0]

    splits, cleaning = rebuild_exact_splits(args.labels, args.audio_dir)
    train_frame, validation_frame = splits[0]
    expected_validation = set(validation_frame["audio_file"].astype(str))
    candidate_dir = args.checkpoint_root / candidate_profile
    candidate_checkpoint_path = candidate_dir / "campp_best_raw.pt"
    candidate_oof_path = (
        candidate_dir / "campp_best_bundle" / "oof_predictions.npz"
    )
    candidate_bundle_binding = validate_raw_bundle_binding(
        candidate_dir, candidate_checkpoint_path, candidate_oof_path
    )
    candidate_oof = align_oof(
        raw_oof,
        load_oof(candidate_oof_path, 0, expected_validation),
    )
    candidate_artifact, candidate_metadata = build_or_load_train_artifact(
        fold=0,
        train_frame=train_frame,
        checkpoint_path=candidate_checkpoint_path,
        cluster_map_path=args.cluster_map,
        cache_path=(
            candidate_cache_dir / "fold0_train_embeddings_centroids.npz"
        ),
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    control_checkpoint_path = (
        args.checkpoint_root / CONTROL_PROFILE / "campp_best_raw.pt"
    )
    control_checkpoint = torch.load(
        control_checkpoint_path, map_location="cpu", weights_only=False
    )
    candidate_checkpoint = torch.load(
        candidate_checkpoint_path, map_location="cpu", weights_only=False
    )
    if control_checkpoint["class_map"] != candidate_checkpoint["class_map"]:
        raise RuntimeError("Control and candidate class maps differ")
    control_split = control_checkpoint["config"]["data"]["split"]
    candidate_split = candidate_checkpoint["config"]["data"]["split"]
    if control_split != candidate_split:
        raise RuntimeError("Control and candidate split provenance differ")
    treatment_config = candidate_checkpoint["config"]
    continuation_source_sha256 = None
    if candidate_profile == CONTINUATION_PROFILE:
        source_checkpoint_path = (
            args.checkpoint_root / CANDIDATE_PROFILE / "campp_best_raw.pt"
        )
        source_checkpoint = torch.load(
            source_checkpoint_path, map_location="cpu", weights_only=False
        )
        if source_checkpoint["class_map"] != candidate_checkpoint["class_map"]:
            raise RuntimeError("Continuation and source class maps differ")
        if (
            source_checkpoint["config"]["data"]["split"]
            != candidate_checkpoint["config"]["data"]["split"]
        ):
            raise RuntimeError("Continuation and source split provenance differ")
        assert_stateful_continuation_contract(
            source_checkpoint["config"], candidate_checkpoint["config"]
        )
        treatment_config = source_checkpoint["config"]
        continuation_source_sha256 = sha256_file(source_checkpoint_path)
        del source_checkpoint
    assert_augmentation_only_contract(control_checkpoint["config"], treatment_config)
    del control_checkpoint, candidate_checkpoint

    control_evidence = probability_evidence(
        raw_artifact,
        raw_oof["embeddings"],
        raw_oof["competition_probs"],
    )
    candidate_evidence = probability_evidence(
        candidate_artifact,
        candidate_oof["embeddings"],
        candidate_oof["competition_probs"],
    )
    _, control_predictions = final_decision(*control_evidence)
    _, candidate_predictions = final_decision(*candidate_evidence)
    _, fusion_predictions = fixed_raw_ema_decision(
        control_evidence, candidate_evidence
    )

    labels = raw_oof["labels"].astype(np.int64)
    control_metrics = metric_bundle(labels, control_predictions)
    if (
        abs(control_metrics["macro_f1"] - CONTROL_FOLD0_LME20_MACRO_F1)
        > 1e-10
    ):
        raise RuntimeError("Locked Control Fold-0 LME20 reproduction failed")
    candidate_metrics = metric_bundle(labels, candidate_predictions)
    fusion_metrics = metric_bundle(labels, fusion_predictions)
    standalone_delta = metric_delta(candidate_metrics, control_metrics)
    fusion_delta = metric_delta(fusion_metrics, control_metrics)
    candidate_changes = error_changes(
        labels, control_predictions, candidate_predictions
    )
    fusion_changes = error_changes(labels, control_predictions, fusion_predictions)
    gate = acceptance_gate(
        standalone_delta,
        fusion_delta,
        float(candidate_changes["rescue_rate"]),
    )

    report = {
        "contract": {
            "scope": "Fold 0 only",
            "single_training_treatment": "channel/session augmentation policy",
            "control_profile": CONTROL_PROFILE,
            "candidate_profile": candidate_profile,
            "stateful_continuation": candidate_profile == CONTINUATION_PROFILE,
            "fusion": "fixed 50/50 probability-evidence average",
            "parameter_search": False,
            "leaderboard_tuning": False,
        },
        "provenance": {
            "cleaning": cleaning,
            "control_checkpoint_sha256": sha256_file(control_checkpoint_path),
            "candidate_checkpoint_sha256": sha256_file(candidate_checkpoint_path),
            "continuation_source_checkpoint_sha256": continuation_source_sha256,
            "candidate_oof_sha256": sha256_file(candidate_oof_path),
            "candidate_bundle_binding": candidate_bundle_binding,
            "control_artifact": raw_metadata[0],
            "candidate_artifact": candidate_metadata,
            "validation_files": int(len(labels)),
            "validation_file_sha256": digest_names(raw_oof["files"].astype(str)),
        },
        "control": control_metrics,
        "candidate": {
            "metrics": candidate_metrics,
            "delta": standalone_delta,
            "error_changes": candidate_changes,
        },
        "fixed_fusion": {
            "metrics": fusion_metrics,
            "delta": fusion_delta,
            "error_changes": fusion_changes,
        },
        "acceptance_gate": gate,
        "decision": "accept" if gate["passed"] else "reject",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(temporary, output_path)
    print(json.dumps({
        "output": str(output_path),
        "decision": report["decision"],
        "control": control_metrics,
        "candidate": report["candidate"],
        "fixed_fusion": report["fixed_fusion"],
        "gate": gate,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
