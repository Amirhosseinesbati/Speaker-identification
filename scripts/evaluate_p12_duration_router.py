"""Evaluate the preregistered P12 structural duration router on aligned OOF."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_control_oof_centroid_crossfit import (  # noqa: E402
    metric_bundle,
    metric_delta,
)
from scripts.analyze_oof_actual_duration import (  # noqa: E402
    align_pair,
    wav_duration_seconds,
)
from scripts.evaluate_fixed_oof_pair import (  # noqa: E402
    class_coverage,
    load_oof,
    observed_class_macro_f1,
    sha256_file,
    transition_bundle,
)


ANALYSIS_ID = "p12-campp-duration-router-f0-v1"
EXPECTED_STATUS = "preregistered_before_router_metric_evaluation"


def verify_sha256(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"{label} SHA mismatch: expected={expected}, actual={actual}"
        )
    return actual


def load_contract(path: Path, expected_sha256: str) -> dict:
    verify_sha256(path, expected_sha256, "P12 preregistration")
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("analysis_id") != ANALYSIS_ID:
        raise RuntimeError(
            f"Unexpected P12 analysis id: {contract.get('analysis_id')!r}"
        )
    if contract.get("status") != EXPECTED_STATUS:
        raise RuntimeError(
            f"Unexpected P12 preregistration status: {contract.get('status')!r}"
        )
    router = contract.get("locked_router") or {}
    if router.get("short_if_duration_seconds_le") != 8.0:
        raise RuntimeError("P12 router cutoff must be exactly 8.0 seconds")
    if router.get("blend") is not False:
        raise RuntimeError("P12 router must not blend model probabilities")
    if router.get("threshold_search_dimensions") != 0:
        raise RuntimeError("P12 router must have zero threshold search dimensions")
    gate = contract.get("gate") or {}
    required_gate = {
        "all_checks_required",
        "min_raw_macro_f1_gain_vs_p0",
        "max_known_accuracy_drop_vs_p0",
        "max_ood_f1_drop_vs_p0",
        "min_routed_subset_p0_error_rescue_rate",
        "require_global_rescued_errors_gt_introduced",
        "require_routed_subset_rescued_errors_gt_introduced",
        "require_identical_file_set_labels_split_and_class_map",
    }
    missing = required_gate - set(gate)
    if missing:
        raise RuntimeError(f"P12 gate lacks keys: {sorted(missing)}")
    if gate["all_checks_required"] is not True:
        raise RuntimeError("P12 requires every gate check")
    return contract


def validate_probabilities(probabilities: np.ndarray) -> None:
    if probabilities.ndim != 2 or probabilities.shape[1] != 447:
        raise RuntimeError(
            f"Routed probabilities have invalid shape: {probabilities.shape}"
        )
    if not np.isfinite(probabilities).all() or np.any(probabilities < -1e-7):
        raise RuntimeError("Routed probabilities contain NaN/Inf or negatives")
    if not np.allclose(
        probabilities.sum(axis=1), 1.0, rtol=0.0, atol=2e-5
    ):
        raise RuntimeError("Routed probability rows do not sum to one")


def evaluate_router(
    baseline: dict[str, np.ndarray],
    specialist: dict[str, np.ndarray],
    durations: np.ndarray,
    contract: dict,
) -> tuple[dict, dict[str, np.ndarray]]:
    files, labels, baseline_probs, specialist_probs, split = align_pair(
        baseline, specialist
    )
    durations = np.asarray(durations, dtype=np.float64)
    if durations.shape != (len(files),):
        raise RuntimeError(
            f"Duration shape mismatch: expected={(len(files),)}, "
            f"observed={durations.shape}"
        )
    if not np.isfinite(durations).all() or np.any(durations <= 0):
        raise RuntimeError("Durations must be finite and positive")

    cutoff = float(contract["locked_router"]["short_if_duration_seconds_le"])
    routed_mask = durations <= cutoff
    routed_probs = baseline_probs.copy()
    routed_probs[routed_mask] = specialist_probs[routed_mask]
    validate_probabilities(routed_probs)
    baseline_predictions = baseline_probs.argmax(axis=1).astype(np.int64)
    specialist_predictions = specialist_probs.argmax(axis=1).astype(np.int64)
    routed_predictions = routed_probs.argmax(axis=1).astype(np.int64)

    baseline_metrics = metric_bundle(labels, baseline_predictions)
    routed_metrics = metric_bundle(labels, routed_predictions)
    delta = metric_delta(routed_metrics, baseline_metrics)
    global_transitions = transition_bundle(
        labels, baseline_predictions, routed_predictions
    )
    routed_labels = labels[routed_mask]
    routed_baseline_predictions = baseline_predictions[routed_mask]
    routed_specialist_predictions = specialist_predictions[routed_mask]
    routed_transitions = transition_bundle(
        routed_labels,
        routed_baseline_predictions,
        routed_specialist_predictions,
    )
    gate = contract["gate"]
    checks = {
        "raw_macro_f1_gain_vs_p0": (
            delta["macro_f1"]
            >= float(gate["min_raw_macro_f1_gain_vs_p0"])
        ),
        "known_accuracy_guardrail_vs_p0": (
            delta["known_accuracy"]
            >= -float(gate["max_known_accuracy_drop_vs_p0"])
        ),
        "ood_f1_guardrail_vs_p0": (
            delta["ood_f1"] >= -float(gate["max_ood_f1_drop_vs_p0"])
        ),
        "routed_subset_p0_error_rescue_rate": (
            routed_transitions["rescue_rate"]
            >= float(gate["min_routed_subset_p0_error_rescue_rate"])
        ),
        "global_rescued_errors_exceed_introduced": (
            global_transitions["rescued_errors"]
            > global_transitions["introduced_errors"]
        ),
        "routed_subset_rescued_errors_exceed_introduced": (
            routed_transitions["rescued_errors"]
            > routed_transitions["introduced_errors"]
        ),
        "integrity": True,
    }
    passed = all(checks.values())
    result = {
        "contract": {
            "analysis_id": ANALYSIS_ID,
            "cutoff_seconds": cutoff,
            "short_boundary_inclusive": True,
            "blend": False,
            "search_dimensions": 0,
            "decision": "Raw probability-average direct argmax",
            "leaderboard_used": False,
            "single_fold_submission_authorized": False,
        },
        "integrity": {
            "rows": int(len(files)),
            "unique_files": int(len(set(files.tolist()))),
            "probability_columns": int(routed_probs.shape[1]),
            "split": split,
        },
        "routing": {
            "short_specialist_rows": int(np.sum(routed_mask)),
            "long_control_rows": int(np.sum(~routed_mask)),
            "short_fraction": float(np.mean(routed_mask)),
            "duration_min": float(durations.min()),
            "duration_max": float(durations.max()),
        },
        "metrics": {
            "p0_control": baseline_metrics,
            "p12_router": routed_metrics,
            "p12_delta_vs_p0": delta,
            "observed_class_macro_f1_descriptive_only": {
                "p0_control": observed_class_macro_f1(
                    labels, baseline_predictions
                ),
                "p12_router": observed_class_macro_f1(
                    labels, routed_predictions
                ),
            },
        },
        "global_error_transitions": global_transitions,
        "routed_subset": {
            "coverage": class_coverage(routed_labels),
            "p0_metrics": metric_bundle(
                routed_labels, routed_baseline_predictions
            ),
            "specialist_metrics": metric_bundle(
                routed_labels, routed_specialist_predictions
            ),
            "error_transitions": routed_transitions,
        },
        "gate": {
            "contract": dict(gate),
            "checks": checks,
            "passed": passed,
        },
        "scientific_decision": (
            "passed_fold0_router_gate_later_folds_require_separate_prereg"
            if passed
            else "rejected_fold0_router_gate_close_p11_p12"
        ),
    }
    output = {
        "files": files,
        "labels": labels,
        "competition_probs": routed_probs.astype(np.float32),
        "duration_seconds": durations.astype(np.float32),
        "short_specialist_mask": routed_mask,
        "router_cutoff_seconds": np.asarray([cutoff], dtype=np.float32),
        "split_fold": np.asarray([split["split_fold"]], dtype=np.int64),
        "split_folds": np.asarray([split["split_folds"]], dtype=np.int64),
        "split_seed": np.asarray([split["split_seed"]], dtype=np.int64),
    }
    return result, output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--expected-baseline-sha256", required=True)
    parser.add_argument("--specialist", type=Path, required=True)
    parser.add_argument("--expected-specialist-sha256", required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--expected-prereg-sha256", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-oof", type=Path, required=True)
    args = parser.parse_args()

    baseline_sha = verify_sha256(
        args.baseline, args.expected_baseline_sha256, "P0 OOF"
    )
    specialist_sha = verify_sha256(
        args.specialist, args.expected_specialist_sha256, "P11 OOF"
    )
    contract = load_contract(args.prereg, args.expected_prereg_sha256)
    baseline = load_oof(args.baseline)
    specialist = load_oof(args.specialist)
    files = baseline["files"].astype(str)
    durations = np.asarray(
        [wav_duration_seconds(args.audio_dir / name) for name in files],
        dtype=np.float64,
    )
    result, output = evaluate_router(
        baseline, specialist, durations, contract
    )
    args.output_oof.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_oof, **output)
    output_oof_sha = sha256_file(args.output_oof)
    report = {
        "provenance": {
            "baseline_oof": str(args.baseline.resolve()),
            "baseline_oof_sha256": baseline_sha,
            "specialist_oof": str(args.specialist.resolve()),
            "specialist_oof_sha256": specialist_sha,
            "preregistration": str(args.prereg.resolve()),
            "preregistration_sha256": args.expected_prereg_sha256,
            "audio_dir": str(args.audio_dir.resolve()),
            "output_oof": str(args.output_oof.resolve()),
            "output_oof_sha256": output_oof_sha,
        },
        "evaluation": result,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
