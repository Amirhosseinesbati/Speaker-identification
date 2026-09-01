"""Evaluate the preregistered fixed ECAPA/CAM++ Fold-0 probability pair.

This evaluator is intentionally specific to P8.  It reads the immutable P8
preregistration, verifies the locked CAM++ comparator against the supplied OOF
file, and evaluates exactly one fusion: 50/50 probability averaging.  It does
not inherit the historical absolute-0.96 gate used by the older no-proto pair.
"""

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
from scripts.evaluate_fixed_oof_pair import (  # noqa: E402
    collapse_labels,
    load_oof,
    sha256_file,
    transition_bundle,
)


PROFILE = "p8-ecapa-frozen-known446-ood-complement-oof-f0"
FUSION_WEIGHTS = (0.5, 0.5)
EXPECTED_SPLIT = {"split_fold": 0, "split_folds": 3, "split_seed": 42}
LOCKED_METRIC_KEYS = {
    "macro_f1": "raw_macro_f1",
    "known_accuracy": "known_accuracy",
    "ood_f1": "ood_f1",
}


def load_preregistration(path: Path, expected_sha256: str) -> dict:
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "Preregistration SHA mismatch: "
            f"expected={expected_sha256}, actual={actual_sha256}"
        )
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("profile") != PROFILE:
        raise RuntimeError(
            f"Unexpected preregistration profile: {contract.get('profile')!r}"
        )
    gate = contract.get("gate")
    comparator = contract.get("locked_comparator")
    if not isinstance(gate, dict) or not isinstance(comparator, dict):
        raise RuntimeError("Preregistration lacks gate/locked_comparator objects")
    required_gate = {
        "standalone_min_raw_macro_f1",
        "fixed_50_50_min_macro_gain",
        "max_known_accuracy_drop",
        "max_ood_f1_drop",
        "min_campp_error_rescue_rate",
        "require_rescued_gt_introduced",
        "all_conditions_required",
    }
    missing_gate = required_gate - set(gate)
    if missing_gate:
        raise RuntimeError(f"Preregistration gate lacks: {sorted(missing_gate)}")
    if comparator.get("profile") != "p0-campp-known446-ood-control-oof-f0":
        raise RuntimeError("Unexpected locked CAM++ comparator profile")
    if gate["all_conditions_required"] is not True:
        raise RuntimeError("P8 requires all gate conditions")
    return contract


def _aligned_pair(campp: dict, ecapa: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    campp_files = campp["files"].astype(str)
    ecapa_files = ecapa["files"].astype(str)
    if set(campp_files.tolist()) != set(ecapa_files.tolist()):
        raise RuntimeError("CAM++ and ECAPA OOF file sets differ")
    ecapa_index = {name: position for position, name in enumerate(ecapa_files)}
    order = np.asarray([ecapa_index[name] for name in campp_files], dtype=np.int64)
    labels = collapse_labels(campp["labels"])
    ecapa_labels = collapse_labels(ecapa["labels"])[order]
    if not np.array_equal(labels, ecapa_labels):
        raise RuntimeError("CAM++ and ECAPA labels differ after file alignment")
    for key, expected in EXPECTED_SPLIT.items():
        observed_campp = int(np.asarray(campp[key]).reshape(-1)[0])
        observed_ecapa = int(np.asarray(ecapa[key]).reshape(-1)[0])
        if observed_campp != expected or observed_ecapa != expected:
            raise RuntimeError(
                f"Paired OOF {key} mismatch: expected={expected}, "
                f"campp={observed_campp}, ecapa={observed_ecapa}"
            )
    campp_probs = campp["competition_probs"].astype(np.float64)
    ecapa_probs = ecapa["competition_probs"].astype(np.float64)[order]
    return labels, campp_probs, ecapa_probs


def _assert_locked_campp_metrics(observed: dict, locked: dict) -> None:
    for metric_key, locked_key in LOCKED_METRIC_KEYS.items():
        expected = float(locked[locked_key])
        actual = float(observed[metric_key])
        if not np.isclose(actual, expected, rtol=0.0, atol=1e-12):
            raise RuntimeError(
                f"Locked CAM++ {metric_key} mismatch: "
                f"expected={expected}, observed={actual}"
            )


def evaluate_preregistered_pair(campp: dict, ecapa: dict, contract: dict) -> dict:
    labels, campp_probs, ecapa_probs = _aligned_pair(campp, ecapa)
    campp_predictions = campp_probs.argmax(axis=1).astype(np.int64)
    ecapa_predictions = ecapa_probs.argmax(axis=1).astype(np.int64)
    fused_probabilities = (
        FUSION_WEIGHTS[0] * campp_probs + FUSION_WEIGHTS[1] * ecapa_probs
    )
    fused_predictions = fused_probabilities.argmax(axis=1).astype(np.int64)

    campp_metrics = metric_bundle(labels, campp_predictions)
    ecapa_metrics = metric_bundle(labels, ecapa_predictions)
    fused_metrics = metric_bundle(labels, fused_predictions)
    _assert_locked_campp_metrics(campp_metrics, contract["locked_comparator"])

    fused_delta = metric_delta(fused_metrics, campp_metrics)
    transitions = transition_bundle(labels, campp_predictions, fused_predictions)
    gate = contract["gate"]
    checks = {
        "ecapa_standalone": (
            ecapa_metrics["macro_f1"]
            >= float(gate["standalone_min_raw_macro_f1"])
        ),
        "fixed_50_50_macro_gain_vs_campp": (
            fused_delta["macro_f1"]
            >= float(gate["fixed_50_50_min_macro_gain"])
        ),
        "known_accuracy_guardrail_vs_campp": (
            fused_delta["known_accuracy"]
            >= -float(gate["max_known_accuracy_drop"])
        ),
        "ood_f1_guardrail_vs_campp": (
            fused_delta["ood_f1"] >= -float(gate["max_ood_f1_drop"])
        ),
        "campp_error_rescue_rate": (
            transitions["rescue_rate"]
            >= float(gate["min_campp_error_rescue_rate"])
        ),
        "rescued_errors_exceed_introduced": (
            transitions["rescued_errors"] > transitions["introduced_errors"]
        ),
    }
    if gate["require_rescued_gt_introduced"] is not True:
        raise RuntimeError("P8 contract must require rescued > introduced")
    return {
        "rows": int(len(labels)),
        "metrics": {
            "campp_locked_control": campp_metrics,
            "ecapa_standalone": ecapa_metrics,
            "fixed_50_50": fused_metrics,
        },
        "fixed_50_50_delta_vs_campp": fused_delta,
        "fixed_50_50_error_transitions_vs_campp": transitions,
        "gate": {
            "contract": dict(gate),
            "checks": checks,
            "passed": all(checks.values()),
        },
    }


def _verify_file_sha256(path: Path, expected_sha256: str, label: str) -> str:
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"{label} SHA mismatch: expected={expected_sha256}, actual={actual_sha256}"
        )
    return actual_sha256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campp", type=Path, required=True)
    parser.add_argument("--expected-campp-sha256", required=True)
    parser.add_argument("--ecapa", type=Path, required=True)
    parser.add_argument("--expected-ecapa-sha256", required=True)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--expected-prereg-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    campp_sha256 = _verify_file_sha256(
        args.campp, args.expected_campp_sha256, "CAM++ OOF"
    )
    ecapa_sha256 = _verify_file_sha256(
        args.ecapa, args.expected_ecapa_sha256, "ECAPA OOF"
    )
    prereg_sha256 = _verify_file_sha256(
        args.prereg, args.expected_prereg_sha256, "preregistration"
    )
    contract = load_preregistration(args.prereg, prereg_sha256)
    result = evaluate_preregistered_pair(
        load_oof(args.campp), load_oof(args.ecapa), contract
    )
    report = {
        "contract": {
            "profile": PROFILE,
            "fusion_method": "probability_average",
            "fusion_weights": list(FUSION_WEIGHTS),
            "search_dimensions": 0,
            "legacy_absolute_macro_0_96_gate_applied": False,
            "leaderboard_used_for_selection": False,
            "single_fold_submission_authorized": False,
        },
        "provenance": {
            "campp_oof": str(args.campp.resolve()),
            "campp_oof_sha256": campp_sha256,
            "ecapa_oof": str(args.ecapa.resolve()),
            "ecapa_oof_sha256": ecapa_sha256,
            "preregistration": str(args.prereg.resolve()),
            "preregistration_sha256": prereg_sha256,
        },
        "evaluation": result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
