"""Apply the locked P7 Fold-0 score, mechanism and provenance gates.

The input evidence must already be produced from aligned immutable Fold-0 OOF
artifacts.  This module performs no search and accepts only the preregistered
Raw probability-average/LME20 decision plus one fixed 50/50 fusion.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any


MINIMUM_MACRO_GAIN = 0.002
MAXIMUM_GUARDRAIL_DROP = 0.001
MINIMUM_SUPPORTING_EPOCHS = 2
MAXIMUM_KNOWN_TO_UNKNOWN = 19
MAXIMUM_GAP_RATIO = 0.80
FIXED_FUSION_WEIGHT = 0.5


def _finite_metric_bundle(bundle: dict[str, Any]) -> bool:
    required = ("macro_f1", "known_accuracy", "ood_f1")
    return all(
        isinstance(bundle.get(key), (int, float))
        and not isinstance(bundle.get(key), bool)
        and math.isfinite(float(bundle[key]))
        for key in required
    )


def _integer(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def acceptance_gate(evidence: dict[str, Any]) -> dict[str, Any]:
    control = evidence.get("matched_control", {})
    external = evidence.get("immutable_external_control", {})
    treatment = evidence.get("treatment", {})
    fusion = evidence.get("fixed_fusion", {})
    metric_inputs_valid = all(
        _finite_metric_bundle(bundle)
        for bundle in (control, external, treatment, fusion)
    )

    if metric_inputs_valid:
        treatment_gain_control = (
            float(treatment["macro_f1"]) - float(control["macro_f1"])
        )
        treatment_gain_external = (
            float(treatment["macro_f1"]) - float(external["macro_f1"])
        )
        fusion_gain_external = (
            float(fusion["macro_f1"]) - float(external["macro_f1"])
        )
        treatment_known_deltas = [
            float(treatment["known_accuracy"])
            - float(reference["known_accuracy"])
            for reference in (control, external)
        ]
        treatment_ood_deltas = [
            float(treatment["ood_f1"]) - float(reference["ood_f1"])
            for reference in (control, external)
        ]
        fusion_known_delta = (
            float(fusion["known_accuracy"])
            - float(external["known_accuracy"])
        )
        fusion_ood_delta = (
            float(fusion["ood_f1"]) - float(external["ood_f1"])
        )
    else:
        treatment_gain_control = float("nan")
        treatment_gain_external = float("nan")
        fusion_gain_external = float("nan")
        treatment_known_deltas = [float("nan"), float("nan")]
        treatment_ood_deltas = [float("nan"), float("nan")]
        fusion_known_delta = float("nan")
        fusion_ood_delta = float("nan")

    supporting_epochs = evidence.get("supporting_epochs", [])
    valid_support = (
        isinstance(supporting_epochs, list)
        and all(isinstance(epoch, int) and epoch > 0 for epoch in supporting_epochs)
    )
    unique_supporting_epochs = (
        sorted(set(supporting_epochs)) if valid_support else []
    )
    external_errors = evidence.get("immutable_external_errors", {})
    treatment_errors = evidence.get("treatment_errors", {})
    external_known_to_unknown = _integer(
        external_errors.get("known_to_unknown")
    )
    external_unknown_to_known = _integer(
        external_errors.get("unknown_to_known")
    )
    treatment_known_to_unknown = _integer(
        treatment_errors.get("known_to_unknown")
    )
    treatment_unknown_to_known = _integer(
        treatment_errors.get("unknown_to_known")
    )
    gap_ratio = evidence.get("clean_aug_ood_gap_ratio")
    gap_ratio_valid = (
        isinstance(gap_ratio, (int, float))
        and not isinstance(gap_ratio, bool)
        and math.isfinite(float(gap_ratio))
        and float(gap_ratio) >= 0.0
    )
    fusion_weight = evidence.get("fusion_weight")
    evaluated_weights = evidence.get("evaluated_fusion_weights")
    fixed_fusion_contract = (
        isinstance(fusion_weight, (int, float))
        and not isinstance(fusion_weight, bool)
        and float(fusion_weight) == FIXED_FUSION_WEIGHT
        and evaluated_weights == [FIXED_FUSION_WEIGHT]
    )

    checks = {
        "finite_metric_inputs": metric_inputs_valid,
        "treatment_gain_vs_matched_control": (
            metric_inputs_valid
            and treatment_gain_control >= MINIMUM_MACRO_GAIN
        ),
        "treatment_gain_vs_immutable_control": (
            metric_inputs_valid
            and treatment_gain_external >= MINIMUM_MACRO_GAIN
        ),
        "gain_supported_by_two_epochs": (
            valid_support
            and len(unique_supporting_epochs) >= MINIMUM_SUPPORTING_EPOCHS
        ),
        "treatment_known_guardrail_vs_both_controls": (
            metric_inputs_valid
            and min(treatment_known_deltas) >= -MAXIMUM_GUARDRAIL_DROP
        ),
        "treatment_ood_guardrail_vs_both_controls": (
            metric_inputs_valid
            and min(treatment_ood_deltas) >= -MAXIMUM_GUARDRAIL_DROP
        ),
        "known_to_unknown_reduced_by_twenty_percent": (
            external_known_to_unknown == 24
            and treatment_known_to_unknown is not None
            and 0 <= treatment_known_to_unknown <= MAXIMUM_KNOWN_TO_UNKNOWN
        ),
        "unknown_to_known_not_increased": (
            external_unknown_to_known == 11
            and treatment_unknown_to_known is not None
            and 0 <= treatment_unknown_to_known <= 11
        ),
        "clean_aug_gap_ratio": (
            gap_ratio_valid and float(gap_ratio) <= MAXIMUM_GAP_RATIO
        ),
        "speaker_logits_byte_identical": (
            evidence.get("speaker_logits_byte_identical") is True
        ),
        "fixed_fusion_contract": fixed_fusion_contract,
        "fixed_fusion_gain": (
            metric_inputs_valid and fusion_gain_external >= MINIMUM_MACRO_GAIN
        ),
        "fixed_fusion_known_guardrail": (
            metric_inputs_valid
            and fusion_known_delta >= -MAXIMUM_GUARDRAIL_DROP
        ),
        "fixed_fusion_ood_guardrail": (
            metric_inputs_valid
            and fusion_ood_delta >= -MAXIMUM_GUARDRAIL_DROP
        ),
        "provenance_complete": evidence.get("provenance_complete") is True,
        "raw_lme20_direct_argmax": (
            evidence.get("decision_policy")
            == "raw_probability_average_lme20_direct_argmax"
        ),
        "leaderboard_tuning_absent": (
            evidence.get("leaderboard_tuning") is False
        ),
    }
    passed = all(checks.values())
    return {
        "decision": "accept_for_replication" if passed else "reject",
        "passed": passed,
        "checks": checks,
        "thresholds": {
            "minimum_macro_gain": MINIMUM_MACRO_GAIN,
            "maximum_guardrail_drop": MAXIMUM_GUARDRAIL_DROP,
            "minimum_supporting_epochs": MINIMUM_SUPPORTING_EPOCHS,
            "maximum_known_to_unknown": MAXIMUM_KNOWN_TO_UNKNOWN,
            "maximum_clean_aug_gap_ratio": MAXIMUM_GAP_RATIO,
            "fixed_fusion_weight": FIXED_FUSION_WEIGHT,
        },
        "deltas": {
            "treatment_macro_vs_matched_control": treatment_gain_control,
            "treatment_macro_vs_immutable_control": treatment_gain_external,
            "treatment_known_vs_controls": treatment_known_deltas,
            "treatment_ood_vs_controls": treatment_ood_deltas,
            "fixed_fusion_macro_vs_immutable_control": fusion_gain_external,
            "fixed_fusion_known_vs_immutable_control": fusion_known_delta,
            "fixed_fusion_ood_vs_immutable_control": fusion_ood_delta,
        },
        "supporting_epochs": unique_supporting_epochs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    report = acceptance_gate(evidence)
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, args.output)
    print(encoded, end="")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
