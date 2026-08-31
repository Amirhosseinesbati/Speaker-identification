"""Apply the preregistered P6 activation gate to preserved P5 evidence."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.campaign_state import CampaignStore  # noqa: E402


PAIR_CAP_USD = 2.80
MINIMUM_SPREAD_RATIO = 0.95
MAXIMUM_KNOWN_OR_OOD_DROP = 0.001
EXPECTED_P5_EPOCHS = 104
EXPECTED_P5_METRIC_KEYS = 33
EXPECTED_MLFLOW_RUN_ID = "481c3e23724546078c64c5be7d884d69"
EXPECTED_P5_APPEND_ARTIFACT_COUNT = 15
EXPECTED_P5_APPEND_BYTES = 284_943_767
REQUIRED_MLFLOW_ARTIFACTS = {
    "models/campp_best.pt",
    "models/campp_best_raw.pt",
    "models/campp_best_ema.pt",
    "models/campp_latest.pt",
    "bundle/MODEL_CARD.md",
    "bundle/class_map.json",
    "bundle/manifest.json",
    "bundle/metadata.json",
    "bundle/oof_predictions.npz",
    "bundle/resolved_config.yaml",
    "bundle/training_history.json",
    "bundle/unknown_clusters.json",
    "provenance/profile.yaml",
    "provenance/campaign.log",
    "analysis/p5_terminal104_audit.json",
    "provenance/p5_terminal_artifact_manifest.json",
}


def _recovery_summary_verified(summary: dict[str, Any]) -> bool:
    """Accept either the legacy backfill or immutable-append receipt.

    Both utilities perform remote SHA-256 verification.  The append receipt is
    additionally pinned to the exact P5 payload and unique manifest path so a
    verified receipt for an unrelated analysis cannot activate P6.
    """

    legacy_verified = (
        summary.get("status") == "verified"
        and summary.get("remote_run_id") == EXPECTED_MLFLOW_RUN_ID
        and summary.get("remote_run_status_after") == "FINISHED"
        and not summary.get("missing_paths")
        and not summary.get("size_mismatches")
        and not summary.get("hash_mismatches")
        and not summary.get("missing_params")
        and not summary.get("missing_metrics")
        and summary.get("hash_verification") == "sha256"
    )
    if legacy_verified:
        return True

    direct_artifacts = REQUIRED_MLFLOW_ARTIFACTS - {
        "provenance/p5_terminal_artifact_manifest.json"
    }
    verified_direct_paths = set(summary.get("uploaded", [])) | set(
        summary.get("already_identical", [])
    )
    return (
        summary.get("status") == "verified"
        and summary.get("remote_run_id") == EXPECTED_MLFLOW_RUN_ID
        and summary.get("remote_run_status") == "FINISHED"
        and int(summary.get("artifact_count", -1))
        == EXPECTED_P5_APPEND_ARTIFACT_COUNT
        and int(summary.get("artifact_bytes", -1)) == EXPECTED_P5_APPEND_BYTES
        and verified_direct_paths == direct_artifacts
        and summary.get("manifest_remote_path")
        == "provenance/p5_terminal_artifact_manifest.json"
        and _sha_present(summary.get("manifest_sha256"))
        and not summary.get("hash_mismatches")
    )


def _sha_present(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value.lower()
    )


def audit_activation(
    p5: dict[str, Any],
    mlflow: dict[str, Any],
    backfill: dict[str, Any],
    campaign_state: dict[str, Any],
    budget: dict[str, float],
) -> dict[str, Any]:
    matched = p5["matched_control"]
    treatment = p5["treatment"]["metrics"]
    spread_ratio = float(p5["collapse_guard"]["treatment_over_matched"])
    known_delta = float(treatment["known_accuracy"]) - float(
        matched["known_accuracy"]
    )
    ood_delta = float(treatment["ood_f1"]) - float(matched["ood_f1"])

    provenance = p5["provenance"]
    binding_checks = []
    for arm in ("matched", "treatment"):
        arm_provenance = provenance[arm]
        binding = arm_provenance["bundle_binding"]
        binding_checks.extend(
            [
                _sha_present(arm_provenance.get("checkpoint_sha256")),
                _sha_present(arm_provenance.get("latest_sha256")),
                _sha_present(arm_provenance.get("oof_sha256")),
                _sha_present(binding.get("manifest_sha256")),
                _sha_present(binding.get("raw_checkpoint_sha256")),
                _sha_present(binding.get("oof_predictions_sha256")),
                binding.get("weight_variant") == "raw",
            ]
        )

    metric_series = mlflow.get("metric_series", {})
    metrics_complete = (
        mlflow.get("run_id") == EXPECTED_MLFLOW_RUN_ID
        and mlflow.get("status") == "FINISHED"
        and int(mlflow.get("metric_key_count", -1)) == EXPECTED_P5_METRIC_KEYS
        and len(metric_series) == EXPECTED_P5_METRIC_KEYS
        and all(
            int(series.get("points", -1)) == EXPECTED_P5_EPOCHS
            and int(series.get("finite_points", -1)) == EXPECTED_P5_EPOCHS
            and int(series.get("min_step", -1)) == 1
            and int(series.get("max_step", -1)) == EXPECTED_P5_EPOCHS
            for series in metric_series.values()
        )
    )
    artifact_paths = set(mlflow.get("artifact_paths", []))
    missing_artifacts = sorted(REQUIRED_MLFLOW_ARTIFACTS - artifact_paths)
    backfill_verified = _recovery_summary_verified(backfill)

    checks = {
        "p5_score_gate_failed": p5.get("decision") == "reject",
        "embedding_spread_guardrail": spread_ratio >= MINIMUM_SPREAD_RATIO,
        "known_guardrail": known_delta >= -MAXIMUM_KNOWN_OR_OOD_DROP,
        "ood_guardrail": ood_delta >= -MAXIMUM_KNOWN_OR_OOD_DROP,
        "local_bundle_and_oof_provenance": all(binding_checks),
        "mlflow_metrics_complete": metrics_complete,
        "mlflow_artifacts_complete": not missing_artifacts,
        "mlflow_backfill_hash_verified": backfill_verified,
        "campaign_idle": campaign_state.get("current_run") is None,
        "pair_budget_available": float(budget["remaining_usd"]) >= PAIR_CAP_USD,
    }
    passed = all(checks.values())
    return {
        "decision": "activate_p6_control" if passed else "blocked",
        "passed": passed,
        "checks": checks,
        "thresholds": {
            "minimum_embedding_spread_ratio": MINIMUM_SPREAD_RATIO,
            "maximum_known_or_ood_drop": MAXIMUM_KNOWN_OR_OOD_DROP,
            "required_metric_keys": EXPECTED_P5_METRIC_KEYS,
            "required_points_per_metric": EXPECTED_P5_EPOCHS,
            "required_pair_budget_usd": PAIR_CAP_USD,
        },
        "evidence": {
            "p5_decision": p5.get("decision"),
            "embedding_spread_ratio": spread_ratio,
            "known_delta_vs_matched": known_delta,
            "ood_delta_vs_matched": ood_delta,
            "mlflow_run_id": mlflow.get("run_id"),
            "mlflow_status": mlflow.get("status"),
            "mlflow_metric_keys": mlflow.get("metric_key_count"),
            "missing_mlflow_artifacts": missing_artifacts,
            "backfill_status": backfill.get("status"),
            "campaign_status": campaign_state.get("status"),
            "estimated_cost_usd": budget["estimated_cost_usd"],
            "remaining_usd": budget["remaining_usd"],
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p5-audit", type=Path, required=True)
    parser.add_argument("--mlflow-audit", type=Path, required=True)
    parser.add_argument("--backfill-summary", type=Path, required=True)
    parser.add_argument(
        "--state",
        type=Path,
        default=ROOT / "data" / "experiments" / "campaign_state.json",
    )
    parser.add_argument(
        "--events",
        type=Path,
        default=ROOT / "data" / "experiments" / "campaign_events.jsonl",
    )
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    store = CampaignStore(args.state, args.events)
    campaign_state = store.load()
    report = audit_activation(
        json.loads(args.p5_audit.read_text(encoding="utf-8")),
        json.loads(args.mlflow_audit.read_text(encoding="utf-8")),
        json.loads(args.backfill_summary.read_text(encoding="utf-8")),
        campaign_state,
        store.budget_snapshot(campaign_state),
    )
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
