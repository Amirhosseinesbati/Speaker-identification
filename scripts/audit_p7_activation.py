"""Gate P7 activation on terminal P6 evidence, provenance and budget.

This script never starts a run.  It converts the preregistered ordering into a
machine-checkable decision so a rejected or incompletely recorded P6 pair
cannot silently authorize the dormant P7 OOD-head experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.campaign_state import CampaignStore  # noqa: E402


P6_CONTROL = (
    "p6-campp-known446-ood-crossfile-consistency-interclass-control-"
    "es80p20-oof-f0"
)
P6_TREATMENT = (
    "p6-campp-known446-ood-crossfile-consistency-interclass-e01-"
    "es80p20-oof-f0"
)
P6_CONFIG_SHA256 = {
    P6_CONTROL: "1f041abb3035af1f39de1714f389a6b9aadb04fb0a0c34e737c5ca363d92b3ab",
    P6_TREATMENT: "25dd1f1a52e4908c47e66b234de8e58671b82cb78fa528453a3865490f01ba5c",
}
P7_CONTROL_CONFIG = (
    ROOT / "configs" / "experiments" /
    "p7-campp-known446-ood-cleanaug-oodjsd-control-lmft-oof-f0.yaml"
)
P7_TREATMENT_CONFIG = (
    ROOT / "configs" / "experiments" /
    "p7-campp-known446-ood-cleanaug-oodjsd-w12-lmft-oof-f0.yaml"
)
P7_CONFIG_SHA256 = {
    "control": "474f1bb252561669fa92587864f095caec979a2e6ee212e64ec5846eea2faed9",
    "treatment": "ca881f67539936545347cd3dd28901fadc5974183f18a5d16e55e895199061ea",
}
P7_SOURCE_CHECKPOINT = (
    ROOT / "checkpoints" / "p0-campp-known446-ood-control-oof-f0" /
    "campp_best_raw.pt"
)
P7_SOURCE_SHA256 = (
    "f50f67f549b913b57111043b43daca1ff8bcbbf49bebe5dccab91ade8b19ae0d"
)
MINIMUM_REMAINING_USD = 1.10
MINIMUM_METRIC_KEYS = 33
MINIMUM_P6_EPOCHS = 80
MAXIMUM_P6_EPOCHS = 120
REQUIRED_TRACKING_ARTIFACTS = {
    "models/campp_best.pt",
    "models/campp_best_raw.pt",
    "models/campp_best_ema.pt",
    "models/campp_latest.pt",
    "bundle/class_map.json",
    "bundle/manifest.json",
    "bundle/metadata.json",
    "bundle/oof_predictions.npz",
    "bundle/resolved_config.yaml",
    "bundle/training_history.json",
    "bundle/unknown_clusters.json",
    "provenance/profile.yaml",
    "provenance/campaign.log",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha_present(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value.lower()
    )


def _split_is_fold0(split: Any) -> bool:
    return isinstance(split, dict) and split == {
        "scheme": "kfold", "folds": 3, "fold": 0, "seed": 42,
    }


def receipt_valid(receipt: dict[str, Any], profile: str) -> bool:
    canonical = receipt.get("canonical_checkpoint", {})
    oof = receipt.get("oof", {})
    history_points = int(canonical.get("history_points", -1))
    selected_epoch = int(canonical.get("selected_epoch", -1))
    artifacts = receipt.get("receipt_artifacts", [])
    return all([
        receipt.get("passed") is True,
        receipt.get("profile") == profile,
        receipt.get("config_sha256") == P6_CONFIG_SHA256[profile],
        int(receipt.get("exit_code", -1)) == 0,
        bool(artifacts),
        all(
            isinstance(item, dict)
            and isinstance(item.get("path"), str)
            and int(item.get("size_bytes", 0)) > 0
            and _sha_present(item.get("sha256"))
            for item in artifacts
        ),
        int(canonical.get("configured_epochs", -1)) == MAXIMUM_P6_EPOCHS,
        MINIMUM_P6_EPOCHS <= history_points <= MAXIMUM_P6_EPOCHS,
        1 <= selected_epoch <= history_points,
        canonical.get("weight_variant") == "raw",
        int(canonical.get("class_map_size", -1)) == 1001,
        int(canonical.get("competition_num_known", -1)) == 446,
        int(canonical.get("competition_class_count", -1)) == 447,
        _split_is_fold0(canonical.get("split")),
        _sha_present(oof.get("sha256")),
        int(oof.get("rows", -1)) == 1632,
        int(oof.get("unique_files", -1)) == 1632,
        int(oof.get("competition_classes", -1)) == 447,
        int(oof.get("embedding_dim", -1)) == 192,
        _split_is_fold0(oof.get("split")),
    ])


def tracking_valid(tracking: dict[str, Any], history_points: int) -> bool:
    series = tracking.get("metric_series", {})
    artifacts = set(tracking.get("artifact_paths", []))
    run_id = tracking.get("run_id")
    return all([
        isinstance(run_id, str) and len(run_id) == 32,
        tracking.get("status") == "FINISHED",
        int(tracking.get("metric_key_count", -1)) >= MINIMUM_METRIC_KEYS,
        len(series) >= MINIMUM_METRIC_KEYS,
        all(
            int(item.get("points", -1)) == history_points
            and int(item.get("finite_points", -1)) == history_points
            and int(item.get("min_step", -1)) == 1
            and int(item.get("max_step", -1)) == history_points
            for item in series.values()
        ),
        not (REQUIRED_TRACKING_ARTIFACTS - artifacts),
    ])


def backfill_valid(backfill: dict[str, Any], run_id: str) -> bool:
    return all([
        backfill.get("status") == "verified",
        backfill.get("remote_run_id") == run_id,
        backfill.get("remote_run_status_after") == "FINISHED",
        not backfill.get("missing_paths"),
        not backfill.get("size_mismatches"),
        not backfill.get("hash_mismatches"),
        not backfill.get("missing_params"),
        not backfill.get("missing_metrics"),
        backfill.get("hash_verification") == "sha256",
    ])


def p6_audit_valid(p6: dict[str, Any]) -> bool:
    provenance = p6.get("provenance", {})
    profiles = provenance.get("profiles", {})
    configured = provenance.get("config_sha256", {})
    profile_checks = []
    for profile in P6_CONFIG_SHA256:
        arm = profiles.get(profile, {})
        binding = arm.get("bundle_binding", {})
        profile_checks.extend([
            configured.get(profile) == P6_CONFIG_SHA256[profile],
            _sha_present(arm.get("checkpoint_sha256")),
            _sha_present(arm.get("latest_sha256")),
            _sha_present(arm.get("oof_sha256")),
            _sha_present(binding.get("manifest_sha256")),
            _sha_present(binding.get("raw_checkpoint_sha256")),
            _sha_present(binding.get("oof_predictions_sha256")),
            binding.get("weight_variant") == "raw",
        ])
    return all([
        p6.get("decision") == "reject",
        p6.get("acceptance_gate", {}).get("passed") is False,
        int(provenance.get("validation_files", -1)) == 1632,
        _sha_present(provenance.get("validation_file_sha256")),
        all(profile_checks),
    ])


def audit_activation(
    p6: dict[str, Any],
    receipts: dict[str, dict[str, Any]],
    tracking: dict[str, dict[str, Any]],
    backfills: dict[str, dict[str, Any]],
    campaign_state: dict[str, Any],
    budget: dict[str, float],
    p7_config_hashes: dict[str, str],
    source_checkpoint_hash: str,
) -> dict[str, Any]:
    receipt_checks = {
        profile: receipt_valid(receipts.get(profile, {}), profile)
        for profile in P6_CONFIG_SHA256
    }
    tracking_checks = {}
    backfill_checks = {}
    for profile in P6_CONFIG_SHA256:
        receipt = receipts.get(profile, {})
        points = int(
            receipt.get("canonical_checkpoint", {}).get("history_points", -1)
        )
        arm_tracking = tracking.get(profile, {})
        tracking_checks[profile] = tracking_valid(arm_tracking, points)
        backfill_checks[profile] = backfill_valid(
            backfills.get(profile, {}), str(arm_tracking.get("run_id", ""))
        )

    checks = {
        "p6_terminal_rejection_with_complete_audit": p6_audit_valid(p6),
        "p6_receipts_complete": all(receipt_checks.values()),
        "p6_mlflow_series_and_artifacts_complete": all(
            tracking_checks.values()
        ),
        "p6_mlflow_sha256_backfill_verified": all(backfill_checks.values()),
        "p7_config_hashes_locked": p7_config_hashes == P7_CONFIG_SHA256,
        "p7_source_checkpoint_locked": (
            source_checkpoint_hash == P7_SOURCE_SHA256
        ),
        "campaign_idle": campaign_state.get("current_run") is None,
        "pair_budget_available": (
            float(budget["remaining_usd"]) >= MINIMUM_REMAINING_USD
        ),
    }
    passed = all(checks.values())
    return {
        "decision": "activate_p7_control" if passed else "blocked",
        "passed": passed,
        "checks": checks,
        "arm_checks": {
            "receipts": receipt_checks,
            "tracking": tracking_checks,
            "backfills": backfill_checks,
        },
        "thresholds": {
            "minimum_remaining_usd": MINIMUM_REMAINING_USD,
            "minimum_p6_epochs": MINIMUM_P6_EPOCHS,
            "maximum_p6_epochs": MAXIMUM_P6_EPOCHS,
            "minimum_metric_keys": MINIMUM_METRIC_KEYS,
        },
        "evidence": {
            "p6_decision": p6.get("decision"),
            "campaign_status": campaign_state.get("status"),
            "estimated_cost_usd": budget["estimated_cost_usd"],
            "remaining_usd": budget["remaining_usd"],
            "p7_config_sha256": p7_config_hashes,
            "source_checkpoint_sha256": source_checkpoint_hash,
        },
    }


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p6-audit", type=Path, required=True)
    for arm in ("control", "treatment"):
        parser.add_argument(f"--{arm}-receipt", type=Path, required=True)
        parser.add_argument(f"--{arm}-tracking", type=Path, required=True)
        parser.add_argument(f"--{arm}-backfill", type=Path, required=True)
    parser.add_argument(
        "--source-checkpoint", type=Path, default=P7_SOURCE_CHECKPOINT
    )
    parser.add_argument(
        "--state", type=Path,
        default=ROOT / "data" / "experiments" / "campaign_state.json",
    )
    parser.add_argument(
        "--events", type=Path,
        default=ROOT / "data" / "experiments" / "campaign_events.jsonl",
    )
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    store = CampaignStore(args.state, args.events)
    state = store.load()
    report = audit_activation(
        _read(args.p6_audit),
        {P6_CONTROL: _read(args.control_receipt),
         P6_TREATMENT: _read(args.treatment_receipt)},
        {P6_CONTROL: _read(args.control_tracking),
         P6_TREATMENT: _read(args.treatment_tracking)},
        {P6_CONTROL: _read(args.control_backfill),
         P6_TREATMENT: _read(args.treatment_backfill)},
        state,
        store.budget_snapshot(state),
        {
            "control": sha256_file(P7_CONTROL_CONFIG),
            "treatment": sha256_file(P7_TREATMENT_CONFIG),
        },
        sha256_file(args.source_checkpoint),
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
