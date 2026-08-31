from __future__ import annotations

from copy import deepcopy

from scripts.audit_p7_activation import (
    P6_CONFIG_SHA256,
    P6_CONTROL,
    P6_TREATMENT,
    P7_CONFIG_SHA256,
    P7_SOURCE_SHA256,
    REQUIRED_TRACKING_ARTIFACTS,
    audit_activation,
)


def _fixture():
    sha = "a" * 64
    split = {"scheme": "kfold", "folds": 3, "fold": 0, "seed": 42}
    profiles = {}
    receipts = {}
    tracking = {}
    backfills = {}
    for index, profile in enumerate((P6_CONTROL, P6_TREATMENT), start=1):
        profiles[profile] = {
            "checkpoint_sha256": sha,
            "latest_sha256": sha,
            "oof_sha256": sha,
            "bundle_binding": {
                "manifest_sha256": sha,
                "raw_checkpoint_sha256": sha,
                "oof_predictions_sha256": sha,
                "weight_variant": "raw",
            },
        }
        receipts[profile] = {
            "passed": True,
            "profile": profile,
            "config_sha256": P6_CONFIG_SHA256[profile],
            "exit_code": 0,
            "receipt_artifacts": [
                {"path": "model.pt", "size_bytes": 123, "sha256": sha}
            ],
            "canonical_checkpoint": {
                "configured_epochs": 120,
                "history_points": 100,
                "selected_epoch": 87,
                "weight_variant": "raw",
                "class_map_size": 1001,
                "competition_num_known": 446,
                "competition_class_count": 447,
                "split": split,
            },
            "oof": {
                "sha256": sha,
                "rows": 1632,
                "unique_files": 1632,
                "competition_classes": 447,
                "embedding_dim": 192,
                "split": split,
            },
        }
        run_id = str(index) * 32
        metric_series = {
            f"metric_{metric}": {
                "points": 100,
                "finite_points": 100,
                "min_step": 1,
                "max_step": 100,
            }
            for metric in range(33)
        }
        tracking[profile] = {
            "run_id": run_id,
            "status": "FINISHED",
            "metric_key_count": 33,
            "metric_series": metric_series,
            "artifact_paths": sorted(REQUIRED_TRACKING_ARTIFACTS),
        }
        backfills[profile] = {
            "status": "verified",
            "remote_run_id": run_id,
            "remote_run_status_after": "FINISHED",
            "missing_paths": [],
            "size_mismatches": {},
            "hash_mismatches": {},
            "missing_params": [],
            "missing_metrics": [],
            "hash_verification": "sha256",
        }
    p6 = {
        "decision": "reject",
        "acceptance_gate": {"passed": False},
        "provenance": {
            "config_sha256": dict(P6_CONFIG_SHA256),
            "validation_files": 1632,
            "validation_file_sha256": sha,
            "profiles": profiles,
        },
    }
    campaign = {"status": "CAMPAIGN_BLOCKED", "current_run": None}
    budget = {"estimated_cost_usd": 18.8, "remaining_usd": 1.2}
    return p6, receipts, tracking, backfills, campaign, budget


def test_complete_terminal_p6_rejection_activates_p7_control() -> None:
    args = _fixture()
    report = audit_activation(
        *args, deepcopy(P7_CONFIG_SHA256), P7_SOURCE_SHA256
    )
    assert report["passed"] is True
    assert report["decision"] == "activate_p7_control"
    assert all(report["checks"].values())


def test_p6_acceptance_or_active_campaign_blocks_p7() -> None:
    p6, receipts, tracking, backfills, campaign, budget = _fixture()
    p6["decision"] = "accept"
    p6["acceptance_gate"]["passed"] = True
    campaign["current_run"] = {"profile": P6_CONTROL}
    report = audit_activation(
        p6, receipts, tracking, backfills, campaign, budget,
        deepcopy(P7_CONFIG_SHA256), P7_SOURCE_SHA256,
    )
    assert report["passed"] is False
    assert report["checks"]["p6_terminal_rejection_with_complete_audit"] is False
    assert report["checks"]["campaign_idle"] is False


def test_incomplete_receipt_tracking_or_backfill_blocks_p7() -> None:
    p6, receipts, tracking, backfills, campaign, budget = _fixture()
    receipts[P6_CONTROL]["canonical_checkpoint"]["history_points"] = 79
    tracking[P6_TREATMENT]["artifact_paths"].remove(
        "bundle/oof_predictions.npz"
    )
    backfills[P6_CONTROL]["hash_mismatches"] = {"model.pt": "bad"}
    report = audit_activation(
        p6, receipts, tracking, backfills, campaign, budget,
        deepcopy(P7_CONFIG_SHA256), P7_SOURCE_SHA256,
    )
    assert report["passed"] is False
    assert report["checks"]["p6_receipts_complete"] is False
    assert report["checks"]["p6_mlflow_series_and_artifacts_complete"] is False
    assert report["checks"]["p6_mlflow_sha256_backfill_verified"] is False


def test_hash_or_budget_drift_blocks_p7() -> None:
    p6, receipts, tracking, backfills, campaign, budget = _fixture()
    config_hashes = deepcopy(P7_CONFIG_SHA256)
    config_hashes["treatment"] = "b" * 64
    budget["remaining_usd"] = 1.09
    report = audit_activation(
        p6, receipts, tracking, backfills, campaign, budget,
        config_hashes, "c" * 64,
    )
    assert report["passed"] is False
    assert report["checks"]["p7_config_hashes_locked"] is False
    assert report["checks"]["p7_source_checkpoint_locked"] is False
    assert report["checks"]["pair_budget_available"] is False
