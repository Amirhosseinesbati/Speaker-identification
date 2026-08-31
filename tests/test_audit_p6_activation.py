from __future__ import annotations

from scripts.audit_p6_activation import (
    EXPECTED_MLFLOW_RUN_ID,
    REQUIRED_MLFLOW_ARTIFACTS,
    audit_activation,
)


def _fixture() -> tuple[dict, dict, dict, dict, dict]:
    sha = "a" * 64
    binding = {
        "manifest_sha256": sha,
        "raw_checkpoint_sha256": sha,
        "oof_predictions_sha256": sha,
        "weight_variant": "raw",
    }
    arm = {
        "checkpoint_sha256": sha,
        "latest_sha256": sha,
        "oof_sha256": sha,
        "bundle_binding": binding,
    }
    p5 = {
        "decision": "reject",
        "matched_control": {"known_accuracy": 0.95, "ood_f1": 0.96},
        "treatment": {
            "metrics": {"known_accuracy": 0.951, "ood_f1": 0.961}
        },
        "collapse_guard": {"treatment_over_matched": 0.999},
        "provenance": {"matched": arm, "treatment": arm},
    }
    series = {
        f"metric_{index}": {
            "points": 104,
            "finite_points": 104,
            "min_step": 1,
            "max_step": 104,
        }
        for index in range(33)
    }
    mlflow = {
        "run_id": EXPECTED_MLFLOW_RUN_ID,
        "status": "FINISHED",
        "metric_key_count": 33,
        "metric_series": series,
        "artifact_paths": sorted(REQUIRED_MLFLOW_ARTIFACTS),
    }
    backfill = {
        "status": "verified",
        "remote_run_id": EXPECTED_MLFLOW_RUN_ID,
        "remote_run_status_after": "FINISHED",
        "missing_paths": [],
        "size_mismatches": {},
        "hash_mismatches": {},
        "missing_params": [],
        "missing_metrics": [],
        "hash_verification": "sha256",
    }
    campaign = {"status": "CAMPAIGN_BLOCKED", "current_run": None}
    budget = {"estimated_cost_usd": 15.0, "remaining_usd": 5.0}
    return p5, mlflow, backfill, campaign, budget


def test_activation_requires_every_preregistered_gate() -> None:
    values = _fixture()
    report = audit_activation(*values)
    assert report["passed"] is True
    assert report["decision"] == "activate_p6_control"
    assert all(report["checks"].values())


def test_activation_blocks_missing_remote_oof() -> None:
    p5, mlflow, backfill, campaign, budget = _fixture()
    mlflow["artifact_paths"].remove("bundle/oof_predictions.npz")
    report = audit_activation(p5, mlflow, backfill, campaign, budget)
    assert report["passed"] is False
    assert report["checks"]["mlflow_artifacts_complete"] is False
    assert report["evidence"]["missing_mlflow_artifacts"] == [
        "bundle/oof_predictions.npz"
    ]


def test_activation_blocks_budget_or_guardrail_failure() -> None:
    p5, mlflow, backfill, campaign, budget = _fixture()
    p5["treatment"]["metrics"]["known_accuracy"] = 0.948
    budget["remaining_usd"] = 2.79
    report = audit_activation(p5, mlflow, backfill, campaign, budget)
    assert report["passed"] is False
    assert report["checks"]["known_guardrail"] is False
    assert report["checks"]["pair_budget_available"] is False
