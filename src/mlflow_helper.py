"""
MLflow Helper — standalone MLflow tracking independent of ZenML.

ZenML's experiment tracker integration is fragile across versions.
This module provides direct MLflow tracking that works reliably
on any environment (local, Vast.ai, Docker).

Usage:
    from src.mlflow_helper import MLflowTracker

    tracker = MLflowTracker(config)
    tracker.start_run(run_name="training_ecapa")

    # Log params, metrics, artifacts
    tracker.log_params({"lr": 1e-4, "epochs": 20})
    tracker.log_metrics({"val_loss": 0.5}, step=1)
    tracker.log_artifact("checkpoints/best_model.pt")
    tracker.log_code_snapshot()  # zip of src/

    tracker.end_run()
"""

import os
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import mlflow
import yaml


class MLflowTracker:
    """
    Standalone MLflow tracker that works without ZenML.

    Handles:
    - Tracking URI configuration from config + env vars
    - DagsHub authentication
    - Run lifecycle (start/end)
    - Param/metric/artifact logging
    - Code snapshot (zip of src/ directory)
    """

    def __init__(self, config: Optional[dict] = None):
        """
        Args:
            config: Project config dict (from default_config.yaml)
                    If None, reads from default path.
        """
        if config is None:
            config_path = "configs/default_config.yaml"
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

        self.config = config
        self._run: Optional[mlflow.ActiveRun] = None
        self._setup()

    def _setup(self):
        """Configure MLflow tracking URI and authentication."""
        mlops_cfg = self.config.get("mlops", {})
        tracking_cfg = mlops_cfg.get("tracking", {})

        # Resolve tracking URI
        tracking_uri = (
            os.getenv("MLFLOW_TRACKING_URI")
            or tracking_cfg.get("uri", "")
        )
        # Resolve env var placeholders in URI
        for var in ["DAGSHUB_REPO_OWNER", "DAGSHUB_USERNAME",
                     "DAGSHUB_USER_TOKEN", "DAGSHUB_TOKEN"]:
            val = os.getenv(var, "")
            tracking_uri = tracking_uri.replace(f"${{{var}}}", val)

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            print(f"  📊 MLflow URI: {tracking_uri}")

        # Authenticate
        dagshub_user = (os.getenv("DAGSHUB_REPO_OWNER")
                        or os.getenv("DAGSHUB_USERNAME", ""))
        dagshub_token = (os.getenv("DAGSHUB_USER_TOKEN")
                         or os.getenv("DAGSHUB_TOKEN", ""))

        if dagshub_user and dagshub_token:
            os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_user
            os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        # Set experiment
        experiment_name = mlops_cfg.get("experiment_name", "speaker-identification")
        try:
            mlflow.set_experiment(experiment_name)
            print(f"  📊 MLflow experiment: {experiment_name}")
        except Exception as e:
            print(f"  ⚠ Could not set MLflow experiment: {e}")

    # ═══════════════════════════════════════════════════════
    #  Run Lifecycle
    # ═══════════════════════════════════════════════════════

    def start_run(self, run_name: Optional[str] = None) -> mlflow.ActiveRun:
        """
        Start a new MLflow run.

        Args:
            run_name: Human-readable name for this run.
                      Defaults to "run-YYYYMMDD-HHMMSS".

        Returns:
            Active MLflow run
        """
        if run_name is None:
            run_name = f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        # End any existing run
        if mlflow.active_run():
            mlflow.end_run()

        try:
            self._run = mlflow.start_run(run_name=run_name)
            print(f"\n  🚀 MLflow run started: {run_name}")
        except Exception as e:
            # A bad token / unreachable tracker (e.g. DagsHub 403) must not kill
            # the training step — degrade to running WITHOUT tracking instead.
            self._run = None
            print(f"\n  ⚠ MLflow run start failed ({e}) — running WITHOUT tracking. "
                  f"Check your DagsHub/MLflow credentials in .env.")
        return self._run

    def end_run(self):
        """End the current MLflow run."""
        if mlflow.active_run():
            mlflow.end_run()
            print(f"  ✅ MLflow run ended.")
        self._run = None

    @property
    def is_active(self) -> bool:
        """Check if a run is currently active."""
        return mlflow.active_run() is not None

    # ═══════════════════════════════════════════════════════
    #  Logging
    # ═══════════════════════════════════════════════════════

    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters."""
        if self.is_active:
            mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics (optionally with step number)."""
        if self.is_active:
            mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log a file or directory as MLflow artifact."""
        if self.is_active and os.path.exists(local_path):
            mlflow.log_artifact(local_path, artifact_path=artifact_path)

    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        """Log all files in a directory as artifacts."""
        if self.is_active and os.path.isdir(local_dir):
            mlflow.log_artifacts(local_dir, artifact_path=artifact_path)

    def log_dict(self, data: dict, artifact_file: str):
        """Log a dict as JSON artifact."""
        if self.is_active:
            mlflow.log_dict(data, artifact_file)

    def log_model(self, model, artifact_path: str = "model"):
        """Log a PyTorch model."""
        if self.is_active:
            try:
                mlflow.pytorch.log_model(model, artifact_path=artifact_path)
            except Exception as e:
                print(f"  ⚠ Could not log PyTorch model: {e}")

    # ═══════════════════════════════════════════════════════
    #  Code Snapshot
    # ═══════════════════════════════════════════════════════

    def log_code_snapshot(self):
        """
        Create a zip snapshot of the src/ directory and log it as artifact.

        This captures the exact code used for training,
        which is critical for reproducibility.
        """
        if not self.is_active:
            return

        src_dir = Path("src")
        if not src_dir.exists():
            print("  ⚠ src/ directory not found, skipping code snapshot.")
            return

        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            zip_path = tmp.name

        try:
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for file_path in src_dir.rglob("*.py"):
                    # Use relative path within the zip
                    arcname = str(file_path)
                    zf.write(file_path, arcname=arcname)

            mlflow.log_artifact(zip_path, artifact_path="code")
            print(f"  📦 Code snapshot logged ({src_dir})")
        finally:
            os.unlink(zip_path)

    def log_config_snapshot(self):
        """Log the current config YAML as artifact."""
        if not self.is_active:
            return

        config_path = Path("configs/default_config.yaml")
        if config_path.exists():
            mlflow.log_artifact(str(config_path), artifact_path="code")
            print(f"  📦 Config snapshot logged")

    def log_summary(self, summary: dict, metrics: dict):
        """
        Log a comprehensive training summary.

        Args:
            summary: Training stats (best_epoch, best_val_loss, etc.)
            metrics: Final evaluation metrics
        """
        if not self.is_active:
            return

        # Log as params (immutable, best values)
        mlflow.log_params({f"best_{k}": v for k, v in summary.items()
                           if not isinstance(v, (list, dict))})

        # Log as metrics (for comparison across runs)
        mlflow.log_metrics(metrics)

        # Log combined dict as JSON artifact
        full_summary = {"training_summary": summary, "evaluation_metrics": metrics}
        mlflow.log_dict(full_summary, "training_summary.json")

    def log_best_checkpoint(self, checkpoint_path: str = "checkpoints/best_model.pt"):
        """Log the best model checkpoint."""
        if self.is_active and os.path.exists(checkpoint_path):
            mlflow.log_artifact(checkpoint_path, artifact_path="models")
            print(f"  📦 Best checkpoint logged: {checkpoint_path}")


# ═══════════════════════════════════════════════════════════
#  Singleton helper
# ═══════════════════════════════════════════════════════════

_tracker: Optional[MLflowTracker] = None


def get_tracker(config: Optional[dict] = None) -> MLflowTracker:
    """Get or create the global MLflow tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = MLflowTracker(config)
    return _tracker


def reset_tracker():
    """Reset the global tracker (for testing)."""
    global _tracker
    if _tracker is not None:
        _tracker.end_run()
    _tracker = None


# ═══════════════════════════════════════════════════════════
#  Smoke Test
# ═══════════════════════════════════════════════════════════

def _smoke_test():
    """Quick test of MLflowTracker (logs locally)."""
    print("=" * 50)
    print("  MLflow Tracker Smoke Test")
    print("=" * 50)

    config = {
        "mlops": {
            "experiment_name": "test-experiment",
            "tracking": {"uri": ""},
        }
    }

    # Use local tracking for test
    mlflow.set_tracking_uri("")
    tracker = MLflowTracker(config)
    tracker.start_run(run_name="smoke-test")

    tracker.log_params({"lr": 0.001, "epochs": 5})
    tracker.log_metrics({"loss": 0.5, "acc": 0.9}, step=1)
    tracker.log_metrics({"loss": 0.3, "acc": 0.95}, step=2)

    # Test code snapshot
    tracker.log_code_snapshot()
    tracker.log_config_snapshot()
    tracker.log_summary(
        {"best_epoch": 3, "best_val_loss": 0.3},
        {"final_val_loss": 0.31, "final_ood_acc": 0.85},
    )

    # Verify run has artifacts
    run = mlflow.active_run()
    assert run is not None
    print(f"  Run ID: {run.info.run_id}")
    print(f"  Run name: {run.info.run_name}")

    tracker.end_run()

    print()
    print("  ALL MLFLOW TRACKER TESTS PASSED ✅")


if __name__ == "__main__":
    _smoke_test()
