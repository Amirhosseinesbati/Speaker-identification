"""Inspect a remote MLflow run without exposing credentials or mutating it."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from mlflow.tracking import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mlflow_backfill_run import _remote_artifacts, load_tracking_environment


def audit_run(client: MlflowClient, run_id: str) -> dict[str, Any]:
    run = client.get_run(run_id)
    artifacts = _remote_artifacts(client, run_id)
    metric_series: dict[str, Any] = {}
    for key in sorted(run.data.metrics):
        history = client.get_metric_history(run_id, key)
        finite = [item for item in history if math.isfinite(float(item.value))]
        metric_series[key] = {
            "points": len(history),
            "finite_points": len(finite),
            "min_step": min((int(item.step) for item in history), default=None),
            "max_step": max((int(item.step) for item in history), default=None),
            "last_value": float(history[-1].value) if history else None,
        }

    artifact_paths = sorted(artifacts)
    return {
        "run_id": run.info.run_id,
        "status": run.info.status,
        "run_name": run.data.tags.get("mlflow.runName"),
        "start_time": run.info.start_time,
        "end_time": run.info.end_time,
        "parameter_count": len(run.data.params),
        "parameter_keys": sorted(run.data.params),
        "metric_key_count": len(metric_series),
        "metric_series": metric_series,
        "artifact_count": len(artifact_paths),
        "artifact_bytes_known": sum(
            int(size) for size in artifacts.values() if size is not None
        ),
        "artifact_paths": artifact_paths,
        "model_artifacts": [
            path for path in artifact_paths
            if Path(path).suffix.lower() in {".pt", ".pth", ".onnx"}
        ],
        "provenance_artifacts": [
            path for path in artifact_paths
            if path.startswith("provenance/")
            or "config" in Path(path).name.lower()
            or "history" in Path(path).name.lower()
            or "model_card" in Path(path).name.lower()
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-run-id", required=True)
    parser.add_argument("--env-file", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    load_tracking_environment(args.env_file)
    print(
        json.dumps(
            audit_run(MlflowClient(), args.remote_run_id),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
