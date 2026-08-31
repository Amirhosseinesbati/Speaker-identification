"""Inspect a remote MLflow run without exposing credentials or mutating it."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

from mlflow.tracking import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mlflow_backfill_run import _remote_artifacts, load_tracking_environment


def resolve_run_id(
    client: MlflowClient,
    *,
    run_id: str | None,
    run_name: str | None,
    experiment_name: str,
    started_after_ms: int | None,
) -> str:
    """Resolve one exact run without guessing when names are ambiguous."""

    if run_id:
        return run_id
    if not run_name:
        raise ValueError("either run_id or run_name is required")

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"MLflow experiment not found: {experiment_name!r}")

    candidates = []
    for run in client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="",
        max_results=10_000,
        order_by=["attributes.start_time DESC"],
    ):
        if run.data.tags.get("mlflow.runName") != run_name:
            continue
        if started_after_ms is not None and int(run.info.start_time) < started_after_ms:
            continue
        candidates.append(run)

    if len(candidates) != 1:
        safe_candidates = [
            {
                "run_id": run.info.run_id,
                "start_time": int(run.info.start_time),
                "status": run.info.status,
            }
            for run in candidates
        ]
        raise ValueError(
            "expected exactly one MLflow run for "
            f"name={run_name!r}, experiment={experiment_name!r}, "
            f"started_after_ms={started_after_ms!r}; found "
            f"{len(candidates)}: {safe_candidates}"
        )
    return str(candidates[0].info.run_id)


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
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--remote-run-id")
    selector.add_argument("--run-name")
    parser.add_argument("--experiment-name", default="speaker-identification")
    parser.add_argument("--started-after-ms", type=int)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    load_tracking_environment(args.env_file)
    client = MlflowClient()
    run_id = resolve_run_id(
        client,
        run_id=args.remote_run_id,
        run_name=args.run_name,
        experiment_name=args.experiment_name,
        started_after_ms=args.started_after_ms,
    )
    encoded = json.dumps(
        audit_run(client, run_id),
        ensure_ascii=False,
        indent=2,
    ) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, args.output)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
