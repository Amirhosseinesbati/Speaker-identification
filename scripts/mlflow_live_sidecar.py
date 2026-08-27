"""Recover live DagsHub/MLflow charts for an already-running campaign job.

The training pipeline writes a self-describing ``latest_model.pt`` after every
epoch.  This sidecar mirrors its scalar history into a dedicated MLflow run
without touching the training process or its checkout.  It is intentionally a
recovery tool: normal future runs should log directly from the pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import mlflow
import torch


def _clean_env(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _checkpoint_history(path: Path) -> list[dict[str, Any]]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, EOFError, pickle.UnpicklingError):
        return []
    history = checkpoint.get("training_history", checkpoint.get("history", []))
    return history if isinstance(history, list) else []


def _scalar_metrics(row: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in row.items():
        if key == "epoch" or isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            metrics[key] = float(value)
    if "lr" in metrics:
        metrics.setdefault("learning_rate", metrics["lr"])
        metrics.setdefault("head_lr", metrics["lr"])
    return metrics


def _matching_completed_run(state: dict[str, Any], profile: str) -> dict[str, Any] | None:
    for row in reversed(state.get("completed_runs", [])):
        if row.get("profile") == profile:
            return row
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-log", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--max-hours", type=float, default=12.0)
    parser.add_argument("--backfill-local-run-dir", type=Path)
    parser.add_argument("--backfill-local-model-dir", type=Path)
    parser.add_argument("--backfill-env-file", type=Path)
    parser.add_argument(
        "--backfill-extra-artifact",
        action="append",
        default=[],
        metavar="LOCAL::REMOTE",
    )
    args = parser.parse_args()

    tracking_uri = _clean_env(
        os.environ.get("MLFLOW_TRACKING_URI")
        or os.environ.get("DAGSHUB_TRACKING_URI", "")
    )
    owner = _clean_env(os.environ.get("DAGSHUB_REPO_OWNER", ""))
    token = _clean_env(os.environ.get("DAGSHUB_USER_TOKEN", ""))
    if not tracking_uri or not owner or not token:
        raise RuntimeError("DagsHub tracking environment is incomplete")
    os.environ["MLFLOW_TRACKING_USERNAME"] = owner
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("speaker-identification")

    stop_requested = False

    def _stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    started = time.monotonic()
    final_status = "KILLED"
    run_kwargs = (
        {"run_id": args.run_id}
        if args.run_id
        else {"run_name": f"{args.profile}-live-recovery"}
    )
    with mlflow.start_run(**run_kwargs) as active:
        mlflow.set_tags({
            "campaign.live_tracking": "recovery-sidecar",
            "campaign.profile": args.profile,
            "campaign.git_commit": args.git_commit,
            "campaign.config_sha256": args.config_sha256,
            "mlflow.note.content": (
                "بازیابی زندهٔ نمودارهای اجرای در حال آموزش؛ مدل و داده‌های "
                "علمی بدون تغییر باقی مانده‌اند."
            ),
        })
        mlflow.log_params({
            "profile": args.profile,
            "git_commit": args.git_commit,
            "config_sha256": args.config_sha256,
            "tracking_mode": "live-recovery-sidecar",
        })
        if args.config.is_file():
            mlflow.log_artifact(str(args.config), artifact_path="provenance")
        print(json.dumps({
            "status": "started",
            "run_id": active.info.run_id,
            "run_name": active.info.run_name,
        }), flush=True)

        logged_epoch = 0
        while not stop_requested:
            for row in _checkpoint_history(args.checkpoint):
                epoch = int(row.get("epoch", 0))
                if epoch <= logged_epoch:
                    continue
                metrics = _scalar_metrics(row)
                if metrics:
                    mlflow.log_metrics(metrics, step=epoch)
                logged_epoch = epoch
                print(json.dumps({
                    "status": "synced", "epoch": epoch,
                    "val_macro_f1": metrics.get("val_macro_f1"),
                    "val_ema_macro_f1": metrics.get("val_ema_macro_f1"),
                }), flush=True)

            try:
                state = _read_json(args.state)
            except (OSError, json.JSONDecodeError):
                time.sleep(args.poll_seconds)
                continue
            current = state.get("current_run") or {}
            if not (
                state.get("status") == "RUNNING_EXPERIMENT"
                and current.get("profile") == args.profile
            ):
                completed = _matching_completed_run(state, args.profile)
                final_status = (
                    "FINISHED" if completed and completed.get("exit_code") == 0
                    else "FAILED"
                )
                if completed:
                    mlflow.set_tag("campaign.exit_code", str(completed.get("exit_code")))
                break
            if time.monotonic() - started >= args.max_hours * 3600:
                mlflow.set_tag("campaign.sidecar_timeout", "true")
                break
            time.sleep(args.poll_seconds)

        for row in _checkpoint_history(args.checkpoint):
            epoch = int(row.get("epoch", 0))
            if epoch > logged_epoch:
                mlflow.log_metrics(_scalar_metrics(row), step=epoch)
                logged_epoch = epoch
        mlflow.log_metric("live_epochs_mirrored", float(logged_epoch))
        if args.run_log.is_file():
            mlflow.log_artifact(str(args.run_log), artifact_path="logs")
            mlflow.set_tag("campaign.run_log_sha256", _sha256(args.run_log))
        mlflow.set_tag("campaign.final_status", final_status)
        mlflow.end_run(status=final_status)
        print(json.dumps({
            "status": final_status,
            "epochs_mirrored": logged_epoch,
        }), flush=True)

    if final_status == "FINISHED" and args.backfill_local_run_dir:
        backfill_script = Path(__file__).with_name("mlflow_backfill_run.py")
        command = [
            sys.executable,
            str(backfill_script),
            "--remote-run-id",
            active.info.run_id,
            "--local-run-dir",
            str(args.backfill_local_run_dir),
            "--profile",
            args.profile,
            "--git-commit",
            args.git_commit,
            "--config-sha256",
            args.config_sha256,
        ]
        if args.backfill_local_model_dir:
            command.extend(
                ["--local-model-dir", str(args.backfill_local_model_dir)]
            )
        if args.backfill_env_file:
            command.extend(["--env-file", str(args.backfill_env_file)])
        for specification in args.backfill_extra_artifact:
            command.extend(["--extra-artifact", specification])
        print(
            json.dumps(
                {
                    "status": "artifact_backfill_started",
                    "run_id": active.info.run_id,
                }
            ),
            flush=True,
        )
        subprocess.run(command, check=True)
    return 0 if final_status == "FINISHED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
