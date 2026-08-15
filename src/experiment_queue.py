"""
Sequential experiment queue runner (Audit §17.2).

Turns a list of named experiment profiles into an ordered queue and runs them
one at a time via ``src/pipelines/run_pipeline.py --experiment <name>``. State
is persisted to ``data/experiments/queue_state.json`` so an interrupted campaign
can be resumed (completed jobs are skipped unless ``--fresh``).

Also importable as a library so the Streamlit "Experiment Matrix" tab can run
the queue through its local-runner (live logs) instead of a raw CLI.

Usage:
    uv run --no-sync python -m src.experiment_queue --profiles ecapa-full-s42 campp-full-s42
    uv run --no-sync python -m src.experiment_queue --profiles ecapa-full-s42 --run train --fresh
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_SCRIPT = PROJECT_ROOT / "src" / "pipelines" / "run_pipeline.py"
STATE_DIR = PROJECT_ROOT / "data" / "experiments"
STATE_PATH = STATE_DIR / "queue_state.json"
LOG_DIR = STATE_DIR / "logs"

VALID_STAGES = ["all", "data", "train", "eval"]


@dataclass
class Job:
    name: str
    stage: str = "all"
    status: str = "pending"          # pending | running | done | failed
    exit_code: Optional[int] = None
    log_file: str = ""
    started_at: str = ""
    finished_at: str = ""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"jobs": []}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8",
    )


def build_command(name: str, stage: str = "all", no_mlflow: bool = False) -> List[str]:
    cmd = [sys.executable, str(PIPELINE_SCRIPT), "--experiment", name, "--run", stage]
    if no_mlflow:
        cmd.append("--no-mlflow")
    return cmd


def run_queue(
    profiles: Sequence[str],
    stage: str = "all",
    no_mlflow: bool = False,
    fresh: bool = False,
    stop_on_error: bool = False,
) -> dict:
    """Run the given profiles in order, updating the persisted queue state."""
    if stage not in VALID_STAGES:
        raise ValueError(f"Unknown stage {stage!r}. Valid: {VALID_STAGES}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    state = load_state()
    existing = {j["name"]: j for j in state.get("jobs", [])}

    jobs: List[dict] = []
    for name in profiles:
        job = existing.get(name, asdict(Job(name=name)))
        job["stage"] = stage
        jobs.append(job)
    state["jobs"] = jobs
    save_state(state)

    print("=" * 60)
    print(f"  Experiment Queue — {len(jobs)} job(s), stage={stage}")
    print("=" * 60)

    for job in jobs:
        if not fresh and job.get("status") == "done":
            print(f"  ⏭  {job['name']} — already done, skipping (--fresh to re-run)")
            continue

        job["status"] = "running"
        job["started_at"] = _now()
        save_state(state)

        log_file = LOG_DIR / f"{job['name']}.log"
        job["log_file"] = str(log_file)
        cmd = build_command(job["name"], job["stage"], no_mlflow)
        print(f"\n  ▶ {job['name']}  →  {' '.join(cmd)}")

        # Stream the child's output line-by-line to BOTH the log file and our
        # own stdout, so a remote/long run is visible live (the old code wrote
        # only to the file, which made the console look stuck).
        # Forward the job name so the run's MLflow deployment envelope records
        # which queue job it belongs to (visible on DagsHub).
        env = {**os.environ, "PYTHONUNBUFFERED": "1",
               "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1",
               "QUEUE_JOB": job["name"]}
        proc = subprocess.Popen(
            cmd, cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1, env=env,
        )
        with open(log_file, "w", encoding="utf-8") as lf:
            for line in iter(proc.stdout.readline, ""):
                lf.write(line)
                lf.flush()
                print(line, end="", flush=True)
        proc.stdout.close()
        proc.wait()

        job["exit_code"] = int(proc.returncode)
        job["status"] = "done" if proc.returncode == 0 else "failed"
        job["finished_at"] = _now()
        save_state(state)

        marker = "✅" if job["status"] == "done" else "❌"
        print(f"  {marker} {job['name']} (exit {proc.returncode}) → {log_file}")
        if job["status"] == "failed" and stop_on_error:
            print("\n  🛑 Stopping queue on first failure (--stop-on-error).")
            break

    done = sum(1 for j in jobs if j["status"] == "done")
    failed = sum(1 for j in jobs if j["status"] == "failed")
    print(f"\n  Queue finished — {done} done, {failed} failed, {len(jobs)} total.")
    return state


def clear_state() -> None:
    STATE_PATH.unlink(missing_ok=True)
    print(f"  🧹 Queue state cleared ({STATE_PATH}).")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a queue of experiment profiles")
    parser.add_argument("--profiles", nargs="*", default=None,
                        help="Named experiment profiles to run (in order).")
    parser.add_argument("--run", default="all", choices=VALID_STAGES,
                        help="Pipeline stage per job (default: all).")
    parser.add_argument("--no-mlflow", action="store_true",
                        help="Disable MLflow tracking for each job.")
    parser.add_argument("--fresh", action="store_true",
                        help="Re-run jobs already marked done.")
    parser.add_argument("--stop-on-error", action="store_true",
                        help="Abort the queue at the first failed job.")
    parser.add_argument("--clear", action="store_true",
                        help="Clear the persisted queue state and exit.")
    args = parser.parse_args()

    if args.clear:
        clear_state()
        return 0

    if not args.profiles:
        parser.error("--profiles is required unless --clear is given.")

    run_queue(
        args.profiles, stage=args.run, no_mlflow=args.no_mlflow,
        fresh=args.fresh, stop_on_error=args.stop_on_error,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
