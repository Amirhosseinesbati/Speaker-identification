"""Staged Vast.ai runner for the six controlled CAM++ OOF experiments.

Every child process is launched through ``uv run --no-sync``.  The default
phase intentionally runs only the no-proto fold-0 reproducibility gate; costly
follow-up phases must be requested explicitly after inspecting its artifacts.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent

PHASES = {
    "no-proto-f0": ["p0-campp-no-proto-repro-oof-f0"],
    "no-proto-rest": [
        "p0-campp-no-proto-repro-oof-f1",
        "p0-campp-no-proto-repro-oof-f2",
    ],
    "metric-f0": ["p0-campp-metric-only-repro-oof-f0"],
    "metric-rest": [
        "p0-campp-metric-only-repro-oof-f1",
        "p0-campp-metric-only-repro-oof-f2",
    ],
}


def _run(command: list[str], dry_run: bool) -> None:
    print("\n$ " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def _oof_path(profile: str) -> Path:
    return (ROOT / "checkpoints" / profile / "campp_best_bundle" /
            "oof_predictions.npz")


def _aggregate_if_ready(family: str, profiles: list[str], dry_run: bool) -> None:
    paths = [_oof_path(profile) for profile in profiles]
    if not dry_run and not all(path.is_file() for path in paths):
        return
    output = ROOT / "reports" / "generated" / f"{family}_controlled_oof.json"
    _run([
        "uv", "run", "--no-sync", "python", "-X", "utf8",
        "scripts/aggregate_oof_results.py",
        *[str(path.relative_to(ROOT)) for path in paths],
        "--out", str(output.relative_to(ROOT)),
    ], dry_run)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=[*PHASES, "all"], default="no-proto-f0",
        help="Staged block to run; default is the single fold-0 safety gate",
    )
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Skip a profile only when its final OOF bundle exists")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selected_phases = list(PHASES) if args.phase == "all" else [args.phase]
    selected_profiles = [p for phase in selected_phases for p in PHASES[phase]]
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": args.phase,
        "profiles": selected_profiles,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "status": "started",
    }
    print(json.dumps(manifest, indent=2), flush=True)

    _run([
        "uv", "run", "--no-sync", "python", "-X", "utf8",
        "scripts/verify_oof_experiments.py",
    ], args.dry_run)

    for profile in selected_profiles:
        final_oof = _oof_path(profile)
        checkpoint_dir = ROOT / "checkpoints" / profile
        if args.resume and final_oof.is_file():
            print(f"\nSKIP complete profile: {profile}")
            continue
        if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
            raise RuntimeError(
                f"Refusing to overwrite non-empty {checkpoint_dir}. "
                "Use --resume only for already-complete runs, or archive the "
                "directory before launching a fresh run."
            )
        command = [
            "uv", "run", "--no-sync", "python", "-X", "utf8", "-m",
            "src.pipelines.run_pipeline", "--experiment", profile,
            "--run", "eval",
        ]
        if args.no_mlflow:
            command.append("--no-mlflow")
        _run(command, args.dry_run)

    no_proto = [f"p0-campp-no-proto-repro-oof-f{i}" for i in range(3)]
    metric = [f"p0-campp-metric-only-repro-oof-f{i}" for i in range(3)]
    _aggregate_if_ready("campp_no_proto", no_proto, args.dry_run)
    _aggregate_if_ready("campp_metric_only", metric, args.dry_run)

    manifest["status"] = "dry_run" if args.dry_run else "complete"
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    output = ROOT / "reports" / "generated" / (
        f"controlled_oof_runner_{args.phase}_" +
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".json"
    )
    if not args.dry_run:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"\nRunner manifest: {output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
