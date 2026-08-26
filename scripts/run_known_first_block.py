"""Gated Vast.ai runner for the known-first CAM++ architecture campaign.

The default ``gate`` phase runs exactly two fold-0 experiments.  Inspect their
OOF artifacts before choosing *one* confirmation phase; the runner deliberately
has no ``all`` option to prevent spending GPU time on both losing follow-ups.

Examples:
    uv run --no-sync python scripts/run_known_first_block.py --phase gate
    uv run --no-sync python scripts/run_known_first_block.py --phase auxmetric-rest
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent

PHASES = {
    "gate": [
        "p0-campp-known446-ood-control-oof-f0",
        "p0-campp-known446-ood-auxmetric-oof-f0",
    ],
    "control-rest": [
        "p0-campp-known446-ood-control-oof-f1",
        "p0-campp-known446-ood-control-oof-f2",
    ],
    "auxmetric-rest": [
        "p0-campp-known446-ood-auxmetric-oof-f1",
        "p0-campp-known446-ood-auxmetric-oof-f2",
    ],
}


def _run(command: list[str], dry_run: bool) -> None:
    print("\n$ " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def _oof_path(profile: str) -> Path:
    return ROOT / "checkpoints" / profile / "campp_best_bundle" / "oof_predictions.npz"


def _aggregate_if_complete(family: str, dry_run: bool) -> None:
    profiles = [f"p0-campp-known446-ood-{family}-oof-f{i}" for i in range(3)]
    paths = [_oof_path(profile) for profile in profiles]
    if not dry_run and not all(path.is_file() for path in paths):
        return
    output = ROOT / "reports" / "generated" / f"campp_known446_ood_{family}_oof.json"
    _run([
        "uv", "run", "--no-sync", "python", "-X", "utf8",
        "scripts/aggregate_oof_results.py",
        *[str(path.relative_to(ROOT)) for path in paths],
        "--out", str(output.relative_to(ROOT)),
    ], dry_run)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=list(PHASES), default="gate")
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    profiles = PHASES[args.phase]
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": args.phase,
        "profiles": profiles,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "status": "started",
    }
    print(json.dumps(manifest, indent=2), flush=True)

    _run([
        "uv", "run", "--no-sync", "python", "-X", "utf8",
        "scripts/verify_known_first_experiments.py",
    ], args.dry_run)

    for profile in profiles:
        final_oof = _oof_path(profile)
        checkpoint_dir = ROOT / "checkpoints" / profile
        if args.resume and final_oof.is_file():
            print(f"\nSKIP complete profile: {profile}")
            continue
        if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
            raise RuntimeError(
                f"Refusing to overwrite non-empty {checkpoint_dir}. "
                "Archive it first, or use --resume only for complete runs."
            )
        command = [
            "uv", "run", "--no-sync", "python", "-X", "utf8", "-m",
            "src.pipelines.run_pipeline", "--experiment", profile, "--run", "eval",
        ]
        if args.no_mlflow:
            command.append("--no-mlflow")
        _run(command, args.dry_run)

    if args.phase == "control-rest":
        _aggregate_if_complete("control", args.dry_run)
    elif args.phase == "auxmetric-rest":
        _aggregate_if_complete("auxmetric", args.dry_run)

    manifest["status"] = "dry_run" if args.dry_run else "complete"
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    if not args.dry_run:
        output = ROOT / "reports" / "generated" / (
            "known_first_runner_" + args.phase + "_"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".json"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"\nRunner manifest: {output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
