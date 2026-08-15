"""
Promote the current artifacts into a leaderboard submission (Audit §17.5).

Closes the "experiment → decision → submission" loop:
    1. Rebuild the submission package via scripts/build_submission.py.
    2. Append a one-line record to reports/lb_log.md (config/label, package
       contents, note) so every submission is traceable.
    3. Optionally run scripts/verify_submission.py (--verify) if the zip exists.

Usage:
    uv run --no-sync python scripts/promote_run.py --label "campp-full-ft-s42" --note "first OOF promote"
    uv run --no-sync python scripts/promote_run.py --label "hp0-best" --verify
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "scripts" / "build_submission.py"
VERIFY = ROOT / "scripts" / "verify_submission.py"
LB_LOG = ROOT / "reports" / "lb_log.md"
ZIP = ROOT / "submission_leaderboard.zip"
PROCESSED = ROOT / "data" / "processed"


def _run(cmd: list) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(ROOT))


def package_contents_summary() -> str:
    """One-line summary of what the current submission package will ship."""
    ckpts = sorted(p.name for p in (ROOT / "checkpoints").glob("*_best.pt"))
    centroids = sorted(p.name for p in PROCESSED.glob("centroids_*.npz"))
    has_decision = (PROCESSED / "decision_config.json").exists()
    ckpt_names = ",".join(c.replace("_best.pt", "") for c in ckpts) or "none"
    cent_names = ",".join(c.replace("centroids_", "").replace(".npz", "")
                          for c in centroids) or "none"
    return (f"ckpt:{ckpt_names} | centroids:{cent_names} | "
            f"decision:{'yes' if has_decision else 'no'}")


def append_lb_log(label: str, note: str) -> None:
    """Append (or insert before the ## TODO section) one markdown table row."""
    lines = LB_LOG.read_text(encoding="utf-8").splitlines(keepends=True)
    row = (
        f"| {datetime.now().strftime('%Y-%m-%d')} | {label} | "
        f"{package_contents_summary()} | — | {note or '—'} | ? |\n"
    )
    idx = next((i for i, l in enumerate(lines)
                if l.lstrip().startswith("## TODO")), None)
    if idx is not None:
        lines.insert(idx, row)
    else:
        lines.append(row)
    LB_LOG.write_text("".join(lines), encoding="utf-8")


def promote(label: str = "manual", note: str = "", verify: bool = False) -> int:
    print("=" * 60)
    print("  Promote → Submission")
    print("=" * 60)
    print(f"  Label: {label} | Note: {note or '—'}")

    print("\n[1/3] Building submission package (scripts/build_submission.py)...")
    r = _run([sys.executable, str(BUILD)])
    if r.returncode != 0:
        print("  ❌ Build failed — nothing recorded.")
        return r.returncode

    print("\n[2/3] Recording reports/lb_log.md row...")
    append_lb_log(label, note)
    print(f"  ✓ lb_log.md updated: {package_contents_summary()}")

    if verify:
        print("\n[3/3] Verifying submission_leaderboard.zip...")
        if not ZIP.exists():
            print(f"  ⚠ {ZIP} missing — build the zip first, skipping verify.")
        else:
            r = _run([sys.executable, str(VERIFY)])
            if r.returncode != 0:
                print("  ⚠ Verify failed (see output above).")
                return r.returncode
    else:
        print("\n[3/3] Verify skipped — run scripts/verify_submission.py before upload.")

    print("\n✅ Promote complete. Zip `submission/` → `submission_leaderboard.zip`, "
          "run verify, then upload.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Promote artifacts to a submission")
    parser.add_argument("--label", default="manual",
                        help="Config / run label recorded in lb_log.md.")
    parser.add_argument("--note", default="",
                        help="Free-text note for the lb_log.md row.")
    parser.add_argument("--verify", action="store_true",
                        help="Also run scripts/verify_submission.py.")
    args = parser.parse_args()
    return promote(label=args.label, note=args.note, verify=args.verify)


if __name__ == "__main__":
    sys.exit(main())
