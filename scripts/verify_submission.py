"""One-shot verification of submission_leaderboard.zip.

Replays the exact flow the leaderboard runs:
    1. extract the zip into a fresh dir
    2. copy a few real audio samples into a test data dir
    3. run `submission.py` from a DIFFERENT cwd (NOT the package dir)
    4. assert the run is silent (0 bytes stdout/stderr) so a real server
       error would be the first thing shown
    5. validate the CSV (audio_file,speaker_id with valid UUIDs / unknown)

Usage (cmd.exe):
    D:\\Projects\\My projects\\IAAA_Compet\\leaderbordvenv\\.venv\\Scripts\\python.exe
        D:\\Projects\\My projects\\IAAA_Compet\\Speaker-identification\\scripts\\verify_submission.py
"""

import csv
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ZIP = ROOT / "submission_leaderboard.zip"
RAW = ROOT / "data" / "raw"
LEGACY_LEADERBOARD_PYTHON = Path(
    r"D:\Projects\My projects\IAAA_Compet\leaderbordvenv\.venv\Scripts\python.exe"
)
N_SAMPLES = 8
UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
SUFFIXES = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify one leaderboard submission ZIP")
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument(
        "--python", type=Path,
        default=Path(os.environ.get("SUBMISSION_VERIFY_PYTHON", ""))
        if os.environ.get("SUBMISSION_VERIFY_PYTHON")
        else LEGACY_LEADERBOARD_PYTHON
        if LEGACY_LEADERBOARD_PYTHON.exists()
        else Path(sys.executable),
        help="Python interpreter used to replay the leaderboard entry point",
    )
    parser.add_argument("--samples", type=int, default=N_SAMPLES)
    args = parser.parse_args()
    zip_path = args.zip.resolve()
    # Preserve virtualenv symlinks on Linux. Path.resolve() dereferences
    # ``.venv/bin/python`` to the system interpreter and silently loses that
    # environment's site-packages when the replay runs from a different cwd.
    python_path = Path(os.path.abspath(args.python))
    print("=" * 70)
    print(f"  Verification of {zip_path.name}")
    print("=" * 70)

    if not zip_path.exists():
        print(f"  ZIP missing: {zip_path}")
        return 1
    if not python_path.exists():
        print(f"  python missing: {python_path}")
        return 1

    work = Path(tempfile.mkdtemp(prefix="verify_sub_"))
    extract, cwd, data = work / "extract", work / "cwd", work / "data"
    extract.mkdir()
    cwd.mkdir()
    data.mkdir()

    # ── 1. extract ──
    print("\n[1/5] Extracting zip ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(extract)
    entry = extract / "submission.py"
    if not entry.exists():
        print(f"  FAIL: submission.py missing after extract -> {extract}")
        return 1
    n_files = sum(1 for f in extract.rglob("*") if f.is_file())
    print(f"      OK ({n_files} files extracted)")

    # ── 2. sample audio ──
    print("[2/5] Copying sample audio ...")
    mp3s = sorted(RAW.glob("*.mp3"))[:args.samples]
    if not mp3s:
        print(f"  FAIL: no audio files in {RAW}")
        return 1
    for m in mp3s:
        shutil.copy2(m, data / m.name)
    print(f"      OK ({len(mp3s)} files -> {data})")

    # ── 3. run from a different cwd (like the leaderboard) ──
    print(f"[3/5] Running submission.py from cwd = {cwd}")
    print("      (deliberately NOT the package dir)")
    out_csv = cwd / "out.csv"
    try:
        proc = subprocess.run(
            [str(python_path), str(entry), "--data-dir", str(data),
             "--predictions-file-path", str(out_csv)],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=3600,
        )
    except subprocess.TimeoutExpired:
        print("  FAIL: run exceeded 1h timeout")
        return 1

    print(f"      exit code: {proc.returncode}")
    print(f"      stdout: {len(proc.stdout)} bytes | stderr: {len(proc.stderr)} bytes")
    if proc.returncode != 0:
        print("      --- stderr (first 40 lines) ---")
        print("\n".join(proc.stderr.splitlines()[:40]))
        print("  FAIL: submission crashed")
        return 1

    # ── 4. silence check ──
    # The package intentionally prints one-line "[diag] " diagnostics (the GPU
    # probe in submission.py + the centroid-merge note in inference.py) — the
    # first thing the leaderboard surfaces. EVERYTHING else must be silent so
    # a real server error would be the first unexpected line.
    print("[4/5] Silence check (only [diag] lines allowed on stdout) ...")
    out_lines = proc.stdout.splitlines()
    diag_lines = [ln for ln in out_lines if ln.startswith("[diag] ")]
    unexpected = [ln for ln in out_lines if ln.strip() and not ln.startswith("[diag] ")]
    if unexpected or proc.stderr:
        print("  FAIL: unexpected output on stdout/stderr")
        for ln in unexpected[:20]:
            print("  stdout:", ln)
        print("  stderr:", proc.stderr[:500])
        return 1
    print(f"      OK ({len(diag_lines)} [diag] line(s), no unexpected output)")

    # ── 5. CSV validation ──
    print("[5/5] Validating CSV ...")
    with open(out_csv, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if rows[0] != ["audio_file", "speaker_id"]:
        print(f"  FAIL: bad header {rows[0]}")
        return 1
    bad = [
        r for r in rows[1:]
        if len(r) != 2
        or not r[0].lower().endswith(tuple(SUFFIXES))
        or (r[1] != "unknown" and not UUID_RE.match(r[1]))
    ]
    ok = len(rows) == args.samples + 1 and not bad
    print(f"      rows={len(rows)-1} bad={len(bad)} header={rows[0]}")
    for r in rows[1:min(4, len(rows))]:
        print("        ", r)
    if not ok:
        print("  FAIL: CSV invalid")
        return 1
    print("      OK")

    print("\n" + "=" * 70)
    print("  ALL CHECKS PASSED ✅  zip is ready to submit.")
    print("=" * 70)

    shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
