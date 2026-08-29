"""Replay a packaged submission on a complete audio directory.

The rehearsal deliberately extracts the ZIP to a temporary directory and
runs ``submission.py`` from a different working directory, matching the
leaderboard execution contract.  It records runtime/GPU telemetry and
requires exact, one-to-one filename coverage in the output CSV.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_predictions(csv_path: Path, expected_names: list[str]) -> dict:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)

    names = [row.get("audio_file", "") for row in rows]
    labels = [row.get("speaker_id", "") for row in rows]
    expected = set(expected_names)
    actual = set(names)
    invalid_labels = sorted(
        {label for label in labels if label != "unknown" and not UUID_RE.fullmatch(label)}
    )
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    return {
        "valid": (
            fieldnames == ["audio_file", "speaker_id"]
            and len(rows) == len(expected_names)
            and not duplicates
            and actual == expected
            and not invalid_labels
        ),
        "fieldnames": fieldnames,
        "rows": len(rows),
        "expected_rows": len(expected_names),
        "missing": sorted(expected - actual),
        "unexpected": sorted(actual - expected),
        "duplicates": duplicates,
        "invalid_labels": invalid_labels,
        "unknown_predictions": sum(label == "unknown" for label in labels),
        "known_predictions": sum(label != "unknown" for label in labels),
    }


def poll_gpu(stop: threading.Event, samples: list[dict]) -> None:
    command = [
        "nvidia-smi",
        "--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    while not stop.wait(0.5):
        try:
            line = subprocess.check_output(
                command, text=True, stderr=subprocess.DEVNULL, timeout=5
            ).splitlines()[0]
            memory, util, temperature, power = (item.strip() for item in line.split(","))
            samples.append(
                {
                    "memory_mib": float(memory),
                    "utilization_percent": float(util),
                    "temperature_c": float(temperature),
                    "power_w": float(power),
                }
            )
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            return


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    args = parser.parse_args()

    zip_path = args.zip.absolute()
    data_dir = args.data_dir.absolute()
    python_path = args.python.absolute()  # preserve a virtualenv symlink
    expected_names = sorted(
        path.name for path in data_dir.iterdir() if path.suffix.lower() in AUDIO_SUFFIXES
    )
    if not expected_names:
        raise SystemExit(f"no supported audio files found in {data_dir}")

    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    telemetry: list[dict] = []
    timed_out = False

    with tempfile.TemporaryDirectory(prefix="full_submission_rehearsal_") as tmp:
        tmp_path = Path(tmp)
        extract_dir = tmp_path / "extract"
        cwd = tmp_path / "cwd"
        extract_dir.mkdir()
        cwd.mkdir()
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(extract_dir)
            package_entries = len(archive.infolist())
        entry = extract_dir / "submission.py"
        if not entry.is_file():
            raise SystemExit("submission.py missing at ZIP root")

        process = subprocess.Popen(
            [
                str(python_path),
                str(entry),
                "--data-dir",
                str(data_dir),
                "--predictions-file-path",
                str(args.predictions.absolute()),
            ],
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stop = threading.Event()
        monitor = threading.Thread(target=poll_gpu, args=(stop, telemetry), daemon=True)
        monitor.start()
        monotonic_start = time.perf_counter()
        try:
            stdout, stderr = process.communicate(timeout=args.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            stdout, stderr = process.communicate()
        elapsed = time.perf_counter() - monotonic_start
        stop.set()
        monitor.join(timeout=5)

    validation = (
        validate_predictions(args.predictions, expected_names)
        if args.predictions.is_file()
        else {"valid": False, "error": "predictions file missing"}
    )
    diag_lines = [line for line in stdout.splitlines() if line.startswith("[diag] ")]
    unexpected_stdout = [
        line for line in stdout.splitlines() if line.strip() and not line.startswith("[diag] ")
    ]
    report = {
        "status": "passed"
        if process.returncode == 0
        and not timed_out
        and not stderr
        and not unexpected_stdout
        and validation["valid"]
        else "failed",
        "started_unix": started,
        "elapsed_seconds": elapsed,
        "throughput_files_per_second": len(expected_names) / elapsed,
        "timeout_seconds": args.timeout_seconds,
        "timed_out": timed_out,
        "exit_code": process.returncode,
        "zip_path": str(zip_path),
        "zip_sha256": sha256_file(zip_path),
        "zip_bytes": zip_path.stat().st_size,
        "package_entries": package_entries,
        "data_dir": str(data_dir),
        "audio_files": len(expected_names),
        "predictions_path": str(args.predictions.absolute()),
        "predictions_sha256": sha256_file(args.predictions)
        if args.predictions.is_file()
        else None,
        "stdout_diag_lines": diag_lines,
        "unexpected_stdout": unexpected_stdout,
        "stderr": stderr,
        "gpu_samples": len(telemetry),
        "peak_gpu_memory_mib": max((x["memory_mib"] for x in telemetry), default=None),
        "peak_gpu_utilization_percent": max(
            (x["utilization_percent"] for x in telemetry), default=None
        ),
        "peak_gpu_temperature_c": max(
            (x["temperature_c"] for x in telemetry), default=None
        ),
        "peak_gpu_power_w": max((x["power_w"] for x in telemetry), default=None),
        "validation": validation,
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
