"""Evaluate raw-PCM recovery on files excluded from the clean OOF split.

The audited corpus contains non-empty stereo PCM16 payloads without a RIFF
header but with an ``.mp3`` suffix.  This script wraps only structurally
plausible payloads in a temporary WAV container, runs an already-built
submission ZIP, and compares the predictions with the locked decode-failure
policy (always ``unknown``).  No recovered audio is added to training or to
the prototype artifact.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
import wave
import zipfile
from pathlib import Path

import numpy as np


UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)


def inspect_pcm16_stereo(path: Path, sample_rate: int = 16_000) -> dict:
    data = path.read_bytes()
    result = {
        "bytes": len(data),
        "eligible": False,
        "reason": "",
        "duration_seconds": 0.0,
        "channel_equal_fraction": 0.0,
        "rms": 0.0,
        "peak": 0.0,
    }
    if not data:
        result["reason"] = "empty"
        return result
    if data[:4] == b"RIFF" or data[:3] == b"ID3":
        result["reason"] = "containerized"
        return result
    if len(data) % 4:
        result["reason"] = "not_stereo_pcm16_aligned"
        return result
    samples = np.frombuffer(data, dtype="<i2").reshape(-1, 2)
    mono = samples.astype(np.float32).mean(axis=1) / 32768.0
    result.update(
        duration_seconds=len(samples) / sample_rate,
        channel_equal_fraction=float(np.mean(samples[:, 0] == samples[:, 1])),
        rms=float(np.sqrt(np.mean(np.square(mono)))) if len(mono) else 0.0,
        peak=float(np.max(np.abs(mono))) if len(mono) else 0.0,
    )
    result["eligible"] = (
        0.25 <= result["duration_seconds"] <= 300.0
        and result["channel_equal_fraction"] >= 0.99
        and result["peak"] > 0.0
    )
    result["reason"] = "plausible_pcm16_stereo" if result["eligible"] else "quality_gate"
    return result


def macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    scores = []
    for label in sorted(set(y_true) | set(y_pred)):
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        denominator = 2 * tp + fp + fn
        scores.append(2 * tp / denominator if denominator else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def evaluate(y_true: list[str], y_pred: list[str]) -> dict:
    known_mask = [label != "unknown" for label in y_true]
    unknown_mask = [not value for value in known_mask]
    return {
        "files": len(y_true),
        "accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
        "macro_f1_subset": macro_f1(y_true, y_pred),
        "known_files": sum(known_mask),
        "known_accuracy": float(
            np.mean([t == p for t, p, keep in zip(y_true, y_pred, known_mask) if keep])
        )
        if any(known_mask)
        else None,
        "unknown_files": sum(unknown_mask),
        "unknown_accuracy": float(
            np.mean([p == "unknown" for p, keep in zip(y_pred, unknown_mask) if keep])
        )
        if any(unknown_mask)
        else None,
        "predicted_unknown": sum(label == "unknown" for label in y_pred),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--class-map", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args()

    class_map = json.loads(args.class_map.read_text(encoding="utf-8"))
    known_speakers = {key for key in class_map if UUID_RE.fullmatch(key)}
    with args.labels.open(newline="", encoding="utf-8") as handle:
        source_labels = {row["audio_file"]: row["speaker_id"] for row in csv.DictReader(handle)}

    inspected = {}
    candidates = []
    for path in sorted(args.raw_dir.iterdir()):
        if path.suffix.lower() != ".mp3":
            continue
        info = inspect_pcm16_stereo(path)
        if info["reason"] not in {"containerized", "empty"}:
            inspected[path.name] = info
        if info["eligible"]:
            candidates.append(path)
    if not candidates:
        raise SystemExit("no plausible headerless PCM candidates")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="headerless_pcm_recovery_") as tmp:
        tmp_path = Path(tmp)
        audio_dir = tmp_path / "audio"
        extract_dir = tmp_path / "extract"
        cwd = tmp_path / "cwd"
        audio_dir.mkdir()
        extract_dir.mkdir()
        cwd.mkdir()
        for source in candidates:
            target = audio_dir / f"{source.stem}.wav"
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(2)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(source.read_bytes())
        with zipfile.ZipFile(args.zip) as archive:
            archive.extractall(extract_dir)
        output = tmp_path / "predictions.csv"
        started = __import__("time").perf_counter()
        process = subprocess.run(
            [
                str(args.python.absolute()),
                str(extract_dir / "submission.py"),
                "--data-dir",
                str(audio_dir),
                "--predictions-file-path",
                str(output),
            ],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=args.timeout_seconds,
        )
        elapsed = __import__("time").perf_counter() - started
        with output.open(newline="", encoding="utf-8") as handle:
            predictions = {
                f"{Path(row['audio_file']).stem}.mp3": row["speaker_id"]
                for row in csv.DictReader(handle)
            }

    names = [path.name for path in candidates]
    missing_labels = sorted(name for name in names if name not in source_labels)
    missing_predictions = sorted(name for name in names if name not in predictions)
    if missing_labels or missing_predictions or process.returncode:
        raise SystemExit(
            f"recovery replay incomplete: labels={missing_labels} "
            f"predictions={missing_predictions} exit={process.returncode}"
        )
    y_true = [source_labels[name] if source_labels[name] in known_speakers else "unknown" for name in names]
    recovered = [predictions[name] for name in names]
    baseline = ["unknown"] * len(names)
    rescued = [name for name, t, b, r in zip(names, y_true, baseline, recovered) if b != t and r == t]
    introduced = [name for name, t, b, r in zip(names, y_true, baseline, recovered) if b == t and r != t]
    report = {
        "status": "evaluated",
        "selection_rule": {
            "format": "little-endian PCM16 stereo at 16 kHz",
            "byte_alignment": 4,
            "duration_seconds": [0.25, 300.0],
            "minimum_channel_equal_fraction": 0.99,
            "positive_peak_required": True,
        },
        "inspected_non_containerized": len(inspected),
        "eligible_files": len(names),
        "elapsed_seconds": elapsed,
        "exit_code": process.returncode,
        "stderr": process.stderr,
        "unexpected_stdout": [
            line
            for line in process.stdout.splitlines()
            if line.strip() and not line.startswith("[diag] ")
        ],
        "baseline_decode_failure": evaluate(y_true, baseline),
        "recovered_pcm": evaluate(y_true, recovered),
        "rescued": len(rescued),
        "introduced": len(introduced),
        "rescued_files": rescued,
        "introduced_files": introduced,
        "per_file": [
            {
                "audio_file": name,
                "truth": truth,
                "baseline": base,
                "recovered": prediction,
                **inspected[name],
            }
            for name, truth, base, prediction in zip(names, y_true, baseline, recovered)
        ],
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {key: report[key] for key in ["status", "eligible_files", "rescued", "introduced"]}
    summary["baseline"] = report["baseline_decode_failure"]
    summary["recovered"] = report["recovered_pcm"]
    summary["stderr_bytes"] = len(process.stderr)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
