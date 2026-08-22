"""Audit: acoustic features for the 891 native-split val files.

Labels each val file clean / low-snr / non-speech using lightweight
energy-based VAD + spectral flatness, then merges with the error report
(legacy vs cluster verdicts) and saves a merged CSV for downstream analysis.

Usage:
    uv run --no-sync python scripts/audit_acoustic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.cli_utils import setup_utf8_stdio  # noqa: E402
setup_utf8_stdio()

AUDIO_DIR = ROOT / "data" / "processed" / "audio_wav"
ERR_CSV = ROOT / "error_analysis_report.csv"
OUT_CSV = ROOT / "data" / "processed" / "val_acoustic_features.csv"

SR = 16000
FRAME_MS = 25
HOP_MS = 10
FRAME = int(SR * FRAME_MS / 1000)
HOP = int(SR * HOP_MS / 1000)


def frame_features(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame RMS (dB) and spectral flatness over a 25ms/10ms grid."""
    n = (len(x) - FRAME) // HOP + 1
    if n < 1:
        return np.array([]), np.array([]), np.array([])
    idx = np.arange(n)[:, None] * HOP + np.arange(FRAME)[None, :]
    frames = x[idx]  # (n, FRAME)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    rms_db = 20 * np.log10(rms + 1e-9)
    # flatness: geometric/arithmetic mean of per-frame magnitude spectrum
    win = np.hanning(FRAME)
    spec = np.abs(np.fft.rfft(frames * win, axis=1)) + 1e-12
    flat = np.exp(np.log(spec).mean(axis=1)) / (spec.mean(axis=1) + 1e-12)
    return rms_db, flat, rms


def classify(rms_db: np.ndarray, flat: np.ndarray, duration: float) -> tuple[str, float]:
    """Simple 3-bucket split: clean_speech / low_snr_speech / non_speech.

    VAD = frames above a noise-floor-relative threshold. If a file is almost
    all silence the noise floor IS the silence, so we anchor the floor with an
    absolute lower bound (-55 dB) and call a frame speech when it is > floor+12dB.
    """
    if rms_db.size == 0:
        return "non_speech", 0.0
    floor_db = max(np.percentile(rms_db, 10), -55.0)
    speech = rms_db > (floor_db + 12.0)
    vad_ratio = float(speech.mean())
    peak_flat = float(np.median(flat[speech]) if speech.any() else np.median(flat))
    # speech-ish: >30% voiced frames; noisy but some speech; else non-speech
    if vad_ratio >= 0.30:
        bucket = "clean_speech" if peak_flat < 0.35 else "low_snr_speech"
    elif vad_ratio >= 0.05:
        bucket = "low_snr_speech"
    else:
        bucket = "non_speech"
    return bucket, vad_ratio


def main() -> int:
    err = pd.read_csv(ERR_CSV, encoding="utf-8-sig")
    rows = []
    for i, row in err.iterrows():
        wav = AUDIO_DIR / row["audio_file"]
        if not wav.exists():
            rows.append({"audio_file": row["audio_file"], "duration_s": row["duration_s"],
                         "load_error": True})
            continue
        x, sr = sf.read(str(wav), dtype="float32", always_2d=True)
        x = x.mean(axis=1).astype(np.float32)
        if sr != SR:
            n = int(len(x) * SR / sr)
            x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
        rms_db, flat, rms = frame_features(x)
        bucket, vad_ratio = classify(rms_db, flat, len(x) / SR)
        rows.append({
            "audio_file": row["audio_file"],
            "duration_s": len(x) / SR,
            "load_error": False,
            "bucket": bucket,
            "vad_ratio": round(vad_ratio, 4),
            "rms_db": round(float(np.median(rms_db)), 2) if rms_db.size else np.nan,
            "rms_db_p10": round(float(np.percentile(rms_db, 10)), 2) if rms_db.size else np.nan,
            "rms_db_p90": round(float(np.percentile(rms_db, 90)), 2) if rms_db.size else np.nan,
            "flatness_median": round(float(np.median(flat)), 4) if flat.size else np.nan,
            "flatness_speech_median": round(float(np.median(flat[rms_db > (np.percentile(rms_db, 10) + 12)])), 4) if rms_db.size and (rms_db > (np.percentile(rms_db, 10) + 12)).any() else np.nan,
        })
        if (i + 1) % 200 == 0:
            print(f"  ...{i + 1}/{len(err)}")
    feat = pd.DataFrame(rows)
    merged = err.merge(feat, on="audio_file", how="left")
    merged.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n  Saved {OUT_CSV}  ({len(merged)} rows)")
    print("\n  Bucket distribution (all val):")
    print(merged["bucket"].value_counts().to_string())
    cl = merged[merged["cluster_verdict"] != "OK"]
    print("\n  Bucket distribution (cluster errors only):")
    print(cl["bucket"].value_counts().to_string())
    print("\n  Cluster errors:")
    print(cl[["audio_file", "duration_s", "bucket", "vad_ratio", "rms_db",
              "flatness_speech_median", "cluster_verdict"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())