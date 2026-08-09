"""
Unified MP3 → WAV preprocessing, shared by local conversion and the server pipeline.

This module is the SINGLE source of truth for audio conversion. Both the local
script (`scripts/convert_mp3_to_wav.py`) and the ZenML server step
(`src/pipelines/steps.py::convert_audio`) call `convert_all` here, so local and
remote runs produce byte-identical WAVs from the same raw files.

Decoder choice: `soundfile` (libsndfile) is used for every format it supports
(MP3, FLAC, OGG, WAV, ...). libsndfile is bundled inside the `soundfile` wheel
and its version is pinned in `uv.lock`, so decoding is deterministic across
platforms (Windows local / Linux Vast.ai). This matters because
`librosa`/`audioread` routes MP3 to `mpg123` on Windows but `ffmpeg` on Linux,
which can yield different samples on each machine. `librosa.load` is kept only
as a fallback for formats libsndfile cannot decode (e.g. M4A/AAC).

Output: mono 16 kHz PCM-16 WAV in `data/processed/audio_wav/` plus a rewritten
labels CSV (`data/processed/audio_wav_labels.csv`) pointing at the .wav names.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

TARGET_SR = 16000
WAV_SUBTYPE = "PCM_16"
DEFAULT_MAX_WORKERS = 8


def convert_one(src_path: Path, dst_path: Path, target_sr: int = TARGET_SR) -> dict:
    """
    Convert a single audio file to mono `target_sr` PCM-16 WAV.

    Decodes with libsndfile (`sf.read`) — the same bundled decoder on every
    platform — and falls back to `librosa.load` only for formats libsndfile
    cannot read. Stereo is downmixed to mono (channel mean); resampling uses
    `librosa.resample` (soxr) so the result is deterministic given the pinned
    librosa/soxr versions in `uv.lock`.

    Returns:
        dict with "status": "ok" | "error" and metadata.
    """
    try:
        try:
            # libsndfile path: identical decoder on Windows + Linux (bundled wheel)
            waveform, sr = sf.read(str(src_path), dtype="float32", always_2d=False)
        except Exception:
            # libsndfile cannot decode (e.g. M4A/AAC) — librosa fallback
            waveform, sr = librosa.load(str(src_path), sr=target_sr, mono=True)

        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)  # stereo → mono
        if sr != target_sr:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
        if waveform.size == 0:
            raise ValueError("decoded waveform is empty")

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(dst_path), waveform, target_sr, subtype=WAV_SUBTYPE)
        return {
            "status": "ok",
            "duration": len(waveform) / target_sr,
            "samples": len(waveform),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def convert_all(
    raw_dir: Path,
    wav_dir: Path,
    labels_csv: str = "labels.csv",
    labels_out: Optional[Path] = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    force: bool = False,
    progress: bool = True,
) -> dict:
    """
    Convert every audio file listed in `raw_dir/<labels_csv>` to mono 16 kHz WAV.

    The labels CSV is the authoritative file list (not a directory glob), so any
    extension (mp3/ogg/flac/...) is handled identically locally and on the server.
    Files whose `.wav` output already exists are skipped unless `force=True`.

    Writes `<wav_dir.parent>/<wav_dir.name>_labels.csv` (e.g.
    `data/processed/audio_wav_labels.csv`) mapping every row to its `.wav` name.

    Returns:
        dict with counts: total, converted, skipped, failed, errors (list of
        "<fname>: <error>"), labels_path.
    """
    raw_dir = Path(raw_dir)
    wav_dir = Path(wav_dir)
    labels_in = raw_dir / labels_csv
    if not labels_in.exists():
        raise FileNotFoundError(f"Raw labels not found: {labels_in}")

    df = pd.read_csv(labels_in)
    df.columns = df.columns.str.strip()
    if "audio_file" not in df.columns:
        raise ValueError(f"labels CSV missing 'audio_file' column: {labels_in}")

    # ── Build src → dst map from labels (authoritative) ──
    file_map = {}
    for fname in df["audio_file"].dropna().astype(str).unique():
        src = raw_dir / fname
        if not src.exists():
            continue
        wav_name = Path(fname).stem + ".wav"
        file_map[fname] = {"src": src, "dst": wav_dir / wav_name, "wav_name": wav_name}

    # ── Per-file skip (or force re-conversion) ──
    to_convert = {}
    skipped = 0
    for fname, paths in file_map.items():
        if paths["dst"].exists() and not force:
            skipped += 1
        else:
            to_convert[fname] = paths

    converted, failed = 0, 0
    errors = []
    items = list(to_convert.items())
    if items:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(convert_one, paths["src"], paths["dst"]): fname
                for fname, paths in items
            }
            iterator = as_completed(futures)
            if progress:
                from tqdm import tqdm

                iterator = tqdm(iterator, total=len(futures), desc="  Converting")
            for future in iterator:
                fname = futures[future]
                result = future.result()
                if result["status"] == "ok":
                    converted += 1
                else:
                    failed += 1
                    errors.append(f"{fname}: {result['error']}")

    # ── Rewrite labels pointing at .wav names (all rows) ──
    df["audio_file"] = df["audio_file"].apply(
        lambda x: Path(str(x)).stem + ".wav" if pd.notna(x) else x
    )
    if labels_out is None:
        labels_out = wav_dir.parent / f"{wav_dir.name}_labels.csv"
    df.to_csv(labels_out, index=False)

    return {
        "total": len(file_map),
        "converted": converted,
        "skipped": skipped,
        "failed": failed,
        "errors": errors,
        "labels_path": str(labels_out),
    }
