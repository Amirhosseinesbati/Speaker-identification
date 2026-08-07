"""
Convert all MP3 files to WAV (mono, 16kHz) for reliable dataloading.

Why: On Windows, librosa/audioread uses mpg123 for MP3 decoding, which can
fail intermittently during long training runs. WAV files are read natively
by soundfile/librosa with zero external dependencies.

Output:
    data/processed/audio_wav/  — converted WAV files
    data/processed/audio_wav_labels.csv — updated labels pointing to WAV files
"""

import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

# ── Paths ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
LABELS_PATH = DATA_RAW / "labels.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "audio_wav"
OUTPUT_LABELS = PROJECT_ROOT / "data" / "processed" / "audio_wav_labels.csv"

# ── Config ──
TARGET_SR = 16000
MAX_WORKERS = 8


def convert_one(src_path: Path, dst_path: Path) -> dict:
    """
    Convert a single audio file to mono 16kHz WAV.

    Returns:
        dict with "status": "ok" | "error" and metadata
    """
    try:
        waveform, sr = librosa.load(str(src_path), sr=TARGET_SR, mono=True)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(dst_path), waveform, TARGET_SR, subtype="PCM_16")
        return {
            "status": "ok",
            "duration": len(waveform) / TARGET_SR,
            "samples": len(waveform),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def main():
    print("=" * 60)
    print("  MP3 → WAV Converter (mono 16kHz)")
    print("=" * 60)

    # ── Load labels ──
    df = pd.read_csv(LABELS_PATH)
    df.columns = df.columns.str.strip()
    print(f"\n  Labels loaded: {len(df):,} rows")

    # ── Find unique audio files ──
    unique_files = df["audio_file"].unique()
    print(f"  Unique audio files: {len(unique_files):,}")

    # ── Map MP3 → WAV paths ──
    file_map = {}
    for fname in unique_files:
        src = DATA_RAW / fname
        # Replace .mp3 extension with .wav (also handle other extensions)
        wav_name = Path(fname).stem + ".wav"
        dst = OUTPUT_DIR / wav_name
        if src.exists():
            file_map[fname] = {"src": src, "dst": dst, "wav_name": wav_name}

    print(f"  Files to convert: {len(file_map):,}")
    print(f"  Output dir: {OUTPUT_DIR}")

    # ── Check for already-converted files ──
    to_convert = {}
    already_done = 0
    for fname, paths in file_map.items():
        if paths["dst"].exists():
            already_done += 1
        else:
            to_convert[fname] = paths

    print(f"  Already converted: {already_done:,}")
    print(f"  Remaining to convert: {len(to_convert):,}")

    if not to_convert:
        print("\n  ✅ All files already converted!")
    else:
        # ── Parallel conversion ──
        print(f"\n  Converting with {MAX_WORKERS} workers...")
        success, errors = 0, 0

        items = list(to_convert.items())
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(convert_one, paths["src"], paths["dst"]): fname
                for fname, paths in items
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="  Converting"):
                fname = futures[future]
                result = future.result()
                if result["status"] == "ok":
                    success += 1
                else:
                    errors += 1
                    print(f"\n  ⚠ Error converting {fname}: {result['error']}")

        print(f"\n  ✅ Converted: {success:,} | Errors: {errors:,}")

    # ── Build updated labels CSV ──
    wav_name_map = {fname: paths["wav_name"] for fname, paths in file_map.items()}
    df["audio_file"] = df["audio_file"].map(wav_name_map)
    df.to_csv(OUTPUT_LABELS, index=False)
    print(f"  [SAVED] Updated labels: {OUTPUT_LABELS}")
    print(f"          ({len(df):,} rows, pointing to .wav files)")

    # ── Summary ──
    total_wavs = len(list(OUTPUT_DIR.glob("*.wav")))
    print(f"\n  📁 Total WAV files in output: {total_wavs:,}")
    print(f"\n{'='*60}")
    print(f"  ✅ Conversion complete!")
    print(f"  Update config: data.audio_dir: data/processed/audio_wav")
    print(f"  Update config: data.labels_path: data/processed/audio_wav_labels.csv")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
