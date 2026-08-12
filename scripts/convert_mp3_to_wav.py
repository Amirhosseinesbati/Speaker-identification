"""
Convert all MP3 files to WAV (mono, 16kHz) for reliable dataloading.

Thin CLI wrapper around `src.audio_preprocessing.convert_all` — the SAME code
path used by the ZenML server step (`src/pipelines/steps.py::convert_audio`),
so local and server produce byte-identical WAVs (libsndfile decoder, no ffmpeg
dependency).

Why WAV at all: on Windows, librosa/audioread uses mpg123 for MP3 decoding,
which can fail intermittently during long training runs. WAV files are read
natively by soundfile/librosa with zero external dependencies.

Usage:
    uv run --no-sync python scripts/convert_mp3_to_wav.py [--force]

Output:
    data/processed/audio_wav/  — converted WAV files
    data/processed/audio_wav_labels.csv — updated labels pointing to WAV files
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio_preprocessing import convert_all  # noqa: E402

DATA_RAW = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "audio_wav"
OUTPUT_LABELS = PROJECT_ROOT / "data" / "processed" / "audio_wav_labels.csv"


def main():
    parser = argparse.ArgumentParser(
        description="Convert raw audio to mono 16 kHz PCM-16 WAV (unified pipeline)."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-convert even if the WAV output already exists",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  MP3 → WAV Converter (mono 16kHz) — unified pipeline")
    print("=" * 60)
    print(f"  Raw dir: {DATA_RAW}")
    print(f"  Output:  {OUTPUT_DIR}")

    stats = convert_all(
        raw_dir=DATA_RAW,
        wav_dir=OUTPUT_DIR,
        labels_out=OUTPUT_LABELS,
        force=args.force,
    )

    print(f"\n  ✅ Converted: {stats['converted']:,} | "
          f"Skipped: {stats['skipped']:,} | Failed: {stats['failed']:,}")
    for err in stats["errors"]:
        print(f"  ⚠ {err}")
    print(f"  [SAVED] Updated labels: {OUTPUT_LABELS}")
    total_wavs = len(list(OUTPUT_DIR.glob("*.wav")))
    print(f"  📁 Total WAV files in output: {total_wavs:,}")
    print(f"\n{'='*60}")
    print("  ✅ Conversion complete!")
    print("  Update config: data.audio_dir: data/processed/audio_wav")
    print("  Update config: data.labels_path: data/processed/audio_wav_labels.csv")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
