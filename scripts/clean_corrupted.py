"""
Clean corrupted/short audio files from the processed dataset.

Scans data/processed/audio_wav/ for files shorter than min_valid_duration,
reports them, and provides commands to remove them.

Usage:
    python scripts/clean_corrupted.py [--audio-dir data/processed/audio_wav] [--min-duration 1.0] [--dry-run]
"""

import argparse
import json
import os
from pathlib import Path

import librosa
from tqdm import tqdm


def scan_directory(audio_dir: str, min_duration: float = 1.0) -> list[dict]:
    """Scan all WAV files and return list of corrupted/short files."""
    audio_path = Path(audio_dir)
    if not audio_path.exists():
        print(f"❌ Directory not found: {audio_dir}")
        return []

    wav_files = sorted(audio_path.glob("*.wav"))
    print(f"Scanning {len(wav_files)} WAV files (min_duration={min_duration}s)...")

    corrupted = []
    for fpath in tqdm(wav_files, desc="  Scanning"):
        try:
            dur = librosa.get_duration(path=str(fpath))
            if dur < min_duration:
                corrupted.append({
                    "file": fpath.name,
                    "duration": round(dur, 4),
                    "size_bytes": fpath.stat().st_size,
                    "reason": "too_short",
                })
        except Exception as e:
            corrupted.append({
                "file": fpath.name,
                "duration": 0,
                "size_bytes": fpath.stat().st_size if fpath.exists() else 0,
                "reason": f"load_error: {str(e)[:80]}",
            })

    return corrupted


def main():
    parser = argparse.ArgumentParser(description="Clean corrupted audio files")
    parser.add_argument("--audio-dir", default="data/processed/audio_wav")
    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true", help="Only scan, don't delete")
    parser.add_argument("--output-json", default="data/processed/corrupted_files.json")
    args = parser.parse_args()

    # Scan
    corrupted = scan_directory(args.audio_dir, args.min_duration)

    if not corrupted:
        print("\n✅ No corrupted files found!")
        return

    # Report
    print(f"\n🔴 Found {len(corrupted)} corrupted/short files:")
    reasons = {}
    for item in corrupted:
        r = item["reason"].split(":")[0]
        reasons[r] = reasons.get(r, 0) + 1
    for reason, count in reasons.items():
        print(f"   {reason}: {count} files")

    # Save JSON
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(corrupted, f, indent=2)
    print(f"\n📋 Corrupted list saved to: {output_path}")

    # Delete
    if args.dry_run:
        print("\n💡 Dry run — use --no-dry-run to actually delete files.")
        print(f"   Would delete: {len(corrupted)} files")
        return

    deleted = 0
    for item in corrupted:
        fpath = Path(args.audio_dir) / item["file"]
        if fpath.exists():
            fpath.unlink()
            deleted += 1

    print(f"\n🗑️  Deleted {deleted}/{len(corrupted)} corrupted files.")

    # Also clean the labels CSV if it exists
    labels_csv = Path(args.audio_dir).parent / "audio_wav_labels.csv"
    if labels_csv.exists():
        import pandas as pd
        df = pd.read_csv(labels_csv)
        corrupted_names = {item["file"] for item in corrupted}
        before = len(df)
        df = df[~df["audio_file"].isin(corrupted_names)]
        after = len(df)
        if before != after:
            df.to_csv(labels_csv, index=False)
            print(f"📝 Cleaned labels: {labels_csv} ({before} → {after} rows)")


if __name__ == "__main__":
    main()
