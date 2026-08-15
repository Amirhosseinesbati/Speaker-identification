"""
Q5 — Clean label noise from MD5-duplicate groups (quick win, 0 GPU).

Drops byte-identical files that carry contradictory labels (the 46-file group
with 4 labels + the 2-file group with 2 labels) and dedupes non-conflicting
duplicate groups to a single copy. Produces a cleaned labels CSV for future
training (prerequisite for the Phase-2 full fine-tune, C1).

The class mapping is unchanged: the 446 known speaker UUIDs are untouched, so
the cleaned file stays compatible with the checkpoints' embedded ``class_map``.

Usage:
    uv run --no-sync python scripts/clean_labels_conflicts.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.cli_utils import setup_utf8_stdio  # noqa: E402
setup_utf8_stdio()

from src.data_pipeline import clean_conflicting_labels  # noqa: E402

LABELS_PATH = ROOT / "data" / "processed" / "audio_wav_labels.csv"
AUDIO_DIR = ROOT / "data" / "processed" / "audio_wav"
OUT_CSV = ROOT / "data" / "processed" / "labels_cleaned_noconflict.csv"
OUT_JSON = ROOT / "data" / "processed" / "label_cleanup_stats.json"


def main() -> int:
    print("=" * 60)
    print("  Q5 — Conflicting-duplicate label cleanup")
    print("=" * 60)

    df = pd.read_csv(LABELS_PATH)
    df.columns = df.columns.str.strip()
    df["speaker_id"] = df["speaker_id"].astype(str).str.strip()
    df["audio_file"] = df["audio_file"].astype(str).str.strip()
    print(f"  Input labels: {len(df):,} rows ({LABELS_PATH.name})")

    cleaned, stats = clean_conflicting_labels(df, str(AUDIO_DIR))

    n_known = int((cleaned["speaker_id"] != "unknown").sum())
    n_unknown = int((cleaned["speaker_id"] == "unknown").sum())
    n_speakers = int(cleaned[cleaned["speaker_id"] != "unknown"]["speaker_id"].nunique())
    print(f"  Conflicting files dropped : {stats['n_conflicting_files_dropped']}")
    print(f"  Non-conflicting deduped   : {stats['n_nonconflicting_duplicates_dropped']}")
    print(f"  After clean               : {stats['n_files_after_clean']:,} "
          f"({n_known} known / {n_unknown} unknown, {n_speakers} speakers)")

    cleaned.to_csv(OUT_CSV, index=False)
    print(f"  ✓ Saved cleaned labels to {OUT_CSV}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps({"input": str(LABELS_PATH), **stats,
                    "n_known": n_known, "n_unknown": n_unknown,
                    "n_known_speakers": n_speakers},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  ✓ Stats saved to {OUT_JSON}")
    print("\n✅ Label cleanup complete (class mapping unchanged).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
