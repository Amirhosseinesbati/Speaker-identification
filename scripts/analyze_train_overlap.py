"""
Leak-fairness analysis: how much does each model's TRAIN set overlap the
reference val split?

The old checkpoint (campp_best.pt) was trained on the single-scheme train
split — its val (1 held-out file/speaker) is clean by construction.
The new checkpoint (campp_best (2).pt) embeds a kfold (fold 0/3, seed 42)
split in its config, so its training set may contain EXACT files that appear
in the reference val set (a speaker's "single-scheme val file" is one of the
speaker's files, and kfold keeps all of a speaker's files in the same fold —
many of those speakers are in the new model's training fold).

This script quantifies that overlap so the comparison can be interpreted
honestly.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.cli_utils import setup_utf8_stdio  # noqa: E402
setup_utf8_stdio()

from src.data_pipeline import prepare_clean_split  # noqa: E402

DATA = ROOT / "data" / "processed"
LABELS_PATH = DATA / "audio_wav_labels.csv"
AUDIO_DIR = DATA / "audio_wav"
PROCESSED_LABELS = DATA / "cleaned_labels.csv"


def main():
    # Reference single-scheme split (same as the comparison script)
    train_single, val_single, _ = prepare_clean_split(
        labels_path=str(LABELS_PATH), audio_dir=str(AUDIO_DIR),
        processed_labels=str(PROCESSED_LABELS),
        val_per_known=1, unknown_val_ratio=0.2, random_seed=42,
    )
    val_files = set(val_single["audio_file"])
    print(f"Reference val files: {len(val_files):,} "
          f"(known={int((val_single['label'] > 0).sum())}, "
          f"unknown={int((val_single['label'] == 0).sum())})")

    # New model's kfold fold-0 train split
    train_kfold, val_kfold, _ = prepare_clean_split(
        labels_path=str(LABELS_PATH), audio_dir=str(AUDIO_DIR),
        processed_labels=str(PROCESSED_LABELS),
        val_per_known=1, unknown_val_ratio=0.2, random_seed=42,
        split_scheme="kfold", fold=0, folds=3,
    )
    kfold_train_files = set(train_kfold["audio_file"])
    kfold_val_files = set(val_kfold["audio_file"])
    print(f"Kfold fold-0 train files: {len(kfold_train_files):,} "
          f"| val files: {len(kfold_val_files):,}")

    # Overlap: reference-val files that the NEW model saw verbatim in training
    overlap = val_files & kfold_train_files
    overlap_df = val_single[val_single["audio_file"].isin(overlap)]
    print(f"\nNEW model: reference-val files that are EXACT train duplicates: "
          f"{len(overlap):,} / {len(val_files):,} "
          f"({100.0 * len(overlap) / len(val_files):.1f}%)")
    print(f"  of which known: {int((overlap_df['label'] > 0).sum())} "
          f"| unknown: {int((overlap_df['label'] == 0).sum())}")

    # Old model's train vs its own val (should be zero overlap by construction)
    old_overlap = val_files & set(train_single["audio_file"])
    print(f"OLD model: reference-val files that are EXACT train duplicates: "
          f"{len(old_overlap):,} (expect 0)")

    # Files in the reference val that the NEW model truly never saw (not in its
    # kfold train). These give an unbiased sub-evaluation for the new model.
    unseen = val_files - kfold_train_files
    unseen_df = val_single[val_single["audio_file"].isin(unseen)]
    print(f"\nNEW model: reference-val files NOT in its kfold train "
          f"(truly unseen): {len(unseen):,} / {len(val_files):,} "
          f"({100.0 * len(unseen) / len(val_files):.1f}%)")
    print(f"  of which known: {int((unseen_df['label'] > 0).sum())} "
          f"| unknown: {int((unseen_df['label'] == 0).sum())}")
    n_unseen_known = int((unseen_df["label"] > 0).sum())

    out = DATA / "train_overlap_analysis.json"
    out.write_text(
        f"{{\n"
        f'  "reference_val_files": {len(val_files)},\n'
        f'  "new_model_exact_duplicates_in_val": {len(overlap)},\n'
        f'  "new_model_dup_known": {int((overlap_df["label"] > 0).sum())},\n'
        f'  "new_model_dup_unknown": {int((overlap_df["label"] == 0).sum())},\n'
        f'  "old_model_exact_duplicates_in_val": {len(old_overlap)},\n'
        f'  "new_model_unseen_val_files": {len(unseen)},\n'
        f'  "new_model_unseen_known": {n_unseen_known},\n'
        f'  "new_model_unseen_unknown": {len(unseen) - n_unseen_known}\n'
        f"}}\n",
        encoding="utf-8",
    )
    print(f"\n  ✓ Saved to {out}")


if __name__ == "__main__":
    main()
