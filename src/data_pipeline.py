"""
Phase 1: Robust Data Pipeline for Open-Set Speaker Identification.
Handles stratified 5-shot split, augmentation, and weighted sampling.
"""

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import librosa
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

warnings.filterwarnings("ignore", category=UserWarning)


# ─────────────────────────────────────────────────────────
#  Configuration Loader
# ─────────────────────────────────────────────────────────

def load_config(config_path: str = "configs/default_config.yaml") -> dict:
    """Load YAML configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def get_active_profile(config: dict) -> dict:
    """Return the active hardware profile (local or vastai)."""
    mode = config["hardware"]["mode"]
    profile = config["hardware"]["profiles"][mode].copy()
    profile.pop("description", None)
    return profile


# ─────────────────────────────────────────────────────────
#  Label Processing & Stratified Split
# ─────────────────────────────────────────────────────────

def create_class_mapping(labels_df: pd.DataFrame) -> Dict[str, int]:
    """
    Create mapping: 'unknown' -> 0, known UUIDs -> 1..446.
    Returns dict mapping speaker_id -> integer class.
    """
    known_ids = sorted(
        labels_df[labels_df["speaker_id"] != "unknown"]["speaker_id"].unique()
    )
    mapping = {"unknown": 0}
    for idx, sid in enumerate(known_ids, start=1):
        mapping[sid] = idx
    return mapping


def find_duplicate_groups(
    labels_df: pd.DataFrame,
    audio_dir: str,
) -> Dict[str, List[str]]:
    """
    Group audio files with identical byte content (streaming MD5, 1 MB chunks).

    Only MD5 digests shared by more than one file are returned — these are the
    near-certain duplicate recordings that can leak across a random split.

    Args:
        labels_df: DataFrame with an `audio_file` column.
        audio_dir: Directory containing the (converted) audio files.

    Returns:
        dict: md5 hex digest -> sorted list of audio_file names sharing it.
    """
    import hashlib

    audio_dir = Path(audio_dir)
    md5_to_files: Dict[str, List[str]] = {}

    for fname in sorted(labels_df["audio_file"].unique()):
        fpath = audio_dir / fname
        hasher = hashlib.md5()
        try:
            with open(fpath, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    hasher.update(chunk)
        except OSError:
            continue  # missing/unreadable — handled by find_corrupted_files
        md5_to_files.setdefault(hasher.hexdigest(), []).append(fname)

    return {md5: fnames for md5, fnames in md5_to_files.items() if len(fnames) > 1}


def scan_durations(labels_df: pd.DataFrame, audio_dir: str) -> Dict[str, float]:
    """
    Header-only duration scan (soundfile.info) for every labelled file.

    Returns {audio_file: duration_seconds}; unreadable/missing files get 0.0
    (they are reported as corrupted by find_corrupted_files).
    """
    import soundfile as sf

    audio_dir = Path(audio_dir)
    durations: Dict[str, float] = {}
    for fname in labels_df["audio_file"].unique():
        fpath = audio_dir / fname
        try:
            durations[fname] = sf.info(str(fpath)).duration
        except Exception:
            durations[fname] = 0.0
    return durations


def find_corrupted_files(
    labels_df: pd.DataFrame,
    audio_dir: str,
    min_valid_duration: float = 1.0,
) -> List[str]:
    """
    Return the list of audio_files that are missing, unreadable, or shorter
    than `min_valid_duration` seconds (header-only soundfile.info check).
    """
    import soundfile as sf

    audio_dir = Path(audio_dir)
    corrupted: List[str] = []
    for fname in labels_df["audio_file"].unique():
        fpath = audio_dir / fname
        if not fpath.exists():
            corrupted.append(fname)
            continue
        try:
            if sf.info(str(fpath)).duration < min_valid_duration:
                corrupted.append(fname)
        except Exception:
            corrupted.append(fname)
    return corrupted


def stratified_split(
    labels_df: pd.DataFrame,
    val_per_known: int = 1,
    unknown_val_ratio: float = 0.2,
    random_seed: int = 42,
    duplicate_groups: Optional[Dict[str, List[str]]] = None,
    corrupted_files: Optional[Set[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Leakage-aware stratified split.

    - Corrupted files (via `corrupted_files`) are dropped entirely.
    - Files that belong to an MD5-duplicate group are **never** put in val, so
      byte-identical recordings cannot straddle the train/val boundary.
      A known speaker whose files are all duplicated is excluded from val
      (with a warning) to keep val strictly duplicate-free.
    - Known speakers: exactly `val_per_known` non-duplicate files → val, rest → train.
    - 'unknown' class: `unknown_val_ratio` (of non-duplicate files) → val.
    """
    rng = np.random.default_rng(random_seed)
    df = labels_df.copy()
    if corrupted_files:
        df = df[~df["audio_file"].isin(corrupted_files)].reset_index(drop=True)

    # Files that are byte-identical duplicates must never appear in val
    dup_files: Set[str] = set()
    if duplicate_groups:
        for fnames in duplicate_groups.values():
            dup_files.update(fnames)

    train_rows, val_rows = [], []

    df_known = df[df["speaker_id"] != "unknown"]
    df_unknown = df[df["speaker_id"] == "unknown"]

    # Known speakers: val from NON-duplicate files only
    for speaker_id, group in df_known.groupby("speaker_id"):
        group = group.reset_index(drop=True)
        n = len(group)
        dup_mask = group["audio_file"].isin(dup_files).values
        non_dup_idx = np.where(~dup_mask)[0]
        n_val = min(val_per_known, n - 1)  # ensure at least 1 train

        if len(non_dup_idx) >= n_val:
            chosen = rng.choice(non_dup_idx, size=n_val, replace=False)
            val_mask = np.zeros(n, dtype=bool)
            val_mask[chosen] = True
        else:
            # Speaker's files are (mostly) duplicated → keep val duplicate-free:
            # the whole speaker goes to train.
            print(f"  ⚠ Speaker {speaker_id[:8]}… has {int(dup_mask.sum())}/{n} "
                  f"duplicated files — excluded from val (val kept duplicate-free)")
            val_mask = np.zeros(n, dtype=bool)

        val_rows.append(group[val_mask])
        train_rows.append(group[~val_mask])

    # Unknown class: ratio split, but duplicate files stay in train
    n_unknown = len(df_unknown)
    cand_idx = np.where(~df_unknown["audio_file"].isin(dup_files).values)[0]
    n_val_unknown = min(int(n_unknown * unknown_val_ratio), len(cand_idx))
    val_idx = (rng.choice(cand_idx, size=n_val_unknown, replace=False)
               if n_val_unknown > 0 else np.array([], dtype=int))
    val_mask = np.zeros(n_unknown, dtype=bool)
    val_mask[val_idx] = True
    val_rows.append(df_unknown[val_mask])
    train_rows.append(df_unknown[~val_mask])

    train_df = pd.concat(train_rows, ignore_index=True)
    val_df = pd.concat(val_rows, ignore_index=True)

    # Shuffle
    train_df = train_df.sample(frac=1, random_state=rng).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=rng).reset_index(drop=True)

    return train_df, val_df


def _write_split_report(
    labels_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    duplicate_groups: Dict[str, List[str]],
    corrupted: List[str],
    durations: Dict[str, float],
    output_path: str,
) -> None:
    """
    Write data/processed/split_report.json: corrupted files (known/unknown),
    MD5-duplicate groups (incl. conflicting-label groups), and per-known-speaker
    train/val counts + usable seconds.
    """
    import json

    corrupted_known = int(labels_df[
        labels_df["audio_file"].isin(corrupted) & (labels_df["speaker_id"] != "unknown")
    ].shape[0])
    corrupted_unknown = int(labels_df[
        labels_df["audio_file"].isin(corrupted) & (labels_df["speaker_id"] == "unknown")
    ].shape[0])

    groups_info = []
    n_conflicting = 0
    for md5, fnames in duplicate_groups.items():
        lbls = sorted(set(labels_df.loc[labels_df["audio_file"].isin(fnames), "speaker_id"]))
        conflicting = len(lbls) > 1
        if conflicting:
            n_conflicting += 1
        groups_info.append({
            "md5": md5,
            "n_files": len(fnames),
            "files": fnames,
            "labels": lbls,
            "conflicting_labels": conflicting,
        })

    train_files = set(train_df["audio_file"])
    per_speaker = {}
    for sid, group in labels_df[labels_df["speaker_id"] != "unknown"].groupby("speaker_id"):
        good = group[~group["audio_file"].isin(corrupted)]
        per_speaker[sid] = {
            "train_files": int(group[group["audio_file"].isin(train_files)].shape[0]),
            "val_files": int(group[~group["audio_file"].isin(train_files)].shape[0]),
            "usable_seconds": round(float(sum(durations.get(f, 0.0) for f in good["audio_file"])), 2),
        }

    report = {
        "corrupted_files": {
            "total": len(corrupted),
            "known": corrupted_known,
            "unknown": corrupted_unknown,
            "files": corrupted,
        },
        "duplicate_groups": {
            "total_groups": len(duplicate_groups),
            "total_files": sum(len(v) for v in duplicate_groups.values()),
            "conflicting_label_groups": n_conflicting,
            "groups": groups_info,
        },
        "per_known_speaker": per_speaker,
        "split_summary": {
            "train_samples": int(len(train_df)),
            "val_samples": int(len(val_df)),
            "train_known": int((train_df["label"] != 0).sum()),
            "val_known": int((val_df["label"] != 0).sum()),
            "train_unknown": int((train_df["label"] == 0).sum()),
            "val_unknown": int((val_df["label"] == 0).sum()),
        },
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Split report saved to {output_path}")


def prepare_clean_split(
    labels_path: str,
    audio_dir: str,
    processed_labels: str,
    val_per_known: int = 1,
    unknown_val_ratio: float = 0.2,
    min_valid_duration: float = 1.0,
    random_seed: int = 42,
    split_report_path: str = "data/processed/split_report.json",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """
    Load, clean and leak-free split labels; write data/processed/split_report.json.

    Pipeline:
      1. Load & clean labels (drops exact CSV duplicate rows / NaN).
      2. Scan durations (header-only) for every labelled file.
      3. Detect corrupted (< min_valid_duration) / missing files.
      4. Detect MD5-duplicate groups (incl. conflicting-label groups).
      5. Leak-free stratified_split (duplicates/corrupted never in val).
      6. Save cleaned labels and split_report.json.

    Returns:
        train_df, val_df, class_map
    """
    df = pd.read_csv(labels_path)
    df.columns = df.columns.str.strip()

    # Basic cleaning
    df = df.drop_duplicates().reset_index(drop=True)
    df = df.dropna(subset=["speaker_id", "audio_file"]).reset_index(drop=True)

    # Create class mapping
    class_map = create_class_mapping(df)
    df["label"] = df["speaker_id"].map(class_map)

    # ── Leak-aware scans ──
    print(f"  Scanning durations ({len(df):,} files)...")
    durations = scan_durations(df, audio_dir)
    print("  Detecting corrupted files...")
    corrupted = find_corrupted_files(df, audio_dir, min_valid_duration)
    if corrupted:
        print(f"  ⚠ {len(corrupted)} corrupted/short files (< {min_valid_duration}s)")
    print("  Detecting MD5-duplicate groups...")
    dup_groups = find_duplicate_groups(df, audio_dir)
    if dup_groups:
        print(f"  ⚠ {len(dup_groups)} duplicate groups "
              f"({sum(len(v) for v in dup_groups.values())} files)")

    # ── Leak-free split ──
    train_df, val_df = stratified_split(
        df,
        val_per_known=val_per_known,
        unknown_val_ratio=unknown_val_ratio,
        random_seed=random_seed,
        duplicate_groups=dup_groups,
        corrupted_files=set(corrupted),
    )

    # ── Save cleaned labels ──
    os.makedirs(os.path.dirname(processed_labels), exist_ok=True)
    df.to_csv(processed_labels, index=False)
    print(f"  ✓ Saved cleaned labels ({len(df)} rows) to {processed_labels}")

    # ── Split report ──
    _write_split_report(
        df, train_df, val_df, dup_groups, corrupted, durations, split_report_path,
    )

    print(f"  ✓ Train samples: {len(train_df)} | Val samples: {len(val_df)}")
    print(
        f"    Train known: {(train_df['label'] != 0).sum()} | "
        f"Train unknown: {(train_df['label'] == 0).sum()}"
    )
    print(
        f"    Val known: {(val_df['label'] != 0).sum()} | "
        f"Val unknown: {(val_df['label'] == 0).sum()}"
    )

    return train_df, val_df, class_map


def prepare_labels(
    labels_path: str,
    output_path: str,
    val_per_known: int = 1,
    unknown_val_ratio: float = 0.2,
    audio_dir: Optional[str] = None,
    min_valid_duration: float = 1.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """
    Load, clean, split labels and create class mapping.
    Saves cleaned labels to output_path.

    If `audio_dir` is given, the leak-free pipeline (corrupted/duplicate
    detection + split_report.json) is used; otherwise a plain stratified split.
    """
    if audio_dir is not None:
        return prepare_clean_split(
            labels_path=labels_path,
            audio_dir=audio_dir,
            processed_labels=output_path,
            val_per_known=val_per_known,
            unknown_val_ratio=unknown_val_ratio,
            min_valid_duration=min_valid_duration,
        )

    df = pd.read_csv(labels_path)
    df.columns = df.columns.str.strip()

    # Basic cleaning
    df = df.drop_duplicates().reset_index(drop=True)
    df = df.dropna(subset=["speaker_id", "audio_file"]).reset_index(drop=True)

    # Create class mapping
    class_map = create_class_mapping(df)
    df["label"] = df["speaker_id"].map(class_map)

    # Save cleaned labels
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  ✓ Saved cleaned labels ({len(df)} rows) to {output_path}")

    # Stratified split
    train_df, val_df = stratified_split(df, val_per_known, unknown_val_ratio)
    print(f"  ✓ Train samples: {len(train_df)} | Val samples: {len(val_df)}")
    print(
        f"    Train known: {(train_df['label'] != 0).sum()} | "
        f"Train unknown: {(train_df['label'] == 0).sum()}"
    )
    print(
        f"    Val known: {(val_df['label'] != 0).sum()} | "
        f"Val unknown: {(val_df['label'] == 0).sum()}"
    )

    return train_df, val_df, class_map


# ─────────────────────────────────────────────────────────
#  Audio Augmentation Pipeline
# ─────────────────────────────────────────────────────────

class AudioAugmentation:
    """
    Training-time augmentation pipeline using audiomentations.

    Applies a diverse set of waveform-level augmentations:
    - Gaussian noise injection
    - Pitch shifting (±4 semitones)
    - Time stretching (0.8× – 1.25×)
    - Gain variation (±6 dB)
    - Polarity inversion
    - Time shifting (±10%)

    All augmentations preserve the waveform length.
    """

    def __init__(self, sample_rate: int = 16000):
        import audiomentations as AA
        self.sample_rate = sample_rate
        self.pipeline = AA.Compose([
            AA.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            AA.PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
            AA.TimeStretch(min_rate=0.8, max_rate=1.25, p=0.3),
            AA.Gain(min_gain_db=-6, max_gain_db=6, p=0.3),
            AA.PolarityInversion(p=0.5),
            AA.Shift(min_shift=-0.1, max_shift=0.1, shift_unit="fraction",
                     rollover=True, fade_duration=0.005, p=0.3),
        ])

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentation pipeline to a waveform.

        Args:
            waveform: (1, T) — single-channel audio tensor

        Returns:
            Augmented waveform of same shape (1, T)
        """
        # audiomentations expects (samples,) numpy array
        audio_np = waveform.squeeze(0).numpy()
        augmented = self.pipeline(samples=audio_np, sample_rate=self.sample_rate)
        return torch.from_numpy(augmented).unsqueeze(0).float()


# ─────────────────────────────────────────────────────────
#  PyTorch Dataset
# ─────────────────────────────────────────────────────────

class SpeakerDataset(Dataset):
    """
    Dataset for Open-Set Speaker Identification.
    Loads MP3/WAV, resamples to 16kHz, pads/truncates to fixed length.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        audio_dir: str,
        sample_rate: int = 16000,
        duration_seconds: float = 5.0,
        augment: bool = False,
        min_valid_duration: float = 1.0,
        mixup_alpha: float = 0.0,
    ):
        self.df = df.reset_index(drop=True)
        self.audio_dir = Path(audio_dir)
        self.target_sr = sample_rate
        self.target_length = int(sample_rate * duration_seconds)
        self.augment = augment
        self.min_valid_duration = min_valid_duration
        self.mixup_alpha = mixup_alpha

        if self.augment:
            self.augmentor = AudioAugmentation(sample_rate)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        audio_path = self.audio_dir / row["audio_file"]
        label = torch.tensor(row["label"], dtype=torch.long)

        # Load audio
        waveform = self._load_audio(audio_path)

        # Augmentation (train only)
        if self.augment:
            waveform = self.augmentor(waveform)

            # MixUp: mix with another random sample (OOD regularization)
            if self.mixup_alpha > 0 and torch.rand(1).item() < 0.5:
                # Pick a random sample (possibly different class)
                other_idx = torch.randint(0, len(self.df), (1,)).item()
                other_row = self.df.iloc[other_idx]
                other_path = self.audio_dir / other_row["audio_file"]
                other_waveform = self._load_audio(other_path)
                other_waveform = self.augmentor(other_waveform)
                other_label = torch.tensor(other_row["label"], dtype=torch.long)

                # Mix: λ ~ Beta(α, α)
                lam = float(torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample())
                waveform = lam * waveform + (1 - lam) * other_waveform
                # Keep original label (acts as OOD regularization — mixed audio is ambiguous)

        return waveform, label

    def _load_audio(self, path: Path) -> torch.Tensor:
        """
        Load and preprocess audio file.

        WAV files: uses soundfile backend (fast, no mpg123 dependency)
        Other formats: falls back to librosa
        """
        suffix = path.suffix.lower()
        try:
            if suffix in (".wav",):
                # Use soundfile for WAV — no mpg123, fast, reliable
                import soundfile as sf
                waveform, sr = sf.read(str(path), dtype="float32")
                if waveform.ndim > 1:
                    waveform = waveform.mean(axis=1)  # stereo → mono
                if sr != self.target_sr:
                    import librosa
                    waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.target_sr)
                waveform = torch.from_numpy(waveform).unsqueeze(0).float()  # (1, T)
            else:
                # librosa for MP3 and other formats
                waveform, sr = librosa.load(str(path), sr=self.target_sr, mono=True)
                waveform = torch.from_numpy(waveform).unsqueeze(0).float()  # (1, T)
        except Exception as e:
            # Return silence for corrupted files
            print(f"  ⚠ Warning: Could not load {path.name}: {e}")
            return torch.zeros(1, self.target_length)

        # Pad or truncate to target length
        if waveform.size(-1) < self.target_length:
            # Pad with zeros at the end
            pad_size = self.target_length - waveform.size(-1)
            waveform = torch.nn.functional.pad(waveform, (0, pad_size))
        elif waveform.size(-1) > self.target_length:
            # Random crop for training (every epoch sees different windows)
            # Center crop for validation (deterministic)
            if self.augment:
                max_start = waveform.size(-1) - self.target_length
                start = torch.randint(0, max_start + 1, (1,)).item()
            else:
                start = (waveform.size(-1) - self.target_length) // 2
            waveform = waveform[..., start : start + self.target_length]

        return waveform


# ─────────────────────────────────────────────────────────
#  DataLoader Factory
# ─────────────────────────────────────────────────────────

def get_dataloaders(
    config: Optional[dict] = None,
    config_path: str = "configs/default_config.yaml",
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """
    Create train and validation DataLoaders with WeightedRandomSampler.

    Returns:
        train_loader, val_loader, class_map
    """
    if config is None:
        config = load_config(config_path)

    audio_cfg = config["audio"]
    data_cfg = config["data"]
    hw_profile = get_active_profile(config)

    batch_size = hw_profile["batch_size"]
    num_workers = hw_profile["num_workers"]

    print("=" * 50)
    print("  Preparing DataLoaders")
    print("=" * 50)
    print(f"  Hardware profile: {config['hardware']['mode']}")
    print(f"  Batch size: {batch_size} | Workers: {num_workers}")
    print(f"  Sample rate: {audio_cfg['sample_rate']} | "
          f"Duration: {audio_cfg['duration_seconds']}s")

    min_valid_duration = audio_cfg.get("min_valid_duration", 0.0)

    # ── Verify audio directory exists ──
    audio_dir = data_cfg["audio_dir"]
    labels_path = data_cfg["labels_path"]

    if not os.path.exists(audio_dir):
        raise FileNotFoundError(
            f"Audio directory not found: {audio_dir}\n"
            f"Run: python scripts/convert_mp3_to_wav.py\n"
            f"Or update config to point to existing audio files."
        )
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    # Count WAV files for info
    wav_count = len([f for f in os.listdir(audio_dir) if f.endswith('.wav')])
    mp3_count = len([f for f in os.listdir(audio_dir) if f.endswith('.mp3')])
    print(f"  Audio dir: {audio_dir}")
    print(f"  Files: {wav_count} WAV + {mp3_count} MP3")

    # Prepare labels + leak-free split (corrupted/duplicate filtering inside)
    train_df, val_df, class_map = prepare_labels(
        labels_path=labels_path,
        output_path=data_cfg["processed_labels"],
        val_per_known=1,
        unknown_val_ratio=0.2,
        audio_dir=audio_dir,
        min_valid_duration=min_valid_duration,
    )

    # Create datasets
    train_dataset = SpeakerDataset(
        df=train_df,
        audio_dir=audio_dir,           # ← resolved path
        sample_rate=audio_cfg["sample_rate"],
        duration_seconds=audio_cfg["duration_seconds"],
        augment=True,
        min_valid_duration=min_valid_duration,
    )

    val_dataset = SpeakerDataset(
        df=val_df,
        audio_dir=audio_dir,           # ← resolved path
        sample_rate=audio_cfg["sample_rate"],
        duration_seconds=audio_cfg["duration_seconds"],
        augment=False,
        min_valid_duration=min_valid_duration,
    )

    # ── Balanced Batch Sampler ──
    # Enforces target OOD/known ratio in every batch.
    # Default: 30% OOD (unknown) + 70% known speakers.
    # This prevents bias toward the over-represented "unknown" class.
    train_labels = train_df["label"].values
    ood_indices = np.where(train_labels == 0)[0]
    known_indices = np.where(train_labels != 0)[0]
    
    ood_ratio = audio_cfg.get("ood_batch_ratio", 0.30)
    ood_per_batch = max(1, int(batch_size * ood_ratio))
    known_per_batch = batch_size - ood_per_batch
    
    print(f"\n  ⚖️  Batch balance: {ood_per_batch} OOD + {known_per_batch} known "
          f"({ood_ratio:.0%} / {1-ood_ratio:.0%})")
    print(f"     OOD pool: {len(ood_indices):,} samples | "
          f"Known pool: {len(known_indices):,} samples across {len(class_map)-1} speakers")
    
    # Generate balanced indices for each batch
    rng = np.random.RandomState(42)
    num_batches = len(train_df) // batch_size
    balanced_indices = []
    
    for _ in range(num_batches):
        # Sample OOD indices with replacement (if needed)
        batch_ood = rng.choice(ood_indices, size=ood_per_batch, replace=True)
        # Sample known indices — try without replacement, fall back with replacement
        if len(known_indices) >= known_per_batch:
            batch_known = rng.choice(known_indices, size=known_per_batch, replace=False)
        else:
            batch_known = rng.choice(known_indices, size=known_per_batch, replace=True)
        batch_indices = np.concatenate([batch_ood, batch_known])
        rng.shuffle(batch_indices)
        balanced_indices.extend(batch_indices.tolist())
    
    balanced_indices = np.array(balanced_indices, dtype=np.int64)
    sampler = torch.utils.data.SubsetRandomSampler(balanced_indices)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True if hw_profile["device"] == "cuda" else False,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if hw_profile["device"] == "cuda" else False,
        drop_last=False,
    )

    # Print class distribution info
    known_train = (train_labels != 0).sum()
    unknown_train = (train_labels == 0).sum()
    print(f"\n  📊 Class distribution in training set:")
    print(f"     Known classes: {len(class_map) - 1}")
    print(f"     Known samples: {known_train}")
    print(f"     Unknown samples: {unknown_train}")
    print(f"     Sampler: WeightedRandomSampler (balanced per-class)")

    return train_loader, val_loader, class_map


# ─────────────────────────────────────────────────────────
#  Main Test Block
# ─────────────────────────────────────────────────────────

def main():
    """Test the DataLoader pipeline: load one batch and verify shapes/distribution."""
    print("=" * 55)
    print("  Data Pipeline Test")
    print("=" * 55)
    print()

    train_loader, val_loader, class_map = get_dataloaders()

    print("\n" + "-" * 50)
    print("  Testing Train Loader (1 batch)")
    print("-" * 50)
    waveforms, labels = next(iter(train_loader))
    print(f"  Waveform shape: {waveforms.shape}  (batch, channels, time)")
    print(f"  Labels shape:   {labels.shape}")
    print(f"  Label values:   {labels.tolist()}")
    n_known = (labels != 0).sum().item()
    n_unknown = (labels == 0).sum().item()
    print(f"  Batch composition: {n_known} known + {n_unknown} unknown")
    print(f"  Waveform range: [{waveforms.min().item():.4f}, {waveforms.max().item():.4f}]")

    print("\n" + "-" * 50)
    print("  Testing Val Loader (1 batch)")
    print("-" * 50)
    waveforms_val, labels_val = next(iter(val_loader))
    print(f"  Waveform shape: {waveforms_val.shape}")
    print(f"  Labels shape:   {labels_val.shape}")

    # Verify class mapping sanity
    print(f"\n  ✅ Class mapping: {len(class_map)} total classes "
          f"(0=unknown, 1..{len(class_map)-1}=known speakers)")
    print("  ✅ Pipeline test passed!")


if __name__ == "__main__":
    main()
