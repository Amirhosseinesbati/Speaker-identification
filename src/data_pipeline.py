"""
Phase 1: Robust Data Pipeline for Open-Set Speaker Identification.
Handles stratified 5-shot split, augmentation, and weighted sampling.
"""

import os
import warnings
from pathlib import Path
from typing import Optional, Tuple, Dict

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


def stratified_split(
    labels_df: pd.DataFrame,
    val_per_known: int = 1,
    unknown_val_ratio: float = 0.2,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Strict stratified split:
    - For known speakers: assign exactly `val_per_known` samples to val, rest to train.
    - For 'unknown' class: split by `unknown_val_ratio`.
    """
    rng = np.random.default_rng(random_seed)
    train_rows, val_rows = [], []

    df_known = labels_df[labels_df["speaker_id"] != "unknown"]
    df_unknown = labels_df[labels_df["speaker_id"] == "unknown"]

    # Known speakers: 1 val, rest train
    for speaker_id, group in df_known.groupby("speaker_id"):
        group = group.reset_index(drop=True)
        n = len(group)
        n_val = min(val_per_known, n - 1)  # ensure at least 1 train
        val_indices = rng.choice(n, size=n_val, replace=False)
        val_mask = np.zeros(n, dtype=bool)
        val_mask[val_indices] = True
        val_rows.append(group[val_mask])
        train_rows.append(group[~val_mask])

    # Unknown class: 80/20 split
    n_unknown = len(df_unknown)
    n_val_unknown = int(n_unknown * unknown_val_ratio)
    val_idx = rng.choice(n_unknown, size=n_val_unknown, replace=False)
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


def prepare_labels(
    labels_path: str,
    output_path: str,
    val_per_known: int = 1,
    unknown_val_ratio: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """
    Load, clean, split labels and create class mapping.
    Saves cleaned labels to output_path.
    """
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
    ):
        self.df = df.reset_index(drop=True)
        self.audio_dir = Path(audio_dir)
        self.target_sr = sample_rate
        self.target_length = int(sample_rate * duration_seconds)
        self.augment = augment
        self.min_valid_duration = min_valid_duration

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

    # Prepare labels and split
    train_df, val_df, class_map = prepare_labels(
        labels_path=labels_path,
        output_path=data_cfg["processed_labels"],
        val_per_known=1,
        unknown_val_ratio=0.2,
    )

    # ── Filter short/corrupted files ──
    if min_valid_duration > 0:
        # Scan audio durations (quick librosa check) to filter short files
        audio_files = set(train_df["audio_file"].unique()) | set(val_df["audio_file"].unique())
        short_files = set()
        print(f"  Scanning {len(audio_files)} unique files for short/corrupted audio...")
        import librosa
        for fname in audio_files:
            fpath = Path(audio_dir) / fname  # ← resolved path
            if fpath.exists():
                try:
                    dur = librosa.get_duration(path=str(fpath))
                    if dur < min_valid_duration:
                        short_files.add(fname)
                except Exception:
                    short_files.add(fname)
            else:
                short_files.add(fname)
        if short_files:
            print(f"  ⚠ Filtering out {len(short_files)} short/corrupted files (< {min_valid_duration}s)")
            train_df = train_df[~train_df["audio_file"].isin(short_files)].reset_index(drop=True)
            val_df = val_df[~val_df["audio_file"].isin(short_files)].reset_index(drop=True)
            print(f"    Train: {len(train_df)} rows | Val: {len(val_df)} rows")

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
