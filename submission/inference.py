"""
Phase 4: Inference & Submission Script for IAAA 2026 Speaker Identification.

CLI Interface (competition-mandated format):
    python submission/inference.py \\
        --data-dir /path/to/test-set \\
        --predictions-file-path /path/to/submission.csv

Output CSV columns: `id` (audio filename) + `0` through `446` (probabilities)
The sum of probabilities for each row equals exactly 1.0.
"""

import os
import sys
import warnings
from pathlib import Path
from typing import List, Optional

import click
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
import librosa
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import TwoHeadedSpeakerModel
from src.model_factory import create_model_from_config

warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────

SAMPLE_RATE = 16000
DURATION_SECONDS = 3.0
TARGET_LENGTH = int(SAMPLE_RATE * DURATION_SECONDS)  # 48000
NUM_CLASSES = 448  # 0 (unknown) + 447 known speakers (competition spec)


# ─────────────────────────────────────────────────────────
#  Audio Processing
# ─────────────────────────────────────────────────────────

def load_audio(file_path: str) -> Optional[np.ndarray]:
    """
    Load audio file and resample to 16kHz mono.
    Returns numpy array of shape (T,) or None on failure.
    """
    try:
        waveform, _ = librosa.load(str(file_path), sr=SAMPLE_RATE, mono=True)
        return waveform
    except Exception as e:
        print(f"  ⚠ Error loading {Path(file_path).name}: {e}", file=sys.stderr)
        return None


def pad_or_truncate(waveform: np.ndarray, length: int = TARGET_LENGTH) -> np.ndarray:
    """Pad or truncate waveform to exactly `length` samples."""
    if len(waveform) < length:
        # Pad with zeros
        waveform = np.pad(waveform, (0, length - len(waveform)))
    elif len(waveform) > length:
        # Center crop
        start = (len(waveform) - length) // 2
        waveform = waveform[start : start + length]
    return waveform


def process_audio_tta(waveform: np.ndarray) -> List[np.ndarray]:
    """
    Test-Time Augmentation: split long audio into 3-second chunks.
    For audio <= 3s, returns a single chunk.
    For audio > 3s, returns overlapping chunks with 50% overlap.
    """
    if len(waveform) <= TARGET_LENGTH:
        return [pad_or_truncate(waveform)]

    chunks = []
    hop = TARGET_LENGTH // 2  # 50% overlap
    for start in range(0, len(waveform) - TARGET_LENGTH + 1, hop):
        chunk = waveform[start : start + TARGET_LENGTH]
        chunks.append(chunk)

    # If no chunk was generated (shouldn't happen), fallback
    if not chunks:
        chunks.append(pad_or_truncate(waveform))

    return chunks


# ─────────────────────────────────────────────────────────
#  Model Loading
# ─────────────────────────────────────────────────────────

def load_model(
    checkpoint_path: str,
    config_path: str,
    device: torch.device,
) -> TwoHeadedWavLM:
    """
    Load model from checkpoint with proper class mapping.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    class_map = checkpoint.get("class_map", None)

    if class_map is None:
        # Fallback: try to load from config
        raise ValueError("No class_map found in checkpoint.")

    num_known = len(class_map) - 1  # exclude unknown (class 0)

    # Load config
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(f"  Loaded class map: {len(class_map)} classes "
          f"(0=unknown, 1..{num_known}=known)")

    # Build model via factory (supports any encoder/head combo)
    model = create_model_from_config(config, num_known_speakers=num_known)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"  Model loaded from epoch {checkpoint.get('epoch', '?')}")
    print(f"  Best val loss: {checkpoint.get('val_loss', 'N/A'):.4f}")

    return model


# ─────────────────────────────────────────────────────────
#  Prediction
# ─────────────────────────────────────────────────────────

@torch.no_grad()
def predict_probs(
    model: TwoHeadedWavLM,
    waveform: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """
    Run inference with TTA and return averaged probability vector.

    Returns:
        probs: (447,) numpy array with sum ≈ 1.0
    """
    chunks = process_audio_tta(waveform)
    chunk_probs = []

    for chunk in chunks:
        # Convert to tensor: (1, 1, T)
        tensor = torch.from_numpy(chunk).float().unsqueeze(0).unsqueeze(0)
        tensor = tensor.to(device)

        # Get probabilities
        probs = model.predict_proba(tensor)  # (1, 447)
        chunk_probs.append(probs.cpu().numpy()[0])

    # Average over TTA chunks
    avg_probs = np.mean(chunk_probs, axis=0)

    # Ensure sum == 1.0 (numerical safety)
    avg_probs = avg_probs / avg_probs.sum()

    return avg_probs


# ─────────────────────────────────────────────────────────
#  Inference Runner
# ─────────────────────────────────────────────────────────

def run_inference(
    data_dir: str,
    checkpoint_path: str,
    config_path: str,
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    """
    Run inference on all audio files in data_dir.

    Returns:
        DataFrame with columns: id, 0, 1, ..., 446
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 55)
    print("  Inference Engine — Open-Set Speaker ID")
    print("=" * 55)
    print(f"  Device: {device}")
    print(f"  Data dir: {data_dir}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Config: {config_path}")
    print()

    # Load model
    print("  [1/3] Loading model...")
    try:
        model = load_model(checkpoint_path, config_path, device)
    except (FileNotFoundError, ValueError) as e:
        print(f"  ❌ {e}", file=sys.stderr)
        # Return uniform distribution as fallback
        return _fallback_predictions(data_dir)

    # Find audio files
    print("\n  [2/3] Finding audio files...")
    audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(Path(data_dir).glob(f"*{ext}"))
        audio_files.extend(Path(data_dir).glob(f"**/*{ext}"))

    audio_files = sorted(set(audio_files))  # deduplicate
    print(f"  Found {len(audio_files)} audio files")

    if len(audio_files) == 0:
        print("  ❌ No audio files found!", file=sys.stderr)
        return _fallback_predictions(data_dir)

    # Run inference
    print("\n  [3/3] Running inference...")
    results = []

    for audio_path in tqdm(audio_files, desc="  Predicting"):
        waveform = load_audio(str(audio_path))

        if waveform is None:
            # Fallback: uniform distribution (weighted towards unknown)
            probs = np.ones(NUM_CLASSES) / NUM_CLASSES
        else:
            probs = predict_probs(model, waveform, device)

        row = {"id": audio_path.name}
        for i in range(NUM_CLASSES):
            row[str(i)] = float(probs[i])
        results.append(row)

        # Verify sum
        assert abs(sum(row[str(i)] for i in range(NUM_CLASSES)) - 1.0) < 1e-5, \
            f"Probabilities don't sum to 1.0 for {audio_path.name}"

    df = pd.DataFrame(results)
    columns = ["id"] + [str(i) for i in range(NUM_CLASSES)]
    df = df[columns]  # enforce column order

    print(f"\n  ✅ Inference complete! {len(df)} predictions generated.")
    return df


def _fallback_predictions(data_dir: str) -> pd.DataFrame:
    """Generate fallback predictions (uniform distribution) when model fails to load."""
    print("  ⚠ Generating fallback predictions (uniform distribution)...")

    audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(Path(data_dir).glob(f"*{ext}"))
        audio_files.extend(Path(data_dir).glob(f"**/*{ext}"))
    audio_files = sorted(set(audio_files))

    results = []
    uniform_prob = 1.0 / NUM_CLASSES
    for audio_path in audio_files:
        row = {"id": audio_path.name}
        for i in range(NUM_CLASSES):
            row[str(i)] = uniform_prob
        results.append(row)

    df = pd.DataFrame(results)
    columns = ["id"] + [str(i) for i in range(NUM_CLASSES)]
    return df[columns]


# ─────────────────────────────────────────────────────────
#  CLI (Competition Format)
# ─────────────────────────────────────────────────────────

@click.command()
@click.option(
    "--data-dir",
    required=True,
    type=str,
    help="Directory containing input data files (test audio).",
)
@click.option(
    "--predictions-file-path",
    required=True,
    type=str,
    help="Path to write the output predictions CSV.",
)
def main(data_dir: str, predictions_file_path: str):
    """
    IAAA 2026 Speaker Identification — Inference & Submission.

    Generates a CSV with 447 probability columns (0=unknown, 1-446=known speakers).
    """
    # ── Resolve paths ──
    data_dir = os.path.abspath(data_dir)
    predictions_file_path = os.path.abspath(predictions_file_path)

    default_checkpoint = os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pt")
    default_config = os.path.join(PROJECT_ROOT, "configs", "default_config.yaml")

    checkpoint_path = default_checkpoint
    config_path = default_config

    # ── Check prerequisites ──
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        print("   Please train a model first: python -m src.train", file=sys.stderr)
        sys.exit(1)

    # ── Run inference ──
    df = run_inference(
        data_dir=data_dir,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
    )

    # ── Save predictions ──
    os.makedirs(os.path.dirname(predictions_file_path), exist_ok=True)
    df.to_csv(predictions_file_path, index=False)
    click.echo(f"\n✅ Saved {len(df)} predictions to {predictions_file_path}")
    click.echo(f"   Columns: {list(df.columns[:3])}...{list(df.columns[-3:])}")
    click.echo(f"   Shape: {df.shape}")


if __name__ == "__main__":
    main()
