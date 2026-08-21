"""
Competition entry point — IAAA 2026 Open-Set Speaker Identification.

The leaderboard runs:

    python submission.py --data-dir <test_folder> --predictions-file-path <csv>

The output CSV has columns ``audio_file,speaker_id``:
    - ``audio_file`` — the test audio file's full name (e.g. ``x.mp3``)
    - ``speaker_id`` — the predicted speaker UUID, or ``unknown`` for OOD

The model scores a 447-way probability distribution per file (0 = unknown,
1..446 = known speaker UUIDs in class-map order) and the argmax index is
mapped back to the speaker_id via the checkpoint's class map.

Everything this script needs ships inside the package folder:
    - src/            project code
    - weights/        pretrained encoder weights (offline, allow_hub_download=false)
    - checkpoints/    trained <encoder>_best.pt models
    - vendor/         vendored pure-Python deps missing on the leaderboard
    - ensemble_fusion_weights.json  best fusion config (weighted_average)
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import click
import numpy as np

# ── Self-contained offline environment (set BEFORE any encoder import) ──
PKG_DIR = Path(__file__).resolve().parent
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("MODELSCOPE_CACHE", str(PKG_DIR / "weights" / "campp"))
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

# Silence third-party chatter (speechbrain/transformers/nemo deprecation
# warnings + modelscope/nemo/torch logging) — the leaderboard only surfaces
# the first few log lines, so the submission itself must stay silent and a
# server error must be the first thing shown.
import logging
import warnings
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

# Windows cp1252 fix: force UTF-8 stdio so emoji output never crashes.
from src.cli_utils import setup_utf8_stdio
setup_utf8_stdio()

import torch

# ── GPU diagnostic — the ABSOLUTE FIRST line of output ──
# This is the single most useful signal for diagnosing an EXECUTION TIMEOUT:
# it tells us whether the eval environment exposed a CUDA GPU to this process.
# It prints before any heavy import (transformers/modelscope/nemo are all lazy
# inside the encoders) and before model loading, so it survives even if the run
# later times out. Keep it terse and on one line.
def _print_gpu_diagnostics() -> None:
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    gpu_name = ""
    if device_count > 0:
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = "?"
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    torch_cuda_ver = getattr(torch.version, "cuda", "None")
    print(
        f"[diag] cuda_avail={cuda_available} devices={device_count} "
        f"gpu=\"{gpu_name}\" CUDA_VISIBLE_DEVICES={cuda_visible} "
        f"torch.cuda={torch_cuda_ver}",
        flush=True,
    )

_print_gpu_diagnostics()

# Import the shared inference core. `inference.py` sits next to this file both
# in the repo (submission/) and at the zip root (when the package is shipped).
try:
    from inference import score_ensemble, load_centroids  # zip-root layout (leaderboard)
except ImportError:                       # repo layout (python -m submission.submission)
    from submission.inference import score_ensemble, load_centroids

DEFAULT_FUSION_WEIGHTS = PKG_DIR / "ensemble_fusion_weights.json"
SUPPORTED_AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

# Last files / class map from load_data + predict — used by save_predictions.
_LAST_FILES: List[Path] = []
_LAST_CLASS_MAP: Optional[dict] = None  # label -> index (0 = unknown)


# ────────────────────────────────────────────────────────────────
#  Contract functions (submissionforleaderbord.txt)
# ────────────────────────────────────────────────────────────────

def load_data(data_dir: str) -> List[Path]:
    """Return the sorted list of audio files in ``data_dir``."""
    global _LAST_FILES
    files = sorted(
        p for p in Path(data_dir).iterdir()
        if p.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES
    )
    if not files:
        raise click.ClickException(
            f"No audio files ({sorted(SUPPORTED_AUDIO_SUFFIXES)}) found in {data_dir}"
        )
    _LAST_FILES = files
    return files


def _discover_checkpoints() -> List[str]:
    """Auto-discover ``checkpoints/*_best.pt`` in the fusion-config order."""
    ckpt_dir = PKG_DIR / "checkpoints"
    if not ckpt_dir.exists():
        raise click.ClickException(
            f"checkpoints/ not found next to {__file__}. "
            "The submission package must ship its trained checkpoints."
        )
    fw_path = PKG_DIR / "ensemble_fusion_weights.json"
    order = None
    if fw_path.exists():
        try:
            order = json.loads(fw_path.read_text(encoding="utf-8")).get("encoder_names")
        except Exception:
            order = None

    checkpoints = []
    if order:
        for enc in order:
            c = ckpt_dir / f"{enc}_best.pt"
            if c.exists():
                checkpoints.append(str(c))
    if not checkpoints:  # fallback: any *_best.pt, alphabetically
        checkpoints = sorted(str(p) for p in ckpt_dir.glob("*_best.pt"))
    if not checkpoints:
        raise click.ClickException(
            "No *_best.pt checkpoints found in checkpoints/ — nothing to run."
        )
    return checkpoints


def _encoder_name(ckpt_path: str) -> str:
    """'checkpoints/campp_best.pt' → 'campp'."""
    return Path(ckpt_path).name.replace("_best.pt", "")


def _load_centroids(checkpoint_paths: List[str]) -> Optional[dict]:
    """Load ``centroids/<enc>.npz`` for the checkpoints being run.

    Returns None when no centroids are shipped (plain argmax fallback).
    """
    cdir = PKG_DIR / "centroids"
    if not cdir.exists():
        return None
    encoders = [_encoder_name(c) for c in checkpoint_paths]
    # Cluster mode: read the first checkpoint's embedded config so the k-locked
    # centroids_unknown_<enc>_k<k>.npz is merged — the model's own k — not a
    # different-k experiment's centroids that happens to be in the package.
    num_unknown_clusters = 0
    try:
        ck = torch.load(checkpoint_paths[0], map_location="cpu", weights_only=False)
        num_unknown_clusters = int(
            (ck.get("config", {}).get("model", {}) or {})
            .get("num_unknown_clusters", 0) or 0)
    except Exception:
        pass
    centroids = load_centroids(str(cdir), encoders,
                               num_unknown_clusters=num_unknown_clusters)
    return centroids or None


def _load_decision_params() -> Optional[dict]:
    """Load tuned decision-layer params from ``decision_config.json``.

    Returns None when the file is absent (plain argmax fallback).
    """
    path = PKG_DIR / "decision_config.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("decision_params", data)


def predict(data_dir: str) -> np.ndarray:
    """Score every file in ``data_dir`` and return the predicted class indices.

    Returns a 1-D integer array of argmax labels (0..446), one per file.
    ``save_predictions`` maps these indices to ``speaker_id`` values via the
    checkpoint class map (0 → ``unknown``, 1..446 → known speaker UUIDs).

    Uses the best fusion config in ``ensemble_fusion_weights.json``
    (weighted_average with optimised per-encoder weights) and, when shipped,
    the centroid + OOD-gate decision layer in ``decision_config.json``.
    """
    global _LAST_FILES, _LAST_CLASS_MAP
    files = load_data(data_dir)

    checkpoint_path = _discover_checkpoints()
    fusion_weights = json.loads(
        DEFAULT_FUSION_WEIGHTS.read_text(encoding="utf-8")
    )["weights"]

    result = score_ensemble(
        data_dir=data_dir,
        checkpoint_path=checkpoint_path,
        fusion_method="weighted_average",
        fusion_weights=fusion_weights,
        centroids=_load_centroids(checkpoint_path),
        decision_params=_load_decision_params(),
    )

    _LAST_FILES = result["files"]
    _LAST_CLASS_MAP = result["class_map"]
    return result["labels"].astype(np.int64)


def save_predictions(predictions: np.ndarray, output_path: str) -> None:
    """Write the competition CSV: columns ``audio_file,speaker_id``.

    Each row maps a test audio file (full name, e.g. ``x.mp3``) to a
    ``speaker_id``: the known speaker's UUID, or ``unknown`` for OOD. The
    integer class indices returned by ``predict`` are mapped through the
    checkpoint class map (0 = unknown, 1..446 = known speaker UUIDs).
    """
    predictions = np.asarray(predictions)
    if predictions.ndim != 1:
        raise click.ClickException(
            f"predictions must be 1-D (predicted class per file), "
            f"got shape {predictions.shape}"
        )
    if len(_LAST_FILES) != len(predictions):
        raise click.ClickException(
            f"predictions length ({len(predictions)}) != files count "
            f"({len(_LAST_FILES)}). Call load_data(data_dir) / predict(data_dir) first."
        )
    if not _LAST_CLASS_MAP:
        raise click.ClickException(
            "No class map available — call predict(data_dir) before "
            "save_predictions()."
        )

    # index -> label ("unknown" or speaker UUID)
    inverse = {int(idx): label for label, idx in _LAST_CLASS_MAP.items()}

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["audio_file", "speaker_id"])
        for fpath, pred in zip(_LAST_FILES, predictions):
            speaker = inverse.get(int(pred), "unknown")
            # Closed-set cluster experiment: internal pseudo ids (unknown_<n>)
            # are the aggregated "unknown" from the competition's point of view.
            if isinstance(speaker, str) and speaker.startswith("unknown_"):
                speaker = "unknown"
            writer.writerow([fpath.name, speaker])


# ────────────────────────────────────────────────────────────────
#  CLI (the leaderboard runs this)
# ────────────────────────────────────────────────────────────────

@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--data-dir", required=True, type=click.Path(exists=True, file_okay=False),
              help="Directory containing input data files.")
@click.option("--predictions-file-path", required=True, type=click.Path(),
              help="Path to write the output predictions CSV.")
def main(data_dir: str, predictions_file_path: str) -> None:
    """Run the competition entry point and write the submission CSV."""
    load_data(data_dir)
    predictions = predict(data_dir)
    save_predictions(predictions, predictions_file_path)


if __name__ == "__main__":
    main()
