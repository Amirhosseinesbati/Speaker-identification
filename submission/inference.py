"""
Submission inference CLI for Open-Set Speaker Identification (IAAA 2026).

Usage:
    uv run --no-sync python -m submission.inference \
        --data-dir <test_audio_dir> \
        --predictions-file-path predictions.csv

Produces a CSV with columns `id,0,1,...,446` — one row per test file, 447
class probabilities summing to 1. Column order follows the class-map
convention: column 0 = unknown, columns 1..446 = known speaker UUIDs in
lexicographic order (same as `src.data_pipeline.create_class_mapping`).

Features (Phase 5 optimizations):
  - Each checkpoint loads with the config embedded in it (``ckpt["config"]``),
    so the 5-encoder ensemble needs no shared config file.
  - Sequential ensemble: one model in GPU memory at a time — load → score all
    files → free → next (per-model + total wall-time logged).
  - ``torch.inference_mode()`` + fp16 autocast + ``cudnn.benchmark``.
  - Files sorted by duration for better kernel reuse.
  - Multi-window TTA: 8 s windows, 50% overlap, per-window probabilities
    averaged and renormalised.
  - Safe fallback: undecodable files get a uniform 1/447 row so the CSV is
    always well-formed.

Environment for the offline simulation:
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    MODELSCOPE_CACHE=<submission>/weights/campp   (pointed at the local cache)
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import click
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Windows cp1252 fix: force UTF-8 stdio so emoji output never crashes.
from src.cli_utils import setup_utf8_stdio
setup_utf8_stdio()


# ────────────────────────────────────────────────────────────────
#  Model loading — each checkpoint carries its OWN config
# ────────────────────────────────────────────────────────────────

def load_model(
    checkpoint_path: str,
    device: torch.device,
    fallback_config_path: Optional[str] = None,
) -> Tuple[torch.nn.Module, dict, Dict[str, int], Optional[float]]:
    """Build the model from the config embedded in the checkpoint.

    Returns:
        model, config, class_map, ood_threshold
    """
    from src.model_factory import create_model_from_config

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    config = checkpoint.get("config")
    if config is None:
        if fallback_config_path is None:
            raise click.ClickException(
                f"Checkpoint {checkpoint_path} has no embedded config — pass "
                "--config-path as the fallback."
            )
        from src.data_pipeline import load_config
        config = load_config(fallback_config_path)
        print(f"  ⚠ {checkpoint_path}: no embedded config — used {fallback_config_path}")

    class_map = checkpoint.get("class_map")
    if class_map is None:
        labels_path = config["data"]["labels_path"]
        df = __import__("pandas").read_csv(labels_path)
        df.columns = df.columns.str.strip()
        from src.data_pipeline import create_class_mapping
        class_map = create_class_mapping(df)
        print(f"  ⚠ class_map not in checkpoint — rebuilt from {labels_path}")

    num_known = config.get("model", {}).get(
        "competition_num_known", len(class_map) - 1,
    )
    model = create_model_from_config(config, num_known_speakers=num_known)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    ood_threshold = checkpoint.get("ood_threshold")
    return model, config, class_map, ood_threshold


# ────────────────────────────────────────────────────────────────
#  Audio loading + multi-window TTA (mirrors SpeakerDataset._eval_windows)
# ────────────────────────────────────────────────────────────────

def _load_waveform(audio_path: Path, sample_rate: int) -> Optional[torch.Tensor]:
    """Load a file to (1, N) float32 at `sample_rate`; None on decode error."""
    import librosa
    import soundfile as sf

    try:
        if audio_path.suffix.lower() == ".wav":
            wav, sr = sf.read(str(audio_path), dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
        else:
            # libsndfile first — the same decoder used by MP3→WAV conversion
            # (src/audio_preprocessing.py); librosa as fallback for formats
            # libsndfile cannot read (e.g. M4A/AAC).
            try:
                wav, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
            except Exception:
                wav, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != sample_rate:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=sample_rate)
        return torch.from_numpy(np.asarray(wav, dtype=np.float32)).unsqueeze(0)
    except Exception:
        return None


def make_windows(
    waveform: torch.Tensor,
    sample_rate: int,
    duration_seconds: float,
    eval_hop_ratio: float,
    max_eval_windows: int,
) -> List[torch.Tensor]:
    """Sliding windows over the full file (same logic as SpeakerDataset)."""
    T = int(sample_rate * duration_seconds)
    n = waveform.size(-1)
    if n <= T:
        w = torch.nn.functional.pad(waveform, (0, T - n))
        return [w] * max_eval_windows

    hop = max(1, int(T * eval_hop_ratio))
    starts = list(range(0, n - T + 1, hop))
    if len(starts) > max_eval_windows:
        starts = np.unique(np.linspace(0, n - T, max_eval_windows).astype(int)).tolist()
    windows = [waveform[..., s : s + T] for s in starts]
    while len(windows) < max_eval_windows:
        windows.append(windows[-1])
    return windows


@torch.inference_mode()
def predict_file_probs(
    model: torch.nn.Module,
    audio_path: Path,
    device: torch.device,
    sample_rate: int = 16000,
    duration_seconds: float = 8.0,
    eval_hop_ratio: float = 0.5,
    max_eval_windows: int = 8,
    use_amp: bool = True,
) -> Optional[np.ndarray]:
    """Multi-window TTA: average per-window 447-way probabilities.

    Returns a (num_classes,) probability vector or None if the file cannot be
    decoded (caller should fall back to a uniform row).
    """
    waveform = _load_waveform(audio_path, sample_rate)
    if waveform is None:
        return None

    windows = make_windows(waveform, sample_rate, duration_seconds,
                           eval_hop_ratio, max_eval_windows)

    probs_sum = None
    autocast_ctx = (
        torch.autocast(device_type="cuda", enabled=use_amp and device.type == "cuda")
    )
    with autocast_ctx:
        for w in windows:
            p = model.predict_proba(w.unsqueeze(0).to(device)).float().cpu().numpy()[0]
            probs_sum = p if probs_sum is None else probs_sum + p

    probs = probs_sum / len(windows)
    probs = probs / probs.sum()
    return probs


@torch.inference_mode()
def compute_file_embedding(
    model: torch.nn.Module,
    audio_path: Path,
    device: torch.device,
    sample_rate: int = 16000,
    duration_seconds: float = 8.0,
    eval_hop_ratio: float = 0.5,
    max_eval_windows: int = 8,
) -> Optional[np.ndarray]:
    """Multi-window mean encoder embedding (for centroid fusion)."""
    waveform = _load_waveform(audio_path, sample_rate)
    if waveform is None:
        return None
    windows = make_windows(waveform, sample_rate, duration_seconds,
                           eval_hop_ratio, max_eval_windows)
    embs = []
    for w in windows:
        hidden, _ = model.encoder(w.unsqueeze(0).to(device))
        embs.append(hidden.squeeze(1).float().cpu().numpy()[0])
    return np.mean(embs, axis=0)


# ────────────────────────────────────────────────────────────────
#  Centroid fusion (step-6 embedding cache)
# ────────────────────────────────────────────────────────────────

class CentroidFuser:
    """Fuses model probabilities with the centroid classifier output."""

    def __init__(self, alpha: float = 0.5):
        from src.centroid_baseline import (
            _cache_paths, fit_centroids, centroid_probs_global,
        )

        paths = _cache_paths()
        if not all(p.exists() for p in paths.values()):
            raise FileNotFoundError(
                "Embedding cache missing — run `uv run --no-sync python -m "
                "src.centroid_baseline` first (or drop --fuse-centroid)."
            )
        train_embs = np.load(paths["train_embs"])
        train_labels = np.load(paths["train_labels"])
        self.centroids, self.speakers = fit_centroids(train_embs, train_labels)
        self.alpha = alpha
        print(f"  🎯 Centroid fuser ready: {len(self.speakers)} centroids, "
              f"alpha_model={alpha}")

    def probs(self, embedding: np.ndarray, num_classes: int) -> np.ndarray:
        """Embedding (D,) → (num_classes,) centroid probability vector."""
        sims = embedding.reshape(1, -1)
        sims = sims / (np.linalg.norm(sims, axis=1, keepdims=True) + 1e-12)
        sims = sims @ self.centroids.T
        ood_scores = 1.0 - sims.max(axis=1)
        return centroid_probs_global(
            sims, ood_scores, self.speakers, num_classes=num_classes,
        )[0]

    def fuse(self, model_probs: np.ndarray, embedding: np.ndarray) -> np.ndarray:
        cprobs = self.probs(embedding, len(model_probs))
        fused = self.alpha * model_probs + (1.0 - self.alpha) * cprobs
        fused = fused / fused.sum()
        return fused


# ────────────────────────────────────────────────────────────────
#  CSV output
# ────────────────────────────────────────────────────────────────

def write_predictions_csv(
    rows: List[Tuple[str, np.ndarray]],
    output_path: str,
    class_map: Dict[str, int],
) -> None:
    """Write id,0..446 probability rows (sum per row = 1).

    Also writes a sidecar `<output>.class_map.json` mapping column index →
    speaker_id so the column order is fully reproducible.
    """
    num_classes = len(class_map)
    columns = ["id"] + [str(i) for i in range(num_classes)]

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for fid, probs in rows:
            row = [fid] + [f"{float(p):.8f}" for p in probs]
            writer.writerow(row)

    # Column index → speaker_id sidecar
    inv = {int(v): k for k, v in class_map.items()}
    mapping = [inv[i] for i in range(num_classes)]
    sidecar = out.with_suffix(".class_map.json")
    sidecar.write_text(json.dumps(mapping, indent=2), encoding="utf-8")

    print(f"  ✓ Predictions CSV: {out} ({len(rows)} rows, {num_classes + 1} columns)")
    print(f"  ✓ Column mapping:  {sidecar}")


# ────────────────────────────────────────────────────────────────
#  CLI
# ────────────────────────────────────────────────────────────────

@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--data-dir", required=True, type=click.Path(exists=True, file_okay=False),
              help="Folder containing the test audio files.")
@click.option("--predictions-file-path", required=True, type=click.Path(),
              help="Output CSV path (id,0..446).")
@click.option("--config-path", default=None, type=click.Path(exists=True),
              help="Fallback config for checkpoints that lack an embedded one.")
@click.option("--checkpoint-path", multiple=True,
              default=("checkpoints/best_model.pt",), show_default=True,
              type=click.Path(exists=True),
              help="Checkpoint(s) to load. Each uses its embedded config. "
                   "Pass several for the sequential ensemble.")
@click.option("--id-style", type=click.Choice(["stem", "filename"]), default="stem",
              show_default=True, help="'stem' = file name without extension (default).")
@click.option("--apply-ood-threshold", is_flag=True, default=False,
              help="Hard-gate P(unknown) > saved threshold → class 0 (default OFF: "
                   "the competition scores plain argmax over the 447-way output).")
@click.option("--fuse-centroid", is_flag=True, default=False,
              help="Fuse model probs with the step-6 centroid classifier "
                   "(requires the embedding cache).")
@click.option("--centroid-alpha", default=0.5, type=float, show_default=True,
              help="Weight of the model probs in the fusion (1-alpha for centroid).")
@click.option("--max-eval-windows", default=None, type=int,
              help="Override max_eval_windows from config.")
@click.option("--no-amp", is_flag=True, default=False,
              help="Disable fp16 autocast (default: enabled on CUDA).")
def main(
    data_dir: str,
    predictions_file_path: str,
    config_path: Optional[str],
    checkpoint_path: Tuple[str, ...],
    id_style: str,
    apply_ood_threshold: bool,
    fuse_centroid: bool,
    centroid_alpha: float,
    max_eval_windows: Optional[int],
    no_amp: bool,
) -> None:
    """Run inference on a folder of test audio and write the submission CSV."""
    print("=" * 60)
    print("  Submission Inference (sequential ensemble, offline)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"  cudnn.benchmark: True | fp16 autocast: {not no_amp}")

    # ── Load every checkpoint with its own embedded config ──
    t_load = time.time()
    models = []
    for ckpt in checkpoint_path:
        m, cfg, cm, thr = load_model(ckpt, device, fallback_config_path=config_path)
        models.append((m, cm, thr))
        print(f"  ✓ Loaded {ckpt} [{m.encoder_name}] ({time.time()-t_load:.1f}s cumulative)")
    if not models:
        raise click.ClickException("No checkpoints to run.")
    class_map = models[0][1]
    num_classes = len(class_map)

    # TTA params come from the first checkpoint's embedded config.
    first_ckpt = torch.load(checkpoint_path[0], map_location="cpu", weights_only=False)
    audio_cfg = first_ckpt.get("config", {}).get("audio", {})
    sample_rate = audio_cfg.get("sample_rate", 16000)
    duration_seconds = audio_cfg.get("duration_seconds", 8.0)
    eval_hop_ratio = audio_cfg.get("eval_hop_ratio", 0.5)
    max_windows = max_eval_windows or audio_cfg.get("max_eval_windows", 8)

    ood_thresholds = [thr for _, _, thr in models]
    if apply_ood_threshold:
        print(f"  🎯 OOD thresholds from checkpoints: {ood_thresholds}")

    fuser = None
    if fuse_centroid:
        try:
            fuser = CentroidFuser(alpha=centroid_alpha)
        except FileNotFoundError as e:
            print(f"  ⚠ {e}")

    files = sorted(p for p in Path(data_dir).iterdir()
                   if p.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg", ".m4a"})
    if not files:
        raise click.ClickException(f"No audio files found in {data_dir}")
    # Sort by duration so consecutive files reuse warm kernels (long→short is
    # fine for per-file scoring).
    files = sorted(files, key=lambda p: p.stat().st_size, reverse=True)
    print(f"  Files to score: {len(files):,} | Ensemble size: {len(models)}")

    uniform = np.full(num_classes, 1.0 / num_classes)
    acc = np.zeros((len(files), num_classes))
    scored = np.zeros(len(files), dtype=bool)

    # ── Sequential ensemble: one model in GPU at a time ──
    t_total_start = time.time()
    first_model = None  # kept for centroid-fusion embeddings
    for idx, (model, cm, thr) in enumerate(models):
        t_m = time.time()
        n_ok = 0
        for i, f in enumerate(tqdm(files, desc=f"  Model {idx+1}/{len(models)}")):
            p = predict_file_probs(
                model, f, device, sample_rate=sample_rate,
                duration_seconds=duration_seconds, eval_hop_ratio=eval_hop_ratio,
                max_eval_windows=max_windows, use_amp=not no_amp,
            )
            if p is None:
                continue
            acc[i] += p
            scored[i] = True
            n_ok += 1
        dt = time.time() - t_m
        peak = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0
        print(f"  ⏱ Model {idx+1} ({model.encoder_name}): {dt:.1f}s "
              f"({dt/max(n_ok,1)*1000:.0f} ms/file, {n_ok} scored, peak {peak:.2f} GB)")
        if idx == 0 and fuse_centroid:
            first_model = model  # keep for embedding extraction
        else:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    total_elapsed = time.time() - t_total_start
    print(f"  ⏱ TOTAL ensemble wall-time: {total_elapsed:.1f}s "
          f"({total_elapsed/max(len(files),1)*1000:.0f} ms/file)")

    # ── Average the per-model probabilities ──
    rows: List[Tuple[str, np.ndarray]] = []
    for i, f in enumerate(files):
        fid = f.stem if id_style == "stem" else f.name
        if not scored[i]:
            print(f"  ⚠ Fallback (uniform 1/{num_classes}) for {f.name} — decode error")
            probs = uniform.copy()
        else:
            probs = acc[i] / len(models)
            probs = probs / probs.sum()

            if fuser is not None and first_model is not None:
                emb = compute_file_embedding(
                    first_model, f, device, sample_rate=sample_rate,
                    duration_seconds=duration_seconds,
                    eval_hop_ratio=eval_hop_ratio,
                    max_eval_windows=max_windows,
                )
                if emb is not None:
                    probs = fuser.fuse(probs, emb)

            if apply_ood_threshold and ood_thresholds[0] is not None and probs[0] > ood_thresholds[0]:
                probs = np.zeros(num_classes)
                probs[0] = 1.0

        rows.append((fid, probs))

    write_predictions_csv(rows, predictions_file_path, class_map)
    print(f"\n✅ Inference complete. Total: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
