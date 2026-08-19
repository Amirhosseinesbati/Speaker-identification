"""
Minimal inference core for the IAAA 2026 speaker-ID submission.

The leaderboard only runs ``python submission.py --data-dir <dir>
--predictions-file-path <csv>``; this module backs that single flow:

    score_ensemble(...) → dict(files, probs, class_map, scored, total_elapsed)

Everything needed (weights/, checkpoints/, vendor/, class maps) ships inside
the package folder, fully offline (no hub downloads, no extra config).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

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

VALID_FUSION_METHODS = [
    "average", "weighted_average", "geometric_mean", "rank_average", "max_prob",
]


# ────────────────────────────────────────────────────────────────
#  Model loading — each checkpoint carries its OWN config + class map
# ────────────────────────────────────────────────────────────────

def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[torch.nn.Module, dict, Dict[str, int]]:
    """Build the model from the config embedded in the checkpoint.

    Returns:
        model, config, class_map
    """
    from src.model_factory import create_model_from_config

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    config = checkpoint["config"]
    class_map = checkpoint["class_map"]

    # Head width = non-unknown classes in the checkpoint's class map (446
    # legacy; 1000 when the closed-set cluster experiment is enabled — the
    # model collapses the cluster columns internally, so the 447-way output
    # contract is unchanged).
    num_known = len(class_map) - 1
    model = create_model_from_config(config, num_known_speakers=num_known)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, config, class_map


# ────────────────────────────────────────────────────────────────
#  Audio loading + multi-window TTA
# ────────────────────────────────────────────────────────────────

def _load_waveform(audio_path: Path, sample_rate: int) -> Optional[torch.Tensor]:
    """Load a file to (1, N) float32 at `sample_rate`; None on decode error."""
    import librosa
    import soundfile as sf

    audio_path = Path(audio_path)
    try:
        if audio_path.suffix.lower() == ".wav":
            wav, sr = sf.read(str(audio_path), dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
        else:
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
def predict_file_probs_and_embedding(
    model: torch.nn.Module,
    waveform: torch.Tensor,
    device: torch.device,
    sample_rate: int = 16000,
    duration_seconds: float = 8.0,
    eval_hop_ratio: float = 0.5,
    max_eval_windows: int = 8,
    use_amp: bool = True,
    temperature: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Multi-window TTA on an ALREADY-DECODED waveform → (probs, embedding).

    Decoding is intentionally NOT done here — ``score_ensemble`` decodes each
    file exactly once and passes the waveform so it is reused across every
    model instead of being re-decoded per model (MP3 decode is ~30 ms/file;
    re-decoding 4× wastes ~6 min on the full test set).

    All ``max_eval_windows`` windows are stacked into a (W, 1, T) batch and
    sent to GPU in ONE transfer. The model processes the full batch in one
    forward pass. The returned embedding is the window-averaged (then
    L2-normalised) ArcFace speaker embedding — the centroid cosine-decision
    space, matching ``src.centroid_baseline.build_checkpoint_centroids``.

    Returns:
        probs: (num_classes,) fused head probability vector (rows sum to 1).
        emb:   (embedding_dim,) unit-norm speaker embedding.
    """
    windows = make_windows(waveform, sample_rate, duration_seconds,
                           eval_hop_ratio, max_eval_windows)

    # Stack all windows into (W, 1, T) — one transfer, one forward pass
    batch = torch.stack(windows).to(device)  # (W, 1, T)

    autocast_ctx = (
        torch.autocast(device_type="cuda", enabled=use_amp and device.type == "cuda")
    )
    with autocast_ctx:
        # Single encoder forward returns BOTH probs and embedding (no second pass).
        probs, emb = model.predict_proba_and_embed(batch, temperature=temperature)

    probs = probs.float().cpu().numpy()  # (num_classes,)
    probs = probs / probs.sum()
    emb = emb.float().cpu().numpy()      # (embedding_dim,)
    return probs, emb


# ────────────────────────────────────────────────────────────────
#  Centroid + OOD-gate decision layer (Q4)
# ────────────────────────────────────────────────────────────────

def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax (no scipy dependency)."""
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=axis, keepdims=True) + 1e-12)


def load_centroids(
    centroids_dir: str,
    encoder_names: Sequence[str],
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Load ``centroids_<enc>.npz`` for the given encoders.

    When a ``centroids_unknown_<enc>.npz`` (554 pseudo-identity cluster
    centroids from ``src/unknown_clustering.py``) sits next to it, the two are
    merged into a 1000-way centroid matrix with speaker ids 1..1000 — the
    ``centroid_probs_matrix`` caller then collapses the cluster columns back
    into unknown.

    Returns:
        {encoder: (centroids (S, D), speaker_ids (S,))}
    """
    centroids: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    cdir = Path(centroids_dir)
    for enc in encoder_names:
        path = cdir / f"centroids_{enc}.npz"
        if not path.exists():
            continue
        data = np.load(path)
        cents = data["centroids"].astype(np.float32)
        sids = data["speaker_ids"].astype(np.int64)

        cluster_path = cdir / f"centroids_unknown_{enc}.npz"
        if cluster_path.exists():
            cdata = np.load(cluster_path)
            cluster_cents = cdata["centroids"].astype(np.float32)
            num_known = int(cents.shape[0])
            k = int(cluster_cents.shape[0])
            cluster_ids = np.arange(num_known + 1, num_known + 1 + k,
                                    dtype=np.int64)
            cents = np.vstack([cents, cluster_cents])
            sids = np.concatenate([sids, cluster_ids])
            print(f"[diag] {enc}: merged {k} unknown-cluster centroids "
                  f"({cents.shape[0]} total)")

        centroids[enc] = (cents, sids)
    return centroids


def centroid_probs_matrix(
    embs: np.ndarray,
    centroids: np.ndarray,
    speaker_ids: np.ndarray,
    num_classes: int,
    kappa: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cosine centroid probabilities for a matrix of embeddings.

    Args:
        embs: (N, D) unit-norm speaker embeddings.
        centroids: (S, D) unit-norm centroids (row j = speaker_ids[j]).
        speaker_ids: (S,) global class ids (1..num_known).
        kappa: centroid softmax scale (higher = sharper known distribution).

    Returns:
        probs: (N, num_classes) — p[0] = 1 − max_cosine (unknown), known masses
               distributed by softmax(κ·cosine).
        max_cosine: (N,) — max cosine similarity to any centroid.
    """
    cos = embs @ centroids.T  # (N, S)
    max_cosine = cos.max(axis=1)  # (N,)
    known = _softmax(kappa * cos, axis=1)  # (N, S)
    p_unknown = np.clip(1.0 - max_cosine, 0.0, 1.0)  # (N,)

    probs = np.zeros((embs.shape[0], num_classes), dtype=np.float64)
    probs[:, 0] = p_unknown
    for j, sid in enumerate(speaker_ids):
        probs[:, int(sid)] = (1.0 - p_unknown) * known[:, j]
    probs /= (probs.sum(axis=1, keepdims=True) + 1e-12)
    return probs, max_cosine


def _collapse_centroid_probs(probs: np.ndarray, num_classes: int) -> np.ndarray:
    """(N, 1 + known + clusters) → (N, num_classes) for the 1000-class mode.

    The pseudo-identity cluster columns are summed into column 0 (unknown),
    exactly mirroring the trained model's ``predict_proba`` collapse, so the
    centroid decision layer stays in the fixed 447-way output space.
    """
    out = np.zeros((probs.shape[0], num_classes), dtype=probs.dtype)
    out[:, 0] = probs[:, 0] + probs[:, num_classes:].sum(axis=1)
    out[:, 1:] = probs[:, 1:num_classes]
    return out / (out.sum(axis=1, keepdims=True) + 1e-12)


# ────────────────────────────────────────────────────────────────
#  Ensemble scoring
# ────────────────────────────────────────────────────────────────

def score_ensemble(
    data_dir: str,
    checkpoint_path: Sequence[str],
    fusion_method: str = "weighted_average",
    fusion_weights: Optional[Sequence[float]] = None,
    max_eval_windows: Optional[int] = None,
    no_amp: bool = False,
    centroids: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
    decision_params: Optional[dict] = None,
) -> dict:
    """Score every file in ``data_dir`` and fuse the ensemble probabilities.

    If ``centroids`` and ``decision_params`` are both provided, the fused head
    probabilities are additionally fused with per-encoder cosine-centroid
    probabilities and passed through the OOD gate:

        fused = alpha * head_ens + (1 - alpha) * centroid_ens
        fused[0] *= lambda_unknown
        pred = argmax(fused); forced to 0 (unknown) when max_cosine < tau

    ``decision_params`` keys (all optional): ``alpha``, ``kappa``, ``tau``,
    ``lambda_unknown``, ``temperature``.

    Returns a dict:
      files         — list of scored Paths (in file order)
      probs         — (N, num_classes) fused probability matrix (post decision)
      labels        — (N,) final predicted class ids (post OOD gate)
      max_cosine    — (N,) per-file ensemble max centroid cosine (None if no
                      decision layer)
      class_map     — checkpoint class map (label -> index)
      scored        — (N,) bool array (True = decoded successfully)
      total_elapsed — ensemble wall time (seconds)

    Intentionally silent: the leaderboard only returns the first few lines of
    the job log, so we emit nothing ourselves — a server-side error must be
    the first thing shown.
    """
    # ── GPU diagnostic is printed by submission.py at import time ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # ── Filter out zero-weight models (their output is multiplied by 0) ──
    if fusion_method == "weighted_average" and fusion_weights is not None:
        epsilon = 1e-8
        active = [i for i, w in enumerate(fusion_weights) if w > epsilon]
        if len(active) < len(fusion_weights):
            skipped = len(fusion_weights) - len(active)
            print(f"[diag] skipped {skipped} zero-weight model(s) — "
                  f"{len(active)} remaining")
            checkpoint_path = [checkpoint_path[i] for i in active]
            fusion_weights = [fusion_weights[i] for i in active]
            if not checkpoint_path:
                raise RuntimeError(
                    "All fusion weights are zero — nothing to run. "
                    "Check ensemble_fusion_weights.json."
                )

    # ── Load every checkpoint with its own embedded config ──
    models = []
    for ckpt in checkpoint_path:
        m, cfg, cm = load_model(ckpt, device)
        models.append((m, cm))
    if not models:
        raise RuntimeError("No checkpoints to run.")
    class_map = models[0][1]
    # Competition output width: 447 legacy; with the closed-set cluster
    # experiment the internal class map is wider (1001) but the model's
    # predict_proba_and_embed already collapses the cluster columns into
    # unknown, so every per-model prob vector is 447-wide.
    num_unknown_clusters = int(models[0][0].num_unknown_clusters)
    # Competition output width — always the fixed 447 regardless of the
    # internal head width (cluster columns are collapsed by the model).
    num_classes = int(models[0][0].num_output_classes)
    n_models = len(models)

    encoder_names = [Path(c).name.replace("_best.pt", "") for c in checkpoint_path]

    # ── Decision-layer setup ──
    do_decision = centroids is not None and decision_params is not None
    decision_params = decision_params or {}
    alpha = float(decision_params.get("alpha", 0.5))
    kappa = float(decision_params.get("kappa", 8.0))
    tau = float(decision_params.get("tau", 0.0))
    lambda_unknown = float(decision_params.get("lambda_unknown", 1.0))
    temperature = float(decision_params.get("temperature", 1.0))

    # TTA params come from the first checkpoint's embedded config.
    first_ckpt = torch.load(checkpoint_path[0], map_location="cpu", weights_only=False)
    audio_cfg = first_ckpt.get("config", {}).get("audio", {})
    sample_rate = audio_cfg.get("sample_rate", 16000)
    duration_seconds = audio_cfg.get("duration_seconds", 8.0)
    eval_hop_ratio = audio_cfg.get("eval_hop_ratio", 0.5)
    max_windows = max_eval_windows or audio_cfg.get("max_eval_windows", 8)

    files = sorted(p for p in Path(data_dir).iterdir()
                   if p.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg", ".m4a"})
    if not files:
        raise RuntimeError(f"No audio files found in {data_dir}")
    files = sorted(files, key=lambda p: p.stat().st_size, reverse=True)
    n_files = len(files)

    uniform = np.full(num_classes, 1.0 / num_classes)

    # Per-model probability arrays: (n_models, n_files, num_classes)
    all_model_probs = [np.zeros((n_files, num_classes), dtype=np.float64)
                       for _ in models]
    scored = np.zeros(n_files, dtype=bool)

    # Per-model embeddings only for models that have a centroid (decision path).
    all_model_embs: Dict[int, np.ndarray] = {}
    if do_decision:
        for mi, enc in enumerate(encoder_names):
            if enc in centroids:
                dim = int(centroids[enc][0].shape[1])
                all_model_embs[mi] = np.zeros((n_files, dim), dtype=np.float32)

    # ── File-outer loop: decode each file ONCE, run every model on it ──
    # All models are already resident (loaded above). Looping over files on
    # the outside reuses one decoded waveform across every model instead of
    # re-decoding the same MP3 once per model (~30 ms/file × 4 models wasted).
    t_total_start = time.time()
    for i, f in enumerate(files):
        waveform = _load_waveform(f, sample_rate)  # decode ONCE, not per model
        if waveform is None:
            for m_idx in range(len(models)):
                all_model_probs[m_idx][i] = uniform
            # embeddings stay zero → max_cosine = 0 → gated to unknown
            continue
        scored[i] = True
        for m_idx, (model, cm) in enumerate(models):
            probs, emb = predict_file_probs_and_embedding(
                model, waveform, device, sample_rate=sample_rate,
                duration_seconds=duration_seconds, eval_hop_ratio=eval_hop_ratio,
                max_eval_windows=max_windows, use_amp=not no_amp,
                temperature=temperature if do_decision else 1.0,
            )
            all_model_probs[m_idx][i] = probs
            if m_idx in all_model_embs:
                all_model_embs[m_idx][i] = emb
        del waveform
    total_elapsed = time.time() - t_total_start

    # ── Apply fusion method ──
    from src.ensemble import (
        weighted_average_fusion,
        geometric_mean_fusion,
        rank_average_fusion,
        max_prob_fusion,
    )

    if fusion_method == "average":
        fused = weighted_average_fusion(all_model_probs, weights=None)
    elif fusion_method == "weighted_average":
        fused = weighted_average_fusion(all_model_probs, weights=fusion_weights)
    elif fusion_method == "geometric_mean":
        fused = geometric_mean_fusion(all_model_probs)
    elif fusion_method == "rank_average":
        fused = rank_average_fusion(all_model_probs)
    elif fusion_method == "max_prob":
        fused = max_prob_fusion(all_model_probs)
    else:
        raise RuntimeError(f"Unknown fusion method: {fusion_method}")

    final_probs = fused / fused.sum(axis=1, keepdims=True)
    labels = final_probs.argmax(axis=1).astype(np.int64)
    max_cosine: Optional[np.ndarray] = None

    # ── Centroid + OOD-gate decision layer ──
    if do_decision and all_model_embs:
        # Ensemble weights for the centroid path (same as head, renormalised
        # over the models that actually have a centroid).
        c_weights = np.asarray(
            [fusion_weights[mi] if fusion_weights is not None
             else 1.0 / len(models)
             for mi in all_model_embs],
            dtype=np.float64,
        )
        c_weights = c_weights / (c_weights.sum() + 1e-12)

        c_probs_list, max_cos_list = [], []
        # The per-model centroid matrix may be WIDER than the competition
        # output: submission/load_centroids merges the 554 unknown-cluster
        # centroids whenever centroids_unknown_<enc>.npz ships next to the
        # known ones, regardless of the checkpoint's own head width. Size the
        # matrix from the actual speaker ids and collapse the cluster tail
        # into unknown when it is, so a legacy 447-way checkpoint + merged
        # centroids (the current shipped hybrid) works exactly like a cluster
        # checkpoint. The max_cosine stays over ALL centroids for the tau gate.
        for mi in all_model_embs:
            cent, sids = centroids[encoder_names[mi]]
            cent_cols = int(sids.max()) + 1
            cp, mc = centroid_probs_matrix(
                all_model_embs[mi], cent, sids, cent_cols, kappa)
            if cent_cols > num_classes:
                cp = _collapse_centroid_probs(cp, num_classes)
            c_probs_list.append(cp)
            max_cos_list.append(mc)

        ens_centroid = np.tensordot(c_weights, np.stack(c_probs_list), axes=(0, 0))
        ens_max_cosine = np.tensordot(c_weights, np.stack(max_cos_list), axes=(0, 0))

        fused2 = alpha * final_probs + (1.0 - alpha) * ens_centroid
        fused2[:, 0] *= lambda_unknown
        fused2 /= (fused2.sum(axis=1, keepdims=True) + 1e-12)
        labels = fused2.argmax(axis=1).astype(np.int64)
        labels[ens_max_cosine < tau] = 0  # hard OOD gate
        final_probs = fused2
        max_cosine = ens_max_cosine

    return {
        "files": files,
        "probs": final_probs,
        "labels": labels,
        "max_cosine": max_cosine,
        "class_map": class_map,
        "scored": scored,
        "total_elapsed": total_elapsed,
    }
