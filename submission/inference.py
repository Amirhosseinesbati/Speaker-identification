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
        # The competition corpus is mostly RIFF/WAVE audio stored under an
        # ``.mp3`` suffix.  Route by file signature, not suffix.  Crucially,
        # a corrupt/empty RIFF file must not fall through to mpg123: that
        # native decoder emits thousands of stderr lines before failing and
        # can obscure a real leaderboard error.  Such files already follow
        # the locked decode-failure -> unknown policy.
        with audio_path.open("rb") as handle:
            magic = handle.read(12)
        is_riff_wave = len(magic) >= 12 and magic[:4] == b"RIFF" and magic[8:12] == b"WAVE"
        is_mpeg = magic[:3] == b"ID3" or (
            len(magic) >= 2 and magic[0] == 0xFF and magic[1] & 0xE0 == 0xE0
        )
        if is_riff_wave or audio_path.suffix.lower() == ".wav":
            wav, sr = sf.read(str(audio_path), dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
        else:
            try:
                wav, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
            except Exception:
                if not is_mpeg:
                    return None
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
    speech_aware: bool = False,
    speech_relative_db: float = 35.0,
    short_audio_mode: str = "pad",
) -> List[torch.Tensor]:
    """Shared window policy — identical to ``SpeakerDataset``."""
    from src.audio_windows import make_eval_windows
    return make_eval_windows(
        waveform,
        target_length=int(sample_rate * duration_seconds),
        hop_ratio=eval_hop_ratio,
        max_windows=max_eval_windows,
        sample_rate=sample_rate,
        speech_aware=speech_aware,
        speech_relative_db=speech_relative_db,
        short_audio_mode=short_audio_mode,
    )


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
    speech_aware: bool = False,
    speech_relative_db: float = 35.0,
    short_audio_mode: str = "pad",
    return_open_set_evidence: bool = False,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, dict]:
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
                           eval_hop_ratio, max_eval_windows,
                           speech_aware=speech_aware,
                           speech_relative_db=speech_relative_db,
                           short_audio_mode=short_audio_mode)

    # Stack all windows into (W, 1, T) — one transfer, one forward pass
    batch = torch.stack(windows).to(device)  # (W, 1, T)

    autocast_ctx = (
        torch.autocast(device_type="cuda", enabled=use_amp and device.type == "cuda")
    )
    with autocast_ctx:
        # Single encoder forward returns BOTH probs and embedding (no second pass).
        if return_open_set_evidence:
            probs, emb, evidence = model.predict_proba_embed_and_evidence(
                batch, temperature=temperature,
            )
        else:
            probs, emb = model.predict_proba_and_embed(batch, temperature=temperature)

    probs = probs.float().cpu().numpy()  # (num_classes,)
    probs = probs / probs.sum()
    emb = emb.float().cpu().numpy()      # (embedding_dim,)
    if not return_open_set_evidence:
        return probs, emb
    return probs, emb, {
        "speaker_probs": evidence["speaker_probs"].float().cpu().numpy(),
        "ood_prob": float(evidence["ood_prob"].float().cpu()),
        "window_agreement": float(evidence["window_agreement"].float().cpu()),
    }


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
    num_unknown_clusters: int = 0,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Load ``centroids_<enc>.npz`` for the given encoders.

    When a ``centroids_unknown_<enc>.npz`` (pseudo-identity cluster centroids
    from ``src/unknown_clustering.py``) sits next to it, the two are merged
    into a wider centroid matrix with speaker ids 1..(446+k) — the
    ``centroid_probs_matrix`` caller then collapses the cluster columns back
    into unknown. With ``num_unknown_clusters > 0`` the k-locked
    ``centroids_unknown_<enc>_k<k>.npz`` file is preferred (several k
    experiments can coexist in the package); the plain name is the fallback.

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
        if num_unknown_clusters > 0:
            k_path = cdir / f"centroids_unknown_{enc}_k{num_unknown_clusters}.npz"
            if k_path.exists():
                cluster_path = k_path
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


def load_prototypes(
    prototypes_dir: str,
    encoder_names: Sequence[str],
    *,
    expected_groups: int = 1000,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Load full-data enrollment embeddings and their internal speaker ids.

    Files are named ``prototypes_<encoder>.npz`` and contain unit-normalised
    ``embeddings`` plus dense ``speaker_ids`` in 1..1000 (446 competition-known
    identities followed by 554 pseudo-unknown identities).  Unlike centroid
    loading, every enrollment utterance remains available to the set scorer.
    """
    output: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    directory = Path(prototypes_dir)
    for encoder in encoder_names:
        path = directory / f"prototypes_{encoder}.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as data:
            embeddings = data["embeddings"].astype(np.float32)
            speaker_ids = data["speaker_ids"].astype(np.int64)
        if embeddings.ndim != 2 or len(embeddings) != len(speaker_ids):
            raise RuntimeError(f"Invalid prototype artifact shape: {path}")
        unique_ids = np.unique(speaker_ids)
        expected_ids = np.arange(1, int(expected_groups) + 1, dtype=np.int64)
        if not np.array_equal(unique_ids, expected_ids):
            raise RuntimeError(f"Prototype speaker ids are not dense in {path}")
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.isfinite(embeddings).all() or not np.allclose(
            norms, 1.0, atol=2e-4
        ):
            raise RuntimeError(f"Prototype embeddings are not finite/unit norm: {path}")
        output[encoder] = (embeddings, speaker_ids)
    return output


def prototype_logmeanexp_probs(
    embeddings: np.ndarray,
    enrollment_embeddings: np.ndarray,
    enrollment_speaker_ids: np.ndarray,
    num_classes: int,
    *,
    beta: float,
    kappa: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Score a test embedding against every enrollment, then pool per identity.

    ``log(mean(exp(beta * cosine))) / beta`` smoothly interpolates between the
    mean and nearest enrollment while normalising for unequal group sizes. The
    554 pseudo-unknown group probabilities are collapsed into competition class
    zero after the 1000-way softmax.
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    enrollment_embeddings = np.asarray(enrollment_embeddings, dtype=np.float32)
    enrollment_speaker_ids = np.asarray(enrollment_speaker_ids, dtype=np.int64)
    if beta <= 0.0 or kappa <= 0.0:
        raise RuntimeError("prototype beta and kappa must be positive")
    if embeddings.ndim != 2 or enrollment_embeddings.ndim != 2:
        raise RuntimeError("prototype scorer expects two embedding matrices")
    if len(enrollment_embeddings) == 0 or len(enrollment_speaker_ids) == 0:
        raise RuntimeError("prototype enrollment is empty")
    if len(enrollment_embeddings) != len(enrollment_speaker_ids):
        raise RuntimeError("prototype embeddings/speaker ids length mismatch")
    if embeddings.shape[1] != enrollment_embeddings.shape[1]:
        raise RuntimeError("test/enrollment embedding dimensions differ")
    unique_ids = np.unique(enrollment_speaker_ids)
    if not np.array_equal(unique_ids, np.arange(1, unique_ids[-1] + 1)):
        raise RuntimeError("prototype speaker ids must be dense and start at one")

    similarities = embeddings @ enrollment_embeddings.T
    scores = np.empty((len(embeddings), len(unique_ids)), dtype=np.float64)
    for column, speaker_id in enumerate(unique_ids):
        values = similarities[:, enrollment_speaker_ids == speaker_id].astype(
            np.float64, copy=False
        )
        scaled = float(beta) * values
        maximum = scaled.max(axis=1, keepdims=True)
        scores[:, column] = (
            maximum[:, 0]
            + np.log(np.exp(scaled - maximum).mean(axis=1))
        ) / float(beta)

    internal = np.zeros((len(embeddings), 1 + len(unique_ids)), dtype=np.float64)
    internal[:, 1:] = _softmax(float(kappa) * scores, axis=1)
    probabilities = (
        _collapse_centroid_probs(internal, num_classes)
        if internal.shape[1] > num_classes
        else internal
    )
    return probabilities, scores.max(axis=1)


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
    prototypes: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
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
        active = [i for i, w in enumerate(fusion_weights)
                  if w > epsilon and checkpoint_path[i] is not None]
        if len(active) < len(fusion_weights):
            skipped = len(fusion_weights) - len(active)
            print(f"[diag] skipped {skipped} zero-weight/missing model(s) — "
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
        models.append((m, cm, cfg))
    if not models:
        raise RuntimeError("No checkpoints to run.")
    class_map = models[0][1]
    first_known_map = {k: v for k, v in class_map.items() if int(v) <= 446}
    for _, other_map, _ in models[1:]:
        other_known_map = {k: v for k, v in other_map.items() if int(v) <= 446}
        if other_known_map != first_known_map:
            raise RuntimeError("Ensemble checkpoints have incompatible known-speaker class maps")
    # Competition output width: 447 legacy; with the closed-set cluster
    # experiment the internal class map is wider (1001) but the model's
    # predict_proba_and_embed already collapses the cluster columns into
    # unknown, so every per-model prob vector is 447-wide.
    num_unknown_clusters = int(models[0][0].num_unknown_clusters)
    # Competition output width — always the fixed 447 regardless of the
    # internal head width (cluster columns are collapsed by the model).
    num_classes = int(models[0][0].num_output_classes)
    n_models = len(models)

    encoder_names = [
        str((cfg.get("model", {}) or {}).get("encoder_type",
            Path(c).name.replace("_best.pt", "")))
        for c, (_, _, cfg) in zip(checkpoint_path, models)
    ]

    # ── Decision-layer setup ──
    do_decision = centroids is not None and decision_params is not None
    decision_params = decision_params or {}
    prototype_config = decision_params.get("prototype_aggregation") or {}
    do_prototype = prototypes is not None and bool(prototype_config)
    if do_decision and do_prototype:
        raise RuntimeError("Centroid and prototype decision layers are mutually exclusive")
    open_set_rule = decision_params.get("open_set_rule") or {}
    do_open_set_rule = bool(open_set_rule)
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
    eval_speech_aware = bool(audio_cfg.get("eval_speech_aware", False))
    speech_relative_db = float(audio_cfg.get("speech_relative_db", 35.0))
    short_audio_mode = str(audio_cfg.get("short_audio_mode", "pad"))

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
    all_model_speaker_probs: Dict[int, np.ndarray] = {}
    all_model_ood_probs: Dict[int, np.ndarray] = {}
    if do_open_set_rule:
        for model_index, (model, _, _) in enumerate(models):
            all_model_speaker_probs[model_index] = np.zeros(
                (n_files, int(model.num_known_speakers)), dtype=np.float64,
            )
            all_model_ood_probs[model_index] = np.zeros(n_files, dtype=np.float64)
    scored = np.zeros(n_files, dtype=bool)

    # Per-model embeddings only for models that have a centroid (decision path).
    all_model_embs: Dict[int, np.ndarray] = {}
    if do_decision or do_prototype:
        for mi, enc in enumerate(encoder_names):
            source = (
                centroids.get(enc) if do_decision and centroids is not None
                else prototypes.get(enc) if prototypes is not None else None
            )
            if source is not None:
                dim = int(source[0].shape[1])
                all_model_embs[mi] = np.zeros((n_files, dim), dtype=np.float32)
    if do_prototype and len(all_model_embs) != len(models):
        missing = sorted(
            enc for enc in encoder_names if prototypes is None or enc not in prototypes
        )
        raise RuntimeError(f"Missing prototype artifacts for active encoders: {missing}")

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
        for m_idx, (model, cm, cfg) in enumerate(models):
            prediction = predict_file_probs_and_embedding(
                model, waveform, device, sample_rate=sample_rate,
                duration_seconds=duration_seconds, eval_hop_ratio=eval_hop_ratio,
                max_eval_windows=max_windows, use_amp=not no_amp,
                temperature=temperature if (do_decision or do_prototype) else 1.0,
                speech_aware=eval_speech_aware,
                speech_relative_db=speech_relative_db,
                short_audio_mode=short_audio_mode,
                return_open_set_evidence=do_open_set_rule,
            )
            if do_open_set_rule:
                probs, emb, evidence = prediction
                all_model_speaker_probs[m_idx][i] = evidence["speaker_probs"]
                all_model_ood_probs[m_idx][i] = evidence["ood_prob"]
            else:
                probs, emb = prediction
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
    open_set_score: Optional[np.ndarray] = None

    # ── Cardinality-normalised pseudo-tail decision rule ──
    # The legacy 1000-way collapse sums all 554 pseudo-unknown probabilities,
    # so the unknown score grows with tail width.  A top-k mean measures local
    # unknown evidence without this cardinality bias.  The threshold is tuned
    # offline on a disjoint calibration set and shipped in decision_config.json.
    if do_open_set_rule:
        rule_type = str(open_set_rule.get("type", "")).lower().strip()
        if rule_type != "tail_topk_mean_plus_ood":
            raise RuntimeError(f"Unsupported open_set_rule type: {rule_type}")
        model_weights = np.asarray(
            fusion_weights if fusion_weights is not None
            else [1.0 / n_models] * n_models,
            dtype=np.float64,
        )
        model_weights /= model_weights.sum() + 1e-12
        widths = {matrix.shape[1] for matrix in all_model_speaker_probs.values()}
        if len(widths) != 1:
            raise RuntimeError("Open-set evidence requires equal speaker-head widths")
        speaker_width = widths.pop()
        num_competition_known = num_classes - 1
        if speaker_width <= num_competition_known:
            raise RuntimeError("Open-set tail rule requires pseudo-unknown speaker columns")
        speaker_probs = np.tensordot(
            model_weights,
            np.stack([all_model_speaker_probs[i] for i in range(n_models)]),
            axes=(0, 0),
        )
        known_probs = speaker_probs[:, :num_competition_known]
        tail_probs = speaker_probs[:, num_competition_known:]
        top_k = max(1, min(int(open_set_rule.get("top_k", 5)), tail_probs.shape[1]))
        tail_topk_mean = np.partition(
            tail_probs, tail_probs.shape[1] - top_k, axis=1,
        )[:, -top_k:].mean(axis=1)
        known_max = known_probs.max(axis=1)
        ood_model_index = int(open_set_rule.get("ood_model_index", 0))
        if ood_model_index not in all_model_ood_probs:
            raise RuntimeError(f"Invalid open-set ood_model_index={ood_model_index}")
        ood_prob = np.clip(all_model_ood_probs[ood_model_index], 1e-9, 1.0 - 1e-9)
        ood_logit = np.log(ood_prob / (1.0 - ood_prob))
        score = (
            np.log(np.clip(tail_topk_mean, 1e-9, 1.0))
            - np.log(np.clip(known_max, 1e-9, 1.0))
            + float(open_set_rule.get("ood_weight", 0.5)) * ood_logit
        )
        labels = known_probs.argmax(axis=1).astype(np.int64) + 1
        labels[score > float(open_set_rule["threshold"])] = 0
        open_set_score = score

    # ── Multi-enrollment log-mean-exp decision layer ──
    if do_prototype and all_model_embs:
        if str(prototype_config.get("type", "")).lower() != "logmeanexp":
            raise RuntimeError("Only prototype_aggregation.type=logmeanexp is supported")
        beta = float(prototype_config.get("beta", 20.0))
        p_weights = np.asarray(
            [fusion_weights[mi] if fusion_weights is not None
             else 1.0 / len(models)
             for mi in all_model_embs],
            dtype=np.float64,
        )
        p_weights /= p_weights.sum() + 1e-12
        probability_list, max_score_list = [], []
        for mi in all_model_embs:
            enrollment, group_ids = prototypes[encoder_names[mi]]
            probability, max_score = prototype_logmeanexp_probs(
                all_model_embs[mi], enrollment, group_ids, num_classes,
                beta=beta, kappa=kappa,
            )
            probability_list.append(probability)
            max_score_list.append(max_score)
        ensemble_prototype = np.tensordot(
            p_weights, np.stack(probability_list), axes=(0, 0)
        )
        ensemble_max_score = np.tensordot(
            p_weights, np.stack(max_score_list), axes=(0, 0)
        )
        fused2 = alpha * final_probs + (1.0 - alpha) * ensemble_prototype
        fused2[:, 0] *= lambda_unknown
        fused2 /= fused2.sum(axis=1, keepdims=True) + 1e-12
        labels = fused2.argmax(axis=1).astype(np.int64)
        labels[ensemble_max_score < tau] = 0
        final_probs = fused2
        max_cosine = ensemble_max_score

    # ── Centroid + OOD-gate decision layer ──
    elif do_decision and all_model_embs:
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

    # Decode failures must be deterministic unknowns, never arbitrary knowns.
    if not np.all(scored):
        labels[~scored] = 0
        final_probs[~scored] = 0.0
        final_probs[~scored, 0] = 1.0

    return {
        "files": files,
        "probs": final_probs,
        "labels": labels,
        "max_cosine": max_cosine,
        "open_set_score": open_set_score,
        "class_map": class_map,
        "scored": scored,
        "total_elapsed": total_elapsed,
    }
