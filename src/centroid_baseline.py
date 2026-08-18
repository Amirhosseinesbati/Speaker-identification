"""
Centroid Baseline + Fusion for Open-Set Speaker Identification.

Why: the problem is few-shot (~5 files per known speaker), so a centroid
classifier over strong frozen embeddings is usually at least as good as a
classifier trained from scratch — and it is the fusion partner for the
trained heads.

This module:
  1. Builds an idempotent embedding cache (multi-window ECAPA embeddings for
     the leak-free train/val split) at data/processed/embeddings_{train,val}.npy.
  2. Trains per-speaker centroids on the train split.
  3. Tunes an OOD threshold on the val split **for Macro-F1** (the competition
     metric — not binary F1).
  4. Reports Macro-F1 of: centroid-only, trained-model-only (if a checkpoint
     exists), and their weighted fusion.
  5. (NEW) Multi-encoder centroid ensemble: build centroids from multiple frozen
     encoders and fuse their predictions with the same 6 fusion methods.

Run:
    # Single encoder
    uv run --no-sync python -m src.centroid_baseline

    # Multi-encoder centroid ensemble
    uv run --no-sync python -m src.centroid_baseline \\
        --encoders ecapa campp eres2net titanet
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Windows cp1252 fix: force UTF-8 stdio so emoji output never crashes.
from src.cli_utils import setup_utf8_stdio
setup_utf8_stdio()

DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
CKPT_DIR = PROJECT_ROOT / "checkpoints"
CONFIG_PATH = PROJECT_ROOT / "configs" / "default_config.yaml"
RESULTS_PATH = DATA_PROCESSED / "centroid_baseline_results.json"

TARGET_SR = 16000
DURATION_SECONDS = 8.0
EVAL_HOP_RATIO = 0.5
MAX_EVAL_WINDOWS = 8
BATCH_SIZE = 32
RANDOM_SEED = 42
MIN_VALID_DURATION = 1.0


# ────────────────────────────────────────────────────────────────
#  1. Embedding cache (idempotent)
# ────────────────────────────────────────────────────────────────

def _active_encoder(config_path: str = str(CONFIG_PATH)) -> str:
    """encoder_type from the given config (default: configs/default_config.yaml)."""
    from src.data_pipeline import load_config
    return str(load_config(config_path).get("model", {}).get("encoder_type", "ecapa"))


def _cache_paths(encoder: Optional[str] = None) -> dict:
    """Embedding cache paths keyed by encoder (192-d vs 512-d must not mix)."""
    if encoder is None:
        encoder = _active_encoder()
    return {
        "train_embs": DATA_PROCESSED / f"embeddings_train_{encoder}.npy",
        "val_embs": DATA_PROCESSED / f"embeddings_val_{encoder}.npy",
        "train_labels": DATA_PROCESSED / f"embeddings_train_{encoder}_labels.npy",
        "val_labels": DATA_PROCESSED / f"embeddings_val_{encoder}_labels.npy",
        "meta": DATA_PROCESSED / f"embeddings_{encoder}_meta.json",
    }


def build_embedding_cache(force: bool = False,
                          encoder_type: Optional[str] = None) -> dict:
    """Idempotent: multi-window embeddings for the leak-free split.

    The cache is keyed by the active encoder (model.encoder_type, or
    --encoder-type) so 192-d and 512-d caches never mix. Metadata (encoder,
    dim, pooling) is written alongside for validation at fusion time.
    """
    encoder = encoder_type or _active_encoder()
    paths = _cache_paths(encoder)
    cached = all(p.exists() for p in paths.values())

    if cached and not force:
        print(f"  ✅ Embedding cache found ({encoder}) — loading.")
        return {
            "train_embs": np.load(paths["train_embs"]),
            "val_embs": np.load(paths["val_embs"]),
            "train_labels": np.load(paths["train_labels"]),
            "val_labels": np.load(paths["val_labels"]),
            "encoder": encoder,
            "dim": int(np.load(paths["train_embs"]).shape[1]),
        }

    from src.data_pipeline import load_config, prepare_clean_split
    from src.eda_embeddings import extract_embeddings

    config = load_config(str(CONFIG_PATH))
    data_cfg = config["data"]
    audio_cfg = config["audio"]

    train_df, val_df, class_map = prepare_clean_split(
        labels_path=data_cfg["labels_path"],
        audio_dir=data_cfg["audio_dir"],
        processed_labels=data_cfg["processed_labels"],
        val_per_known=1,
        unknown_val_ratio=0.2,
        min_valid_duration=audio_cfg.get("min_valid_duration", MIN_VALID_DURATION),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Extracting train embeddings ({len(train_df):,} files, {encoder})...")
    train_embs = extract_embeddings(train_df, Path(data_cfg["audio_dir"]), device,
                                    encoder_type=encoder)
    print(f"  Extracting val embeddings ({len(val_df):,} files, {encoder})...")
    val_embs = extract_embeddings(val_df, Path(data_cfg["audio_dir"]), device,
                                  encoder_type=encoder)

    train_labels = train_df["label"].values.astype(np.int64)
    val_labels = val_df["label"].values.astype(np.int64)

    for name, arr in [
        ("train_embs", train_embs), ("val_embs", val_embs),
        ("train_labels", train_labels), ("val_labels", val_labels),
    ]:
        np.save(paths[name], arr)

    # Metadata sidecar — read by CentroidFuser at inference to guarantee the
    # cache matches the checkpoint's encoder + embedding dim.
    enc_cfg = config["model"].get("encoder_config", {}).get(encoder, {})
    meta = {
        "encoder": encoder,
        "dim": int(train_embs.shape[1]),
        "pooling": enc_cfg.get("pooling_type")
                   or config["model"].get("pooling_type", "identity"),
        "n_train": int(len(train_embs)),
        "n_val": int(len(val_embs)),
    }
    paths["meta"].write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  ✅ Embedding cache saved to data/processed/ ({encoder}, {meta['dim']}-d).")

    return {
        "train_embs": train_embs, "val_embs": val_embs,
        "train_labels": train_labels, "val_labels": val_labels,
        "encoder": encoder,
        "dim": meta["dim"],
    }


# ────────────────────────────────────────────────────────────────
#  2. Centroid classifier
# ────────────────────────────────────────────────────────────────

def _l2norm_rows(M: np.ndarray) -> np.ndarray:
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)


def fit_centroids(train_embs: np.ndarray, train_labels: np.ndarray):
    """Per-known-speaker L2-normalised centroids."""
    known_mask = train_labels > 0
    known_embs = train_embs[known_mask]
    known_ids = train_labels[known_mask]
    speakers = np.unique(known_ids)
    centroids = np.stack([known_embs[known_ids == s].mean(axis=0) for s in speakers])
    centroids = _l2norm_rows(centroids)
    return centroids, speakers


def _encoder_key_from_checkpoint(checkpoint_path: str) -> str:
    """'checkpoints/campp_best.pt' → 'campp' (matches build_submission naming)."""
    return Path(checkpoint_path).name.replace("_best.pt", "")


def build_checkpoint_centroids(
    checkpoint_path: str,
    device: torch.device,
    batch_size: int = 32,
    max_eval_windows: Optional[int] = None,
) -> dict:
    """Build per-known-speaker centroids from a trained checkpoint's ArcFace
    embedding space, aligned to global class ids 1..num_known.

    Embeddings are extracted from the leak-free TRAIN split (val_per_known=1 —
    the same split the checkpoint was trained/validated on) via ``model.embed``
    (multi-window TTA, mean-then-L2-norm). Val files never contribute to a
    centroid, so this is a no-leak centroid for inference-time cosine scoring.

    Returns:
        dict:
          centroids      — (num_known, D) float32, row i = centroid of class i+1
          speaker_ids    — (num_known,)  int64  = [1, 2, ..., num_known]
          embedding_dim  — int
          encoder        — str
          n_train_files  — int (known files used)
    """
    from torch.utils.data import DataLoader
    from src.data_pipeline import (
        load_config, prepare_clean_split, SpeakerDataset, split_args_from_config,
    )
    from src.model_factory import create_model_from_config

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config") or load_config(str(CONFIG_PATH))
    class_map = checkpoint["class_map"]
    num_known = config.get("model", {}).get("competition_num_known", len(class_map) - 1)

    model = create_model_from_config(config, num_known_speakers=num_known)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    audio_cfg = config["audio"]
    data_cfg = config["data"]

    train_df, _, _ = prepare_clean_split(
        labels_path=data_cfg["labels_path"],
        audio_dir=data_cfg["audio_dir"],
        processed_labels=data_cfg["processed_labels"],
        val_per_known=1,
        unknown_val_ratio=0.2,
        min_valid_duration=audio_cfg.get("min_valid_duration", MIN_VALID_DURATION),
        **split_args_from_config(config),
    )
    # Align to the CHECKPOINT's class_map (not the freshly rebuilt one) so the
    # centroid rows match the global class indices exactly.
    train_df = train_df.copy()
    train_df["label"] = train_df["speaker_id"].map(class_map).astype(int)

    ds = SpeakerDataset(
        train_df, data_cfg["audio_dir"], sample_rate=audio_cfg["sample_rate"],
        duration_seconds=audio_cfg["duration_seconds"], augment=False,
        num_train_windows=audio_cfg.get("num_train_windows", 1),
        eval_hop_ratio=audio_cfg.get("eval_hop_ratio", EVAL_HOP_RATIO),
        max_eval_windows=max_eval_windows or audio_cfg.get("max_eval_windows", MAX_EVAL_WINDOWS),
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    embs, labels = [], []
    with torch.no_grad():
        for windows, lab in tqdm(dl, desc="  Train embeddings", leave=False):
            emb = model.embed(windows.to(device))
            embs.append(emb.cpu().numpy())
            labels.append(lab.numpy())

    embs = np.concatenate(embs, axis=0)
    labels = np.concatenate(labels, axis=0)

    known_mask = labels > 0
    known_embs = embs[known_mask]
    known_ids = labels[known_mask]
    D = embs.shape[1]

    speakers = np.arange(1, num_known + 1)
    centroids = np.zeros((num_known, D), dtype=np.float32)
    for i, sid in enumerate(speakers):
        m = known_ids == sid
        if m.sum() == 0:
            raise RuntimeError(
                f"No train embeddings for speaker {sid} — check the train split "
                f"({checkpoint_path})."
            )
        centroids[i] = known_embs[m].mean(axis=0)
    centroids = _l2norm_rows(centroids).astype(np.float32)

    return {
        "centroids": centroids,
        "speaker_ids": speakers.astype(np.int64),
        "embedding_dim": D,
        "encoder": _encoder_key_from_checkpoint(checkpoint_path),
        "n_train_files": int(len(known_embs)),
    }


def centroid_scores(test_embs: np.ndarray, centroids: np.ndarray):
    """Cosine similarities (M, S) + argmax speaker index."""
    sims = _l2norm_rows(test_embs) @ centroids.T
    return sims, sims.argmax(axis=1)


def centroid_probs_global(
    sims: np.ndarray,
    ood_scores: np.ndarray,
    speakers: np.ndarray,
    num_classes: int = 447,
    kappa: float = 24.0,
) -> np.ndarray:
    """Turn centroid similarities + OOD scores into a 447-way distribution.

    ``known_probs`` is a sharpened softmax (``softmax(kappa * sims)``). Without
    a large ``kappa`` the 446-way softmax is nearly flat (~1/446 per class), so
    ``p[0] = ood_score`` (≈0.3–0.4 for known files) always wins the argmax and
    every file collapses to ``unknown`` (Macro-F1 ≈ 0.004). With ``kappa ≈ 24``
    the softmax is essentially one-hot, so a file close to its own centroid
    scores ``p[speaker] ≈ 1 − p_unknown > p_unknown`` and the argmax is correct —
    the same sharpening the decision layer uses
    (``submission/inference.py::centroid_probs_matrix``).
    """
    from scipy.special import softmax

    M = sims.shape[0]
    probs = np.zeros((M, num_classes), dtype=np.float64)
    known_probs = softmax(float(kappa) * sims, axis=1)
    p_unknown = np.clip(ood_scores, 0.0, 1.0)
    probs[:, 0] = p_unknown
    for i, sid in enumerate(speakers):
        probs[:, int(sid)] = (1.0 - p_unknown) * known_probs[:, i]
    row_sums = probs.sum(axis=1, keepdims=True)
    return probs / (row_sums + 1e-12)


# ────────────────────────────────────────────────────────────────
#  3. Evaluation helpers (Macro-F1 is the objective)
# ────────────────────────────────────────────────────────────────

def evaluate_centroid(
    val_embs: np.ndarray,
    val_labels: np.ndarray,
    centroids: np.ndarray,
    speakers: np.ndarray,
    num_classes: int = 447,
) -> dict:
    """Centroid-only Macro-F1 + OOD threshold tuned for Macro-F1 on val."""
    from src.metrics import macro_f1_score

    sims, argmax_s = centroid_scores(val_embs, centroids)
    ood_scores = 1.0 - sims.max(axis=1)
    pred_global = speakers[argmax_s]

    pure_argmax_mf1 = macro_f1_score(val_labels, pred_global, num_classes=num_classes)

    best = {"threshold": 0.5, "macro_f1": pure_argmax_mf1}
    for thr in np.arange(0.02, 0.98, 0.02):
        preds = np.where(ood_scores > thr, 0, pred_global)
        mf1 = macro_f1_score(val_labels, preds, num_classes=num_classes)
        if mf1 > best["macro_f1"]:
            best = {"threshold": float(thr), "macro_f1": float(mf1)}

    return {
        "pure_argmax_macro_f1": float(pure_argmax_mf1),
        "best_threshold": best["threshold"],
        "best_macro_f1": best["macro_f1"],
        "ood_scores": ood_scores,
        "sims": sims,
        "argmax_global": pred_global,
    }


def load_trained_val_probs(val_df: pd.DataFrame) -> np.ndarray | None:
    """Predict val-set 447-way probabilities with the trained best checkpoint.

    Returns None if no checkpoint exists yet (fusion then reported as N/A).
    """
    ckpt = CKPT_DIR / "best_model.pt"
    if not ckpt.exists():
        print("  ⚠ No best_model.pt found — trained-model / fusion results skipped.")
        return None

    from torch.utils.data import DataLoader
    from src.data_pipeline import load_config, SpeakerDataset
    from src.model_factory import create_model_from_config

    checkpoint = torch.load(ckpt, map_location="cpu", weights_only=False)
    config = checkpoint.get("config") or load_config(str(CONFIG_PATH))
    audio_cfg = config["audio"]
    num_known = config.get("model", {}).get("competition_num_known", 446)

    model = create_model_from_config(config, num_known_speakers=num_known)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    ds = SpeakerDataset(
        val_df, config["data"]["audio_dir"], sample_rate=audio_cfg["sample_rate"],
        duration_seconds=audio_cfg["duration_seconds"], augment=False,
        num_train_windows=audio_cfg.get("num_train_windows", 1),
        eval_hop_ratio=audio_cfg.get("eval_hop_ratio", 0.5),
        max_eval_windows=audio_cfg.get("max_eval_windows", 8),
    )
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_probs = []
    with torch.no_grad():
        for windows, _labels in tqdm(dl, desc="  Trained-model val probs"):
            all_probs.append(model.predict_proba(windows.to(device)).cpu().numpy())
    return np.concatenate(all_probs, axis=0)


def evaluate_fusion(
    model_probs: np.ndarray | None,
    centroid_probs_g: np.ndarray,
    val_labels: np.ndarray,
    num_classes: int = 447,
) -> dict:
    """Weighted fusion of model + centroid probabilities → argmax → Macro-F1."""
    from src.metrics import macro_f1_score

    if model_probs is None:
        return {"available": False, "alphas": [], "best_alpha": None, "best_macro_f1": None}

    results = []
    for alpha in np.arange(0.0, 1.01, 0.1):
        fused = alpha * model_probs + (1.0 - alpha) * centroid_probs_g
        preds = fused.argmax(axis=1)
        mf1 = macro_f1_score(val_labels, preds, num_classes=num_classes)
        results.append({"alpha_model": round(float(alpha), 2), "macro_f1": float(mf1)})

    best = max(results, key=lambda r: r["macro_f1"])
    return {
        "available": True,
        "alphas": results,
        "best_alpha": best["alpha_model"],
        "best_macro_f1": best["macro_f1"],
    }


# ────────────────────────────────────────────────────────────────
#  Multi-encoder centroid ensemble
# ────────────────────────────────────────────────────────────────

def multi_encoder_centroid_ensemble(
    encoders: List[str],
    force_cache: bool = False,
) -> dict:
    """Build centroid classifiers from multiple frozen encoders and fuse them.

    For each encoder:
      1. Build embedding cache (idempotent, keyed by encoder name)
      2. Fit per-speaker centroids on train
      3. Evaluate centroid-only Macro-F1 on val

    Then fuse all centroid probability vectors using the 6 fusion methods
    from src.ensemble and report the best.

    Returns:
        dict with per-encoder results and fusion comparison.
    """
    from src.metrics import macro_f1_score
    from src.data_pipeline import load_config, prepare_clean_split
    from src.ensemble import (
        weighted_average_fusion,
        geometric_mean_fusion,
        rank_average_fusion,
        max_prob_fusion,
        grid_search_weights,
    )

    config = load_config(str(CONFIG_PATH))
    data_cfg = config["data"]

    print("=" * 60)
    print("  Multi-Encoder Centroid Ensemble")
    print("=" * 60)
    print(f"  Encoders: {encoders}")

    # Reusable split (same for all encoders)
    train_df, val_df, class_map = prepare_clean_split(
        labels_path=data_cfg["labels_path"],
        audio_dir=data_cfg["audio_dir"],
        processed_labels=data_cfg["processed_labels"],
        val_per_known=1,
        unknown_val_ratio=0.2,
        min_valid_duration=config["audio"].get("min_valid_duration", MIN_VALID_DURATION),
    )
    num_classes = len(class_map)
    val_labels = val_df["label"].values.astype(np.int64)

    # ── Per-encoder: cache → centroids → evaluate ──
    per_encoder = {}
    all_centroid_probs: List[np.ndarray] = []

    for enc in encoders:
        print(f"\n  [{enc}] Building embedding cache...")
        cache = build_embedding_cache(force=force_cache, encoder_type=enc)
        train_embs, val_embs = cache["train_embs"], cache["val_embs"]
        train_labels = cache["train_labels"]

        print(f"  [{enc}] Fitting centroids ({train_embs.shape[1]}-d)...")
        centroids, speakers = fit_centroids(train_embs, train_labels)
        print(f"  [{enc}] {len(speakers)} centroids")

        print(f"  [{enc}] Evaluating on val...")
        cent = evaluate_centroid(val_embs, val_labels, centroids, speakers, num_classes)

        centroid_probs = centroid_probs_global(
            cent["sims"], cent["ood_scores"], speakers, num_classes=num_classes,
        )
        all_centroid_probs.append(centroid_probs)
        per_encoder[enc] = {
            "macro_f1": float(cent["best_macro_f1"]),
            "pure_argmax_macro_f1": float(cent["pure_argmax_macro_f1"]),
            "best_threshold": float(cent["best_threshold"]),
            "n_centroids": int(len(speakers)),
            "dim": int(train_embs.shape[1]),
        }
        print(f"  [{enc}] Centroid Macro-F1: {cent['best_macro_f1']:.4f} "
              f"(pure argmax: {cent['pure_argmax_macro_f1']:.4f}, "
              f"thr={cent['best_threshold']:.3f})")

    # ── Fuse centroid probs ──
    print(f"\n{'─' * 60}")
    print("  Centroid Ensemble — Fusion Comparison")
    print(f"{'─' * 60}")

    fusion_results = {}

    # Average
    t0 = time.time()
    avg_probs = weighted_average_fusion(all_centroid_probs, weights=None)
    avg_mf1 = macro_f1_score(val_labels, avg_probs.argmax(axis=1), num_classes)
    fusion_results["average"] = {"method": "average", "macro_f1": float(avg_mf1),
                                  "time_s": round(time.time() - t0, 2)}
    print(f"  📊 average (equal weights):         Macro-F1 = {avg_mf1:.4f}")

    # Weighted average (grid search)
    t0 = time.time()
    gs = grid_search_weights(all_centroid_probs, val_labels, num_classes, step=0.05)
    fusion_results["weighted_average"] = {
        "method": "weighted_average (grid search)",
        "macro_f1": float(gs["best_macro_f1"]),
        "weights": gs["best_weights"],
        "time_s": round(time.time() - t0, 2),
    }
    print(f"  📊 weighted_average (grid search):  Macro-F1 = {gs['best_macro_f1']:.4f}  "
          f"weights={gs['best_weights']}")

    # Geometric mean
    t0 = time.time()
    geo_probs = geometric_mean_fusion(all_centroid_probs)
    geo_mf1 = macro_f1_score(val_labels, geo_probs.argmax(axis=1), num_classes)
    fusion_results["geometric_mean"] = {"method": "geometric_mean",
                                         "macro_f1": float(geo_mf1),
                                         "time_s": round(time.time() - t0, 2)}
    print(f"  📊 geometric_mean:                  Macro-F1 = {geo_mf1:.4f}")

    # Rank average
    t0 = time.time()
    rank_probs = rank_average_fusion(all_centroid_probs)
    rank_mf1 = macro_f1_score(val_labels, rank_probs.argmax(axis=1), num_classes)
    fusion_results["rank_average"] = {"method": "rank_average",
                                       "macro_f1": float(rank_mf1),
                                       "time_s": round(time.time() - t0, 2)}
    print(f"  📊 rank_average:                    Macro-F1 = {rank_mf1:.4f}")

    # Max prob
    t0 = time.time()
    max_probs = max_prob_fusion(all_centroid_probs)
    max_mf1 = macro_f1_score(val_labels, max_probs.argmax(axis=1), num_classes)
    fusion_results["max_prob"] = {"method": "max_prob",
                                   "macro_f1": float(max_mf1),
                                   "time_s": round(time.time() - t0, 2)}
    print(f"  📊 max_prob:                        Macro-F1 = {max_mf1:.4f}")

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("  🏆 Centroid Ensemble Summary")
    print(f"{'=' * 60}")

    sorted_methods = sorted(
        fusion_results.items(),
        key=lambda x: x[1]["macro_f1"],
        reverse=True,
    )
    for rank, (method, result) in enumerate(sorted_methods, 1):
        marker = "👑" if rank == 1 else "  "
        extra = ""
        if method == "weighted_average":
            extra = f" weights={result.get('weights', '?')}"
        print(f"  {marker} #{rank}: {result['method']:<35s} "
              f"Macro-F1 = {result['macro_f1']:.4f}{extra}")

    output = {
        "encoders": encoders,
        "num_classes": num_classes,
        "per_encoder": per_encoder,
        "fusion_results": fusion_results,
        "best_method": sorted_methods[0][0],
        "best_macro_f1": sorted_methods[0][1]["macro_f1"],
    }

    results_path = DATA_PROCESSED / "centroid_ensemble_results.json"
    results_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\n  ✓ Results saved to {results_path}")

    return output


# ────────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────────

def main(force_cache: bool = False, encoder_type: Optional[str] = None,
         encoders: Optional[List[str]] = None):
    """Run centroid baseline (single or multi-encoder).

    If `encoders` is given (e.g. ["ecapa", "campp", "eres2net", "titanet"]),
    run the multi-encoder centroid ensemble pipeline. Otherwise, run the
    original single-encoder pipeline.
    """
    if encoders and len(encoders) >= 2:
        multi_encoder_centroid_ensemble(encoders, force_cache=force_cache)
        return

    print("=" * 60)
    print("  Centroid Baseline + Fusion")
    print("=" * 60)

    print("\n[1/4] Embedding cache...")
    cache = build_embedding_cache(force=force_cache, encoder_type=encoder_type)
    train_embs, val_embs = cache["train_embs"], cache["val_embs"]
    train_labels, val_labels = cache["train_labels"], cache["val_labels"]
    print(f"  Train: {train_embs.shape} | Val: {val_embs.shape}")
    print(f"  Train unknown: {(train_labels == 0).sum():,} | "
          f"Val unknown: {(val_labels == 0).sum():,}")

    print("\n[2/4] Fitting centroids on train...")
    centroids, speakers = fit_centroids(train_embs, train_labels)
    print(f"  {len(speakers)} known-speaker centroids ({train_embs.shape[1]}-d)")

    print("\n[3/4] Centroid-only evaluation on val (threshold tuned for Macro-F1)...")
    cent = evaluate_centroid(val_embs, val_labels, centroids, speakers)
    print(f"  Pure argmax Macro-F1:    {cent['pure_argmax_macro_f1']:.4f}")
    print(f"  Best-threshold Macro-F1: {cent['best_macro_f1']:.4f} "
          f"(thr={cent['best_threshold']:.3f})")

    print("\n[4/4] Fusion with trained model (if checkpoint exists)...")
    from src.data_pipeline import load_config, prepare_clean_split

    config = load_config(str(CONFIG_PATH))
    data_cfg = config["data"]
    train_df, val_df, class_map = prepare_clean_split(
        labels_path=data_cfg["labels_path"],
        audio_dir=data_cfg["audio_dir"],
        processed_labels=data_cfg["processed_labels"],
        val_per_known=1,
        unknown_val_ratio=0.2,
        min_valid_duration=config["audio"].get("min_valid_duration", MIN_VALID_DURATION),
    )

    num_classes = len(class_map)
    model_probs = load_trained_val_probs(val_df)

    centroid_probs_g = centroid_probs_global(
        cent["sims"], cent["ood_scores"], speakers, num_classes=num_classes,
    )
    fusion = evaluate_fusion(model_probs, centroid_probs_g, val_labels, num_classes)

    results = {
        "centroid_only": {
            "pure_argmax_macro_f1": cent["pure_argmax_macro_f1"],
            "best_threshold": cent["best_threshold"],
            "best_macro_f1": cent["best_macro_f1"],
        },
        "fusion": fusion,
        "n_centroids": int(len(speakers)),
        "num_classes": num_classes,
    }
    RESULTS_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"\n  ✓ Results saved to {RESULTS_PATH}")

    if fusion["available"]:
        print(f"  Fusion best Macro-F1: {fusion['best_macro_f1']:.4f} "
              f"at alpha_model={fusion['best_alpha']:.1f}")
        for r in fusion["alphas"]:
            print(f"    alpha={r['alpha_model']:.1f} → Macro-F1={r['macro_f1']:.4f}")
    else:
        print("  (Trained-model fusion skipped — no checkpoint yet.)")

    print("\n✅ Centroid baseline complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Centroid baseline + fusion (single or multi-encoder)")
    parser.add_argument("--force-cache", action="store_true",
                        help="Recompute the embedding cache even if it exists")
    parser.add_argument("--encoder-type", default=None,
                        choices=["ecapa", "wavlm", "campp", "eres2net", "titanet"],
                        help="Override the active encoder (single-encoder mode)")
    parser.add_argument("--encoders", nargs="+", default=None,
                        choices=["ecapa", "wavlm", "campp", "eres2net", "titanet"],
                        help="Multi-encoder centroid ensemble "
                             "(e.g. --encoders ecapa campp eres2net titanet)")
    args = parser.parse_args()
    main(force_cache=args.force_cache, encoder_type=args.encoder_type,
         encoders=args.encoders)

