"""
Closed-set 1000-class experiment: recover the 554 collapsed "unknown"
speaker identities by clustering the unlabeled unknown train files.

The competition spec splits 1,000 people ~50/50 into train/eval and collapses
the 554 "unknown/OOD" speakers into one label — so their voices ARE in the
training set (unlabelled). If the unknown train files can be clustered into
~554 coherent pseudo-identities, a 1000-way model can separate them from the
446 known speakers, and at inference the 554 cluster probabilities are summed
back into the "unknown" column of the fixed 447-way output.

This module is **purely additive** — it does not modify any existing module
or config. The cluster-aware training path (Phase 2) is gated behind config
flags that default to OFF (legacy 447-way behaviour preserved exactly).

    python -m src.unknown_clustering validate
        Phase 0 — measure how well clustering recovers TRUE identities on the
        446 labelled speakers (ARI / NMI / purity) and how coherent the 554
        unknown clusters are. Uses the existing frozen-embedding caches
        (data/processed/embeddings_train_<enc>.npy) when present.

    python -m src.unknown_clustering build --k 554 --checkpoint checkpoints/campp_best.pt
        Build — the UI/CI rebuild path. Clusters the unknown TRAIN files at a
        chosen k (any k in [1, #unknown-train-files]; 554 = the true speaker
        count, more = sub-clusters of real identities) and writes the
        pseudo-label map + cluster centroids. No val artifacts required, so it
        runs on a fresh instance before training.

        --force-cache: re-extract the train embeddings when a cache exists.
        The cache is keyed by checkpoint NAME — a NEW model that overwrites
        <enc>_best.pt would otherwise be clustered in the OLD model's space.
        Pass it the FIRST time you cluster a new/retrained checkpoint.

    python -m src.unknown_clustering phase1 --k 554
        Phase 1 — build 554 cluster centroids in the TRAINED campp ArcFace
        space (the same space as the shipped decision layer) and compare val
        Macro-F1 of the 447-centroid vs 1000-centroid decision layer, with
        and without fusion to the trained head probabilities.

Outputs (Phase 1):
    data/processed/unknown_clusters.json      — {audio_file: cluster_id}
    data/processed/centroids_unknown_<enc>.npz — 554 cluster centroids
    data/processed/unknown_cluster_phase1_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cli_utils import setup_utf8_stdio  # noqa: E402

setup_utf8_stdio()

DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
CKPT_DIR = PROJECT_ROOT / "checkpoints"
CLUSTER_MAP_PATH = DATA_PROCESSED / "unknown_clusters.json"
RESULTS_PHASE1 = DATA_PROCESSED / "unknown_cluster_phase1_results.json"

# Same grids as scripts/tune_decision.py / src/decision_engine.py so the
# Phase-1 comparison is apples-to-apples with the shipped decision bundle.
_ALPHA_GRID = np.round(np.arange(0.0, 1.001, 0.05), 2)
_KAPPA_GRID = np.array([0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0])
_TAU_GRID = np.round(np.arange(0.0, 0.60, 0.02), 3)
_LAMBDA_GRID = np.round(np.arange(0.5, 1.601, 0.05), 2)


# ─────────────────────────────────────────────────────────
#  Clustering primitives (pure — no torch)
# ─────────────────────────────────────────────────────────

def _l2norm_rows(M: np.ndarray) -> np.ndarray:
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)


def cluster_kmeans(embs: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    """Cosine-cluster via L2-normalised KMeans (k known from the spec)."""
    from sklearn.cluster import KMeans

    X = _l2norm_rows(embs).astype(np.float32)
    km = KMeans(n_clusters=int(k), random_state=seed, n_init=10)
    return km.fit_predict(X)


def cluster_agglomerative(embs: np.ndarray, k: int) -> np.ndarray:
    """Average-linkage AHC on cosine distance (speaker-diarisation standard)."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    X = _l2norm_rows(embs).astype(np.float32)
    Z = linkage(pdist(X, metric="cosine"), method="average")
    return fcluster(Z, t=int(k), criterion="maxclust") - 1


def cluster_quality(true_labels: np.ndarray, pred_labels: np.ndarray) -> dict:
    """Unsupervised-clustering quality vs TRUE labels (ARI/NMI/purity).

    ``true_labels`` may use arbitrary integer codes (1..446 for knowns).
    """
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    true = np.asarray(true_labels)
    pred = np.asarray(pred_labels)

    # Purity: fraction of samples assigned to the majority true class per
    # predicted cluster (max-matching).
    uniq_true = np.unique(true)
    contingency = _contingency(pred, true, uniq_true)
    purity = float(contingency.max(axis=1).sum() / len(true))

    return {
        "ari": float(adjusted_rand_score(true, pred)),
        "nmi": float(normalized_mutual_info_score(true, pred)),
        "purity": purity,
        "n_clusters": int(pred.max() + 1),
        "n_true_classes": int(len(uniq_true)),
    }


def _contingency(pred: np.ndarray, true: np.ndarray, uniq_true: np.ndarray) -> np.ndarray:
    """(n_clusters, n_true) contingency table."""
    cont = np.zeros((int(pred.max()) + 1, len(uniq_true)), dtype=np.int64)
    idx = {t: i for i, t in enumerate(uniq_true)}
    for p, t in zip(pred.tolist(), true.tolist()):
        cont[p, idx[t]] += 1
    return cont


def cluster_coherence(embs: np.ndarray, labels: np.ndarray) -> dict:
    """Mean intra-cluster vs inter-cluster cosine similarity (embedding manifold).

    High intra / low inter means the clusters carve tight, separated regions
    — the property the sum-into-unknown trick needs (identity recovery is NOT
    required, only consistency).
    """
    X = _l2norm_rows(embs).astype(np.float64)
    C = X @ X.T  # (N, N) cosine matrix

    n = len(labels)
    intra, inter = [], []
    for i in range(n):
        same = np.where(labels == labels[i])[0]
        diff = np.where(labels != labels[i])[0]
        # Exclude self and average over a bounded sample (N is small here).
        s = same[same != i]
        if len(s) > 0:
            intra.append(float(C[i, s].mean()))
        d = diff[: min(50, len(diff))]  # sample to bound cost
        if len(d) > 0:
            inter.append(float(C[i, d].mean()))

    intra_arr = np.asarray(intra)
    inter_arr = np.asarray(inter)
    return {
        "mean_intra_cosine": float(intra_arr.mean()) if len(intra_arr) else 0.0,
        "mean_inter_cosine": float(inter_arr.mean()) if len(inter_arr) else 0.0,
        "margin": float(intra_arr.mean() - inter_arr.mean()) if len(intra_arr) and len(inter_arr) else 0.0,
    }


def build_centroids(embs: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-cluster L2-normalised mean embeddings → (K, D) + sizes (K,).

    Empty clusters (possible with AHC maxclust) get a zero row — their cosine
    contributions are 0, so they never win an argmax.
    """
    K = int(labels.max()) + 1
    D = embs.shape[1]
    cents = np.zeros((K, D), dtype=np.float32)
    sizes = np.zeros(K, dtype=np.int64)
    for k in range(K):
        m = labels == k
        sizes[k] = int(m.sum())
        if sizes[k] > 0:
            cents[k] = _l2norm_rows(embs[m].mean(axis=0, keepdims=True)).astype(np.float32)[0]
    return cents, sizes


def collapse_unknown_probs(
    probs: np.ndarray,
    num_known_out: int,
    num_clusters: int,
) -> np.ndarray:
    """(N, 1 + num_known_out + num_clusters) → (N, 1 + num_known_out).

    Column 0 (unknown) absorbs the mass of every cluster column; the 446 known
    columns pass through. This is the ONLY place the 1000-way internal space
    meets the fixed 447-way competition output.
    """
    out = np.zeros((probs.shape[0], num_known_out + 1), dtype=probs.dtype)
    out[:, 0] = probs[:, 0] + probs[:, 1 + num_known_out:].sum(axis=1)
    out[:, 1:] = probs[:, 1:1 + num_known_out]
    return out / (out.sum(axis=1, keepdims=True) + 1e-12)


# ─────────────────────────────────────────────────────────
#  Phase 0 — validate the clustering premise
# ─────────────────────────────────────────────────────────

def _load_frozen_cache(encoder: str) -> Tuple[np.ndarray, np.ndarray]:
    """Frozen-encoder train embeddings + labels (single seed-42 split).

    Reuses the caches built by src/centroid_baseline when present; otherwise
    rebuilds them (same idempotent path).
    """
    tr_path = DATA_PROCESSED / f"embeddings_train_{encoder}.npy"
    lb_path = DATA_PROCESSED / f"embeddings_train_{encoder}_labels.npy"
    if tr_path.exists() and lb_path.exists():
        return np.load(tr_path), np.load(lb_path)

    from src.centroid_baseline import build_embedding_cache
    cache = build_embedding_cache(force=False, encoder_type=encoder)
    return cache["train_embs"], cache["train_labels"]


def run_validate(encoder: str = "campp", k_unknown: int = 554) -> dict:
    """Phase 0 — clustering quality on labelled knowns + unknown coherence."""
    embs, labels = _load_frozen_cache(encoder)
    known_mask = labels > 0
    unknown_mask = ~known_mask

    report: dict = {
        "encoder": encoder,
        "n_train_files": int(len(labels)),
        "n_known_files": int(known_mask.sum()),
        "n_unknown_files": int(unknown_mask.sum()),
    }

    # ── Known speakers: does clustering recover true identity? ──
    known_embs = embs[known_mask]
    known_true = labels[known_mask]
    n_true = int(len(np.unique(known_true)))
    report["known_n_speakers"] = n_true

    pred_km = cluster_kmeans(known_embs, n_true)
    report["known_kmeans"] = cluster_quality(known_true, pred_km)

    pred_ahc = cluster_agglomerative(known_embs, n_true)
    report["known_agglomerative"] = cluster_quality(known_true, pred_ahc)

    # ── Unknown speakers: are the ~554 clusters coherent? ──
    unknown_embs = embs[unknown_mask]
    pred_u = cluster_kmeans(unknown_embs, k_unknown)
    sizes = np.bincount(pred_u, minlength=k_unknown)
    report["unknown_kmeans"] = {
        "k": int(k_unknown),
        "n_clusters_nonempty": int((sizes > 0).sum()),
        "cluster_size_min": int(sizes.min()),
        "cluster_size_mean": round(float(sizes.mean()), 2),
        "cluster_size_max": int(sizes.max()),
        "coherence": cluster_coherence(unknown_embs, pred_u),
    }

    print("=" * 60)
    print("  Phase 0 — clustering premise (frozen embeddings)")
    print("=" * 60)
    print(f"  Encoder: {encoder} | train files: {report['n_train_files']:,} "
          f"(known {report['n_known_files']:,} / unknown {report['n_unknown_files']:,})")
    print(f"\n  Known speakers ({n_true}) — does clustering recover identity?")
    for method in ("known_kmeans", "known_agglomerative"):
        q = report[method]
        print(f"    {method:<22s} ARI={q['ari']:.4f}  NMI={q['nmi']:.4f}  "
              f"purity={q['purity']:.4f}")
    print(f"\n  Unknown speakers — cluster coherence (k={k_unknown}):")
    u = report["unknown_kmeans"]
    print(f"    non-empty clusters: {u['n_clusters_nonempty']} / {u['k']}")
    print(f"    cluster sizes: min {u['cluster_size_min']} / "
          f"mean {u['cluster_size_mean']} / max {u['cluster_size_max']}")
    c = u["coherence"]
    print(f"    mean intra-cluster cos: {c['mean_intra_cosine']:.4f} | "
          f"mean inter-cluster cos: {c['mean_inter_cosine']:.4f} | "
          f"margin: {c['margin']:+.4f}")

    out = DATA_PROCESSED / "unknown_cluster_phase0_results.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  ✓ Results saved to {out}")
    return report


# ─────────────────────────────────────────────────────────
#  Cluster-map build (shared by `build` and `phase1`)
# ─────────────────────────────────────────────────────────

def build_cluster_map(
    checkpoint_path: str,
    k_clusters: int = 554,
    device: str = "auto",
    seed: int = 42,
    method: str = "kmeans",
    force_cache: bool = False,
    out_map_path: Optional[str] = None,
    split_scheme: Optional[str] = None,
    fold: Optional[int] = None,
    folds: Optional[int] = None,
    split_seed: Optional[int] = None,
    scope: str = "train",
) -> dict:
    """Cluster the unknown TRAIN files into ``k_clusters`` pseudo-identities.

    This is the single build path for the closed-set 1000-class experiment: it
    produces the pseudo-label map and the cluster centroids that Phase-2
    training and the decision layer consume. Embeddings come from the
    checkpoint's ArcFace space (cached), so the map lives in the exact space
    the shipped decision layer uses.

    ``k_clusters`` is the requested k — the UI knob. Any value in
    ``[1, #unknown-train-files]`` is allowed; k > the true speaker count splits
    real identities into sub-clusters, which is safe because the collapse sums
    every cluster column into unknown at output time (finer granularity may
    help the head, at the cost of smaller/noisier pseudo classes).

    Maps and centroids are K-LOCKED so several k experiments can coexist:
    - map:   ``out_map_path`` (default data/processed/unknown_clusters.json);
             the loader falls back to the committed ``submission/<basename>``
             on a fresh instance, so give each experiment its own filename
             (e.g. ``data/processed/unknown_clusters_k1000.json``).
    - centroids: ``centroids_unknown_<enc>_k<k>.npz`` (immutable per-k record)
             plus the plain ``centroids_unknown_<enc>.npz`` "active" alias
             ONLY when building the default map (legacy consumers).

    Args:
        checkpoint_path: trained checkpoint whose ArcFace space is clustered.
        k_clusters:      requested number of pseudo-identities.
        device:          "auto" | "cpu" | "cuda".
        seed:            kmeans RNG seed (deterministic map rebuilds).
        method:          "kmeans" (default) | "agglomerative".
        force_cache:     re-extract the train embeddings even when a cache
                         exists. The cache is keyed by checkpoint NAME, so a
                         NEW model that overwrites <enc>_best.pt would
                         otherwise be clustered in the OLD model's space —
                         pass this the first time you cluster a new/retrained
                         checkpoint.
        out_map_path:    where to write the pseudo-label map (default
                         data/processed/unknown_clusters.json).

    Returns:
        dict with everything phase1's val comparison needs:
        embs/labels/files/unk_mask/unk_embs/unk_files/known_quality/
        cluster_ids/coherence/cluster_map/centroids/cluster_sizes/ckpt_key.
    """
    import torch

    dev = torch.device("cuda" if (device == "auto" and torch.cuda.is_available())
                       else torch.device(device))

    print("=" * 60)
    print(f"  Build cluster map — k={k_clusters} ({method})")
    print("=" * 60)
    print(f"  Checkpoint: {checkpoint_path} | device: {dev}"
          f"{' | FORCE re-extract embeddings' if force_cache else ''}")

    # ── Train-space embeddings (same as the shipped decision layer) ──
    embs, labels, files = _extract_train_embs(
        checkpoint_path, dev, force=force_cache,
        split_scheme=split_scheme, fold=fold, folds=folds,
        split_seed=split_seed, scope=scope,
    )
    unk_mask = labels == 0
    unk_embs = embs[unk_mask]
    unk_files = [f for f, m in zip(files, unk_mask) if m]
    n_unknown = int(unk_mask.sum())
    if k_clusters > n_unknown:
        raise ValueError(
            f"k_clusters={k_clusters} exceeds the {n_unknown} unknown train "
            f"files — clusters cannot be denser than the data. Lower k (the "
            f"true unknown-speaker count is ~554) or add more unknown files."
        )
    print(f"  Train embeddings: {embs.shape} "
          f"(known {int((~unk_mask).sum()):,} / unknown {n_unknown:,})")

    # ── Known-speaker validation in this space (diagnostic) ──
    known_embs = embs[~unk_mask]
    known_true = labels[~unk_mask]
    num_known_out = 446
    kq = cluster_quality(known_true, cluster_kmeans(known_embs, num_known_out))
    print(f"  Known clustering (trained space): ARI={kq['ari']:.4f} "
          f"purity={kq['purity']:.4f}")

    # ── Cluster the unknown train files ──
    if str(method).lower().startswith("agglo"):
        pred_u = cluster_agglomerative(unk_embs, k_clusters)
    else:
        pred_u = cluster_kmeans(unk_embs, k_clusters, seed=seed)
    sizes = np.bincount(pred_u, minlength=k_clusters)
    coh = cluster_coherence(unk_embs, pred_u)
    print(f"  Unknown clustering ({method}): {int((sizes > 0).sum())}/{k_clusters} "
          f"non-empty | sizes min {int(sizes.min())} / mean {sizes.mean():.2f} / "
          f"max {int(sizes.max())}")
    print(f"    intra-cos={coh['mean_intra_cosine']:.4f} "
          f"inter-cos={coh['mean_inter_cosine']:.4f} "
          f"margin={coh['margin']:+.4f}")

    # ── Persist the pseudo-label map + cluster centroids (Phase 2 reuse) ──
    cluster_map = {f: int(c) for f, c in zip(unk_files, pred_u)}
    actual_k = len(set(cluster_map.values()))
    if actual_k != k_clusters:
        raise ValueError(
            f"clustering produced only {actual_k} non-empty clusters (requested "
            f"{k_clusters}) — some clusters are empty at this k. Lower k "
            f"(clusters cannot be denser than the data's natural grouping) and "
            f"rebuild; the loader validation would reject this map anyway."
        )
    map_path = Path(out_map_path) if out_map_path else CLUSTER_MAP_PATH
    map_path.parent.mkdir(parents=True, exist_ok=True)
    map_path.write_text(
        json.dumps(cluster_map, indent=0, ensure_ascii=False), encoding="utf-8",
    )
    print(f"  ✓ Cluster map saved: {map_path} "
          f"({len(cluster_map)} files, k={actual_k})")

    c_cent, c_sizes = build_centroids(unk_embs, pred_u)
    ckpt_key = Path(checkpoint_path).name.replace("_best.pt", "")
    centroids_paths = [
        DATA_PROCESSED / f"centroids_unknown_{ckpt_key}_k{actual_k}.npz",
    ]
    if map_path == CLUSTER_MAP_PATH:
        # Default map: keep the plain "active" alias for legacy consumers.
        centroids_paths.append(DATA_PROCESSED / f"centroids_unknown_{ckpt_key}.npz")
    for cp in centroids_paths:
        np.savez_compressed(
            cp,
            centroids=c_cent, cluster_sizes=c_sizes,
            cluster_ids=np.arange(actual_k, dtype=np.int64),
        )
    print(f"  ✓ Cluster centroids saved: "
          f"{', '.join(p.name for p in centroids_paths)} "
          f"({int((c_sizes > 0).sum())} non-empty)")

    return {
        "embs": embs, "labels": labels, "files": files,
        "unk_mask": unk_mask, "unk_embs": unk_embs, "unk_files": unk_files,
        "known_quality": kq, "cluster_ids": pred_u, "coherence": coh,
        "cluster_map": cluster_map, "centroids": c_cent,
        "cluster_sizes": c_sizes, "ckpt_key": ckpt_key,
        "num_known_out": num_known_out,
        "actual_k": actual_k, "map_path": str(map_path),
    }


def run_build(
    checkpoint_path: str = "checkpoints/campp_best.pt",
    k_clusters: int = 554,
    device: str = "auto",
    seed: int = 42,
    method: str = "kmeans",
    force_cache: bool = False,
    out: Optional[str] = None,
    split_scheme: Optional[str] = None,
    fold: Optional[int] = None,
    folds: Optional[int] = None,
    split_seed: Optional[int] = None,
    scope: str = "train",
) -> dict:
    """Build-only entry point (no val comparison) — the UI/CI rebuild path."""
    out = build_cluster_map(
        checkpoint_path, k_clusters=k_clusters, device=device, seed=seed,
        method=method, force_cache=force_cache, out_map_path=out,
        split_scheme=split_scheme, fold=fold, folds=folds,
        split_seed=split_seed, scope=scope,
    )
    print(f"\n  ✅ Cluster map ready: k={out['actual_k']} → "
          f"{int((out['cluster_sizes'] > 0).sum())} non-empty pseudo-identities. "
          f"Point model.unknown_cluster_path at {out['map_path']} and set "
          f"model.num_unknown_clusters={out['actual_k']} in the config to train.")
    return out


# ─────────────────────────────────────────────────────────
#  Phase 1 — 1000-centroid decision layer (no training)
# ─────────────────────────────────────────────────────────

def _extract_train_embs(
    checkpoint_path: str,
    device: torch.device,
    batch_size: int = 32,
    force: bool = False,
    split_scheme: Optional[str] = None,
    fold: Optional[int] = None,
    folds: Optional[int] = None,
    split_seed: Optional[int] = None,
    scope: str = "train",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """TRAIN-split embeddings in the checkpoint's ArcFace space.

    Mirrors ``src/decision_engine.dump_val_checkpoint`` (same split args, same
    ``model.embed`` multi-window mean-then-L2norm forward) so the embeddings
    live in the exact space the shipped centroids/val artifacts use.

    The result is cached to ``data/processed/train_emb_<enc>.npy`` (+ labels +
    a sidecar JSON of file names) so re-runs skip the ~4 min encoder pass.
    """
    from torch.utils.data import DataLoader

    import torch
    from src.data_pipeline import (
        SpeakerDataset, prepare_clean_split, split_args_from_config,
    )
    from src.model_factory import create_model_from_config
    from tqdm import tqdm

    ckpt_key = Path(checkpoint_path).name.replace("_best.pt", "")
    scope = str(scope).lower().strip()
    split_tag = str(split_scheme or "checkpoint").lower().strip()
    if scope == "full":
        split_tag = "full"
    elif split_tag == "kfold":
        split_tag = f"kfold{int(folds or 3)}_f{int(fold or 0)}_s{int(split_seed or 42)}"
    cache_key = f"{ckpt_key}_{split_tag}"
    cache_embs = DATA_PROCESSED / f"train_emb_{cache_key}.npy"
    cache_lbls = DATA_PROCESSED / f"train_emb_{cache_key}_labels.npy"
    cache_files = DATA_PROCESSED / f"train_emb_{cache_key}_files.json"
    if not force and cache_embs.exists() and cache_lbls.exists() and cache_files.exists():
        print(f"  ✅ Train embeddings cache found ({ckpt_key}) — loading.")
        return (
            np.load(cache_embs),
            np.load(cache_lbls),
            json.loads(cache_files.read_text(encoding="utf-8")),
        )

    # Extract every usable file only once, then derive fold/single caches by
    # filename. This turns three OOF map builds from three encoder passes into
    # one encoder pass plus cheap deterministic slicing.
    if split_tag != "full":
        full_embs, full_labels, full_files = _extract_train_embs(
            checkpoint_path, device, batch_size=batch_size, force=force,
            split_scheme="full", scope="full",
        )
        ck_for_split = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False)
        cfg_for_split = ck_for_split["config"]
        audio_for_split = cfg_for_split["audio"]
        data_for_split = cfg_for_split["data"]
        split_args = split_args_from_config(cfg_for_split)
        if split_scheme is not None:
            split_args["split_scheme"] = str(split_scheme)
        if fold is not None:
            split_args["fold"] = int(fold)
        if folds is not None:
            split_args["folds"] = int(folds)
        if split_seed is not None:
            split_args["random_seed"] = int(split_seed)
        split_train, _, _ = prepare_clean_split(
            labels_path=data_for_split["labels_path"],
            audio_dir=data_for_split["audio_dir"],
            processed_labels=data_for_split["processed_labels"],
            val_per_known=1,
            unknown_val_ratio=0.2,
            min_valid_duration=audio_for_split.get("min_valid_duration", 1.0),
            clean_duplicates=bool(data_for_split.get("clean_duplicates", True)),
            **split_args,
        )
        wanted = set(split_train["audio_file"].astype(str))
        keep = np.asarray([f in wanted for f in full_files], dtype=bool)
        embs = full_embs[keep]
        labs = full_labels[keep]
        files = [f for f, selected in zip(full_files, keep) if selected]
        np.save(cache_embs, embs)
        np.save(cache_lbls, labs)
        cache_files.write_text(json.dumps(files), encoding="utf-8")
        print(f"  ✓ Derived split embedding cache from full cache: {cache_key} "
              f"({len(files):,} files)")
        return embs, labs, files

    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ck["config"]
    class_map = ck["class_map"]
    # Head width = non-unknown classes in the checkpoint's OWN class map
    # (1000 for a cluster-trained model, 446 legacy). Reading
    # competition_num_known here would build a 446-wide head for a 1000-class
    # checkpoint and crash load_state_dict — the class map is the source of
    # truth (the factory then derives the effective cluster count from it).
    num_known = len(class_map) - 1

    model = create_model_from_config(config, num_known_speakers=num_known)
    model.load_state_dict(ck["model_state_dict"])
    model.to(device).eval()

    audio_cfg = config["audio"]
    data_cfg = config["data"]

    split_args = split_args_from_config(config)
    if split_scheme is not None:
        split_args["split_scheme"] = str(split_scheme)
    if fold is not None:
        split_args["fold"] = int(fold)
    if folds is not None:
        split_args["folds"] = int(folds)
    if split_seed is not None:
        split_args["random_seed"] = int(split_seed)
    if scope == "full":
        split_args["split_scheme"] = "full"

    train_df, _, _ = prepare_clean_split(
        labels_path=data_cfg["labels_path"],
        audio_dir=data_cfg["audio_dir"],
        processed_labels=data_cfg["processed_labels"],
        val_per_known=1,
        unknown_val_ratio=0.2,
        min_valid_duration=audio_cfg.get("min_valid_duration", 1.0),
        clean_duplicates=bool(data_cfg.get("clean_duplicates", True)),
        **split_args,
    )
    # Align labels to the CHECKPOINT's class map (not the fresh one).
    train_df = train_df.copy()
    train_df["label"] = train_df["speaker_id"].map(class_map).astype(int)

    ds = SpeakerDataset(
        train_df, data_cfg["audio_dir"], sample_rate=audio_cfg["sample_rate"],
        duration_seconds=audio_cfg["duration_seconds"], augment=False,
        num_train_windows=audio_cfg.get("num_train_windows", 1),
        eval_hop_ratio=audio_cfg.get("eval_hop_ratio", 0.5),
        max_eval_windows=audio_cfg.get("max_eval_windows", 8),
        eval_speech_aware=audio_cfg.get("eval_speech_aware", False),
        speech_relative_db=audio_cfg.get("speech_relative_db", 35.0),
        short_audio_mode=audio_cfg.get("short_audio_mode", "pad"),
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    embs, labs = [], []
    with torch.no_grad():
        for windows, lab in tqdm(dl, desc="  train embeddings", leave=False):
            embs.append(model.embed(windows.to(device)).cpu().numpy())
            labs.append(lab.numpy())

    embs = np.concatenate(embs, axis=0)
    labs = np.concatenate(labs, axis=0)
    files = train_df["audio_file"].tolist()

    np.save(cache_embs, embs)
    np.save(cache_lbls, labs)
    cache_files.write_text(json.dumps(files), encoding="utf-8")
    print(f"  ✓ Train embeddings cached ({ckpt_key}): {cache_embs.name}")
    return embs, labs, files


def _centroid_eval(
    val_emb: np.ndarray,
    centroids: np.ndarray,
    speaker_ids: np.ndarray,
    num_classes: int,
    kappa: float,
    collapse_clusters: Optional[int],
    num_known_out: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """centroid_probs_matrix → (optional collapse) → (probs, max_cosine)."""
    from submission.inference import centroid_probs_matrix

    probs, max_cos = centroid_probs_matrix(
        val_emb, centroids, speaker_ids, num_classes, kappa,
    )
    if collapse_clusters:
        probs = collapse_unknown_probs(probs, num_known_out, collapse_clusters)
    return probs, max_cos


def _tune_and_report(
    label: str,
    head_probs: Optional[np.ndarray],
    cent_probs: np.ndarray,
    max_cos: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
) -> dict:
    """Coordinate descent over (alpha, kappa already fixed, tau, lambda)."""
    from src.metrics import macro_f1_score

    def evaluate(alpha: float, tau: float, lam: float) -> float:
        if head_probs is not None:
            fused = alpha * head_probs + (1.0 - alpha) * cent_probs
        else:
            fused = cent_probs.copy()
        fused = fused.copy()
        fused[:, 0] *= lam
        fused /= (fused.sum(axis=1, keepdims=True) + 1e-12)
        pred = fused.argmax(axis=1).astype(np.int64)
        pred[max_cos < tau] = 0
        return macro_f1_score(labels, pred, num_classes)

    # Baseline: plain argmax of the (optionally head-fused) centroid probs.
    base_pred = cent_probs.argmax(axis=1).astype(np.int64)
    base_mf1 = macro_f1_score(labels, base_pred, num_classes)

    best = {"alpha": 1.0 if head_probs is not None else None,
            "tau": 0.0, "lambda_unknown": 1.0}
    best_score = base_mf1 if head_probs is None else evaluate(1.0, 0.0, 1.0)

    for rnd in range(3):
        improved = True
        while improved:
            improved = False
            for key in ("alpha", "tau", "lambda_unknown"):
                grid = {"alpha": _ALPHA_GRID, "tau": _TAU_GRID,
                        "lambda_unknown": _LAMBDA_GRID}[key]
                for v in grid:
                    cand = dict(best)
                    if key == "alpha" and head_probs is None:
                        continue
                    cand[key] = float(v)
                    s = evaluate(cand["alpha"] if cand["alpha"] is not None else 1.0,
                                 cand["tau"], cand["lambda_unknown"])
                    if s > best_score:
                        best_score = s
                        best = cand
                        improved = True

    return {
        "label": label,
        "pure_centroid_macro_f1": float(base_mf1),
        "best_macro_f1": float(best_score),
        "best_alpha": best["alpha"],
        "best_tau": float(best["tau"]),
        "best_lambda_unknown": float(best["lambda_unknown"]),
    }


def _val_probs_for_model(
    checkpoint_path: str,
    device: torch.device,
    batch_size: int = 32,
    cluster_map: Optional[Dict[str, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """447-way val probabilities for a checkpoint (optionally cluster-aware).

    Builds the val split from the checkpoint's embedded config — with
    ``cluster_map`` when the checkpoint is 1000-class — and runs the exact
    submission forward path (``predict_proba_and_embed``). Returns
    ``(probs (N, 447), val_labels)`` with labels in the checkpoint's internal
    space (cluster pseudo-ids included).
    """
    import torch

    from src.data_pipeline import (
        SpeakerDataset, prepare_clean_split, split_args_from_config,
    )
    from src.model_factory import create_model_from_config
    from tqdm import tqdm

    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ck["config"]
    class_map = ck["class_map"]
    num_known = len(class_map) - 1

    model = create_model_from_config(config, num_known_speakers=num_known)
    model.load_state_dict(ck["model_state_dict"])
    model.to(device).eval()

    audio_cfg = config["audio"]
    data_cfg = config["data"]

    _, val_df, _ = prepare_clean_split(
        labels_path=data_cfg["labels_path"],
        audio_dir=data_cfg["audio_dir"],
        processed_labels=data_cfg["processed_labels"],
        val_per_known=1,
        unknown_val_ratio=0.2,
        min_valid_duration=audio_cfg.get("min_valid_duration", 1.0),
        **split_args_from_config(config),
        unknown_cluster_map=cluster_map,
    )
    val_df = val_df.copy()
    # Pseudo-identity speakers (unknown_XXXX) only exist in a cluster
    # checkpoint's class map. When a LEGACY checkpoint is evaluated on a
    # cluster-aware val split, its class map does not know them — from its
    # point of view they are the aggregated unknown (label 0), which is also
    # the competition's ground truth for them.
    val_df["label"] = val_df["speaker_id"].map(class_map).fillna(0).astype(int)

    ds = SpeakerDataset(
        val_df, data_cfg["audio_dir"], sample_rate=audio_cfg["sample_rate"],
        duration_seconds=audio_cfg["duration_seconds"], augment=False,
        num_train_windows=audio_cfg.get("num_train_windows", 1),
        eval_hop_ratio=audio_cfg.get("eval_hop_ratio", 0.5),
        max_eval_windows=audio_cfg.get("max_eval_windows", 8),
    )

    probs = np.zeros((len(ds), model.num_output_classes), dtype=np.float32)
    labels = np.zeros((len(ds),), dtype=np.int64)
    with torch.no_grad():
        for i in tqdm(range(len(ds)), desc=f"  [{Path(checkpoint_path).name}] val",
                      leave=False):
            windows, lab = ds[i]
            p, _ = model.predict_proba_and_embed(windows.to(device), temperature=1.0)
            probs[i] = p.cpu().numpy()
            labels[i] = int(lab.item())
    return probs, labels


def run_compare(
    legacy_checkpoint: str = "checkpoints/campp_best.pt",
    cluster_checkpoint: str = "checkpoints/cluster1000/campp_best.pt",
    device: str = "auto",
) -> dict:
    """2×2 val comparison: legacy 447-way vs cluster 1000-way, on both val
    splits (each model's own split AND the other's), Macro-F1 in the fixed
    447-class competition output space."""
    import json as _json
    import torch

    from src.metrics import macro_f1_score

    dev = torch.device("cuda" if (device == "auto" and torch.cuda.is_available())
                       else torch.device(device))
    cluster_map = None
    if CLUSTER_MAP_PATH.exists():
        cluster_map = {k: int(v) for k, v in
                       _json.loads(CLUSTER_MAP_PATH.read_text(encoding="utf-8")).items()}

    print("=" * 60)
    print("  2×2 comparison — legacy 447-way vs cluster 1000-way")
    print("=" * 60)

    models = {
        "legacy_447": (legacy_checkpoint, None),
        "cluster_1000": (cluster_checkpoint, cluster_map),
    }
    results: dict = {}
    for name, (ckpt, cm) in models.items():
        probs, labels = _val_probs_for_model(ckpt, dev, cluster_map=cm)
        # y_true in the competition space: pseudo-cluster ids (>446) → unknown.
        y_true = np.where(labels > 446, 0, labels)
        # argmax over the 447-way output; pseudo-cluster columns already
        # collapsed into 0 by the model.
        y_pred = probs.argmax(axis=1).astype(np.int64)
        results[name] = {
            "n_val": int(len(labels)),
            "n_known": int((y_true > 0).sum()),
            "n_unknown": int((y_true == 0).sum()),
            "macro_f1": float(macro_f1_score(y_true, y_pred, num_classes=447)),
            "known_acc": float((y_pred[y_true > 0] == y_true[y_true > 0]).mean()),
            "unknown_acc": float((y_pred[y_true == 0] == 0).mean()),
        }
        print(f"\n  {name} on its own val split: Macro-F1 = "
              f"{results[name]['macro_f1']:.4f} "
              f"(known_acc {results[name]['known_acc']:.4f}, "
              f"unknown_acc {results[name]['unknown_acc']:.4f})")

    # Cross evaluation: cluster model on the LEGACY val, legacy model on the
    # CLUSTER val.
    print("\n  ── cross evaluation ──")
    # legacy model on cluster val
    probs_lc, labels_lc = _val_probs_for_model(legacy_checkpoint, dev, cluster_map=cluster_map)
    y_true_lc = np.where(labels_lc > 446, 0, labels_lc)
    y_pred_lc = probs_lc.argmax(axis=1).astype(np.int64)
    mf_lc = float(macro_f1_score(y_true_lc, y_pred_lc, num_classes=447))
    # cluster model on legacy val
    probs_cl, labels_cl = _val_probs_for_model(cluster_checkpoint, dev, cluster_map=None)
    y_true_cl = np.where(labels_cl > 446, 0, labels_cl)
    y_pred_cl = probs_cl.argmax(axis=1).astype(np.int64)
    mf_cl = float(macro_f1_score(y_true_cl, y_pred_cl, num_classes=447))

    print(f"  legacy_447 on CLUSTER val:  Macro-F1 = {mf_lc:.4f}")
    print(f"  cluster_1000 on LEGACY val: Macro-F1 = {mf_cl:.4f}")
    results["cross"] = {
        "legacy_on_cluster_val": mf_lc,
        "cluster_on_legacy_val": mf_cl,
    }

    RESULTS_PHASE1.parent.mkdir(parents=True, exist_ok=True)
    out = DATA_PROCESSED / "unknown_cluster_compare.json"
    out.write_text(_json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  ✓ Results saved to {out}")
    return results


def run_phase1(
    checkpoint_path: str,
    k_clusters: int = 554,
    device: str = "auto",
    force_cache: bool = False,
) -> dict:
    """Phase 1 — 447-centroid vs 1000-centroid decision layer on val."""
    import torch

    dev = torch.device("cuda" if (device == "auto" and torch.cuda.is_available())
                       else torch.device(device))
    num_known_out = 446
    num_classes_out = num_known_out + 1

    print("=" * 60)
    print("  Phase 1 — 1000-centroid decision layer (no training)")
    print("=" * 60)
    print(f"  Checkpoint: {checkpoint_path} | clusters k={k_clusters} | "
          f"device: {dev}")

    # ── Train-space embeddings, known-quality, cluster map + centroids ──
    # (shared build path with the `build` subcommand / UI rebuild).
    b = build_cluster_map(
        checkpoint_path, k_clusters=k_clusters, device=device,
        force_cache=force_cache,
    )
    embs, labels = b["embs"], b["labels"]
    unk_embs, unk_files = b["unk_embs"], b["unk_files"]
    kq, coh = b["known_quality"], b["coherence"]
    cluster_map, c_cent, c_sizes = b["cluster_map"], b["centroids"], b["cluster_sizes"]
    ckpt_key = b["ckpt_key"]

    # ── Val artifacts for THIS checkpoint's encoder (same split, trained
    # space). Must be re-dumped for the checkpoint under test —
    # scripts/dump_val_artifacts.py + scripts/build_centroids.py — otherwise a
    # stale artifact from an older model would be compared here.
    def _need(path: Path) -> Path:
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing. Run `uv run --no-sync python "
                f"scripts/dump_val_artifacts.py --checkpoints {checkpoint_path}` "
                f"and `uv run --no-sync python scripts/build_centroids.py "
                f"--checkpoints {checkpoint_path}` for the checkpoint under test."
            )
        return path

    val_emb = np.load(_need(DATA_PROCESSED / f"val_emb_{ckpt_key}.npy")).astype(np.float32)
    val_lab = np.load(_need(DATA_PROCESSED / "val_labels.npy")).astype(np.int64)
    val_head = np.load(_need(DATA_PROCESSED / f"val_probs_{ckpt_key}.npy")).astype(np.float64)
    assert val_emb.shape[0] == len(val_lab)

    # ── Known centroids (shipped) ──
    known_cent = np.load(_need(DATA_PROCESSED / f"centroids_{ckpt_key}.npz"))
    known_ids = known_cent["speaker_ids"].astype(np.int64)
    known_cents = known_cent["centroids"].astype(np.float32)

    # ── 1000-way: known + cluster centroids ──
    all_cents = np.vstack([known_cents, c_cent]).astype(np.float32)
    all_ids = np.concatenate([known_ids, np.arange(num_known_out + 1, num_known_out + 1 + k_clusters, dtype=np.int64)])

    results: dict = {
        "checkpoint": str(checkpoint_path),
        "k_clusters": int(k_clusters),
        "num_known_out": num_known_out,
        "known_clustering": kq,
        "unknown_coherence": coh,
        "cluster_sizes": {
            "nonempty": int((c_sizes > 0).sum()),
            "min": int(c_sizes.min()) if len(c_sizes) else 0,
            "mean": round(float(c_sizes.mean()), 2),
            "max": int(c_sizes.max()),
        },
        "comparison": {},
    }

    # ── Evaluate: 447-centroid baseline vs 1000-centroid ──
    from src.metrics import macro_f1_score

    kappa_sweep = list(_KAPPA_GRID)
    for label, cents, ids, n_cls, collapse in [
        ("centroid_447", known_cents, known_ids, num_classes_out, None),
        ("centroid_1000", all_cents, all_ids, num_known_out + 1 + k_clusters,
         k_clusters),
    ]:
        # Find best kappa on pure centroid first.
        best_kappa = 8.0
        best_pure = -1.0
        for kap in kappa_sweep:
            p, mc = _centroid_eval(val_emb, cents, ids, n_cls, float(kap),
                                   collapse, num_known_out)
            mf1 = macro_f1_score(val_lab, p.argmax(axis=1).astype(np.int64),
                                 num_classes_out)
            if mf1 > best_pure:
                best_pure, best_kappa = mf1, float(kap)

        # Fused with the trained head + tau/lambda tuning at that kappa.
        p, mc = _centroid_eval(val_emb, cents, ids, n_cls, best_kappa,
                               collapse, num_known_out)
        fused = _tune_and_report(
            label, val_head, p, mc, val_lab, num_classes_out,
        )
        fused["best_kappa"] = best_kappa
        fused["pure_argmax_macro_f1"] = best_pure
        results["comparison"][label] = fused
        print(f"\n  {label}: pure centroid Macro-F1 = {best_pure:.4f} "
              f"(kappa={best_kappa:.0f}) | tuned = {fused['best_macro_f1']:.4f} "
              f"(alpha={fused['best_alpha']}, tau={fused['best_tau']:.2f}, "
              f"lambda={fused['best_lambda_unknown']:.2f})")

    # ── Also: the head-only baseline for reference ──
    head_mf1 = macro_f1_score(
        val_lab, val_head.argmax(axis=1).astype(np.int64), num_classes_out,
    )
    results["comparison"]["head_only_argmax"] = {"pure_argmax_macro_f1": float(head_mf1)}
    print(f"\n  head_only_argmax: Macro-F1 = {head_mf1:.4f}")

    RESULTS_PHASE1.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"\n  ✓ Results saved to {RESULTS_PHASE1}")
    return results


# ─────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Unknown-speaker clustering (closed-set 1000-class experiment)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="Phase 0 — clustering premise on frozen embeddings")
    v.add_argument("--encoder", default="campp",
                   choices=["ecapa", "wavlm", "campp", "eres2net", "titanet"])
    v.add_argument("--k-unknown", type=int, default=554)

    bd = sub.add_parser(
        "build",
        help="Build the pseudo-identity cluster map + centroids at a chosen k "
             "(the UI/CI rebuild path; no val comparison).",
    )
    bd.add_argument("--checkpoint", default="checkpoints/campp_best.pt")
    bd.add_argument("--k", type=int, default=554,
                    help="number of pseudo-identities (the UI knob; ~554 = the "
                         "true unknown-speaker count, more splits real speakers "
                         "into sub-clusters).")
    bd.add_argument("--method", default="kmeans",
                    choices=["kmeans", "agglomerative"])
    bd.add_argument("--seed", type=int, default=42)
    bd.add_argument("--out", default=None,
                    help="map output path (default data/processed/"
                         "unknown_clusters.json). Give each k its own file, "
                         "e.g. data/processed/unknown_clusters_k1000.json, and "
                         "point model.unknown_cluster_path at it — several k "
                         "experiments then coexist.")
    bd.add_argument("--device", default="auto", help="auto | cpu | cuda")
    bd.add_argument(
        "--force-cache", action="store_true",
        help="Re-extract the train embeddings even when a cache exists. Use "
             "the FIRST time you cluster a new/retrained checkpoint that "
             "overwrites <enc>_best.pt — the cache is keyed by checkpoint "
             "name, so without this a new model would be clustered in the "
             "OLD model's embedding space.",
    )
    bd.add_argument("--split-scheme", choices=["single", "kfold", "full"], default=None,
                    help="override the checkpoint split used to choose clustering files")
    bd.add_argument("--fold", type=int, default=None)
    bd.add_argument("--folds", type=int, default=None)
    bd.add_argument("--split-seed", type=int, default=None)
    bd.add_argument("--scope", choices=["train", "full"], default="train",
                    help="full clusters every usable unknown file (final retrain only)")

    p1 = sub.add_parser("phase1", help="Phase 1 — 1000-centroid decision layer on val")
    p1.add_argument("--checkpoint", default="checkpoints/campp_best.pt")
    p1.add_argument("--k", type=int, default=554)
    p1.add_argument("--device", default="auto", help="auto | cpu | cuda")
    p1.add_argument(
        "--force-cache", action="store_true",
        help="Re-extract the train embeddings (see `build --help`).",
    )

    cp = sub.add_parser("compare", help="2×2 val comparison: legacy vs cluster model")
    cp.add_argument("--legacy-checkpoint", default="checkpoints/campp_best.pt")
    cp.add_argument("--cluster-checkpoint", default="checkpoints/cluster1000/campp_best.pt")
    cp.add_argument("--device", default="auto", help="auto | cpu | cuda")

    args = parser.parse_args()

    if args.cmd == "validate":
        run_validate(encoder=args.encoder, k_unknown=args.k_unknown)
    elif args.cmd == "build":
        run_build(checkpoint_path=args.checkpoint, k_clusters=args.k,
                  device=args.device, seed=args.seed, method=args.method,
                  force_cache=args.force_cache, out=args.out,
                  split_scheme=args.split_scheme, fold=args.fold,
                  folds=args.folds, split_seed=args.split_seed,
                  scope=args.scope)
    elif args.cmd == "phase1":
        run_phase1(checkpoint_path=args.checkpoint, k_clusters=args.k,
                   device=args.device, force_cache=args.force_cache)
    elif args.cmd == "compare":
        run_compare(legacy_checkpoint=args.legacy_checkpoint,
                    cluster_checkpoint=args.cluster_checkpoint,
                    device=args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
