"""
Fuse the frozen-encoder centroid ensemble with the trained decision layer.

Tests (no retraining) whether adding the raw-space 4-encoder centroid ensemble
(from ``src.centroid_baseline``, best val Macro-F1 0.9496) to the trained campp
decision bundle (head probs + ArcFace-space centroids, val 0.9563) improves the
competition Macro-F1.

Inputs (already dumped under data/processed/):
  - frozen raw embeddings + labels:  embeddings_{train,val}_{enc}.npy
  - trained head probs:              val_probs_campp.npy
  - trained ArcFace embeddings:      val_emb_campp.npy  (already L2-normalised)
  - trained ArcFace centroids:       centroids_campp.npz
  - shared val labels:               val_labels.npy

Usage:
    uv run --no-sync python scripts/fuse_frozen_trained.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics import macro_f1_score  # noqa: E402

DP = ROOT / "data" / "processed"
NUM_CLASSES = 447
KAPPA = 24.0


def l2norm_rows(M: np.ndarray) -> np.ndarray:
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)


def fit_centroids(train_embs: np.ndarray, train_labels: np.ndarray):
    """Per-known-speaker L2-normalised centroids (same as centroid_baseline)."""
    known_mask = train_labels > 0
    known_embs = train_embs[known_mask]
    known_ids = train_labels[known_mask]
    speakers = np.unique(known_ids)
    centroids = np.stack([known_embs[known_ids == s].mean(axis=0) for s in speakers])
    centroids = l2norm_rows(centroids)
    return centroids, speakers


def centroid_probs_matrix(embs, centroids, speaker_ids, num_classes, kappa):
    """Mirror of submission.inference.centroid_probs_matrix (self-contained)."""
    cos = embs @ centroids.T  # (N, S) — embs and centroids are unit-norm
    max_cosine = cos.max(axis=1)
    x = kappa * cos
    x -= x.max(axis=1, keepdims=True)
    e = np.exp(x)
    known = e / (e.sum(axis=1, keepdims=True) + 1e-12)
    p_unknown = np.clip(1.0 - max_cosine, 0.0, 1.0)
    probs = np.zeros((embs.shape[0], num_classes), dtype=np.float64)
    probs[:, 0] = p_unknown
    for j, sid in enumerate(speaker_ids):
        probs[:, int(sid)] = (1.0 - p_unknown) * known[:, j]
    probs /= (probs.sum(axis=1, keepdims=True) + 1e-12)
    return probs, max_cosine


def main() -> None:
    labels = np.load(DP / "val_labels.npy").astype(np.int64)
    n = len(labels)

    # ── 1. Frozen-encoder centroid ensemble (raw space, 4 encoders) ──
    frozen_encs = ["ecapa", "campp", "eres2net", "titanet"]
    frozen_weights = np.array([0.0, 0.35, 0.65, 0.0])  # from centroid_ensemble_results.json
    frozen_probs = []
    for enc in frozen_encs:
        tr = np.load(DP / f"embeddings_train_{enc}.npy")
        tr_lab = np.load(DP / f"embeddings_train_{enc}_labels.npy").astype(np.int64)
        va = np.load(DP / f"embeddings_val_{enc}.npy")
        va_lab = np.load(DP / f"embeddings_val_{enc}_labels.npy").astype(np.int64)
        assert (va_lab == labels).all(), f"label order mismatch for frozen {enc}"
        centroids, speakers = fit_centroids(tr, tr_lab)
        cp, _ = centroid_probs_matrix(l2norm_rows(va), centroids, speakers,
                                      NUM_CLASSES, KAPPA)
        frozen_probs.append(cp)
    frozen_ens = np.tensordot(frozen_weights, np.stack(frozen_probs), axes=(0, 0))
    frozen_ens /= frozen_ens.sum(axis=1, keepdims=True) + 1e-12

    # ── 2. Trained campp decision (head + ArcFace centroid) ──
    head = np.load(DP / "val_probs_campp.npy").astype(np.float64)     # (N, 447)
    val_emb = np.load(DP / "val_emb_campp.npy").astype(np.float64)    # unit-norm
    z = np.load(DP / "centroids_campp.npz")
    cent_t, mc_t = centroid_probs_matrix(val_emb, z["centroids"].astype(np.float64),
                                         z["speaker_ids"].astype(np.int64),
                                         NUM_CLASSES, KAPPA)

    # ── Baselines (reproduce known numbers) ──
    def apply_gate(fused, lam, mc, tau=0.0):
        fused = fused.copy()
        fused[:, 0] *= lam
        fused /= (fused.sum(axis=1, keepdims=True) + 1e-12)
        pred = fused.argmax(axis=1).astype(np.int64)
        pred[mc < tau] = 0
        return pred

    trained_base = 0.2 * head + 0.8 * cent_t
    mf1_trained = macro_f1_score(labels, apply_gate(trained_base, 1.05, mc_t), NUM_CLASSES)
    mf1_frozen = macro_f1_score(labels, frozen_ens.argmax(axis=1), NUM_CLASSES)
    print(f"Trained campp decision (reproduced): {mf1_trained:.4f}")
    print(f"Frozen 4-encoder ensemble alone:      {mf1_frozen:.4f}")

    # ── 3-way fusion sweep: a*head + b*centroid_t + c*frozen ──
    best = None
    best_mf1 = -1.0
    results = []
    for lam in (1.0, 1.05, 1.1):
        for a in np.arange(0.0, 1.0001, 0.1):
            for b in np.arange(0.0, 1.0001 - a, 0.1):
                c = 1.0 - a - b
                fused = a * head + b * cent_t + c * frozen_ens
                pred = apply_gate(fused, lam, mc_t)
                mf1 = macro_f1_score(labels, pred, NUM_CLASSES)
                results.append((mf1, a, b, c, lam))
                if mf1 > best_mf1:
                    best_mf1 = mf1
                    best = (a, b, c, lam)

    results.sort(reverse=True)
    print(f"\nBest 3-way fusion: Macro-F1 = {best_mf1:.4f}  "
          f"(head={best[0]:.2f}, centroid_t={best[1]:.2f}, frozen={best[2]:.2f}, lam={best[3]})")
    print(f"Δ vs trained-only: {best_mf1 - mf1_trained:+.4f}")
    print("\nTop 5:")
    for mf1, a, b, c, lam in results[:5]:
        print(f"  {mf1:.4f}  head={a:.1f} cent_t={b:.1f} frozen={c:.1f} lam={lam}")


if __name__ == "__main__":
    main()
