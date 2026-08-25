"""Raw embedding quality benchmark over every cached clean file.

No classifier checkpoint is used.  For each encoder, this reports:
  * leave-one-out top-1/top-5 known-speaker centroid identification;
  * same-speaker vs different-speaker cosine separation and d-prime;
  * OOD AUC using 1 - nearest-known-centroid cosine;
  * best-threshold 447-class Macro-F1 (known IDs + aggregated unknown).

The cache contains 4,459 valid files (70 corrupted files are excluded).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.metrics import macro_f1_score

DATA = ROOT / "data" / "processed"
OUT = ROOT / "reports" / "generated" / "embedding_quality_full_clean.json"
ENCODERS = ["ecapa", "campp", "eres2net", "titanet"]
NUM_CLASSES = 447


def norm(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def pair_stats(emb, labels, seed=42, max_pairs=150_000):
    rng = np.random.default_rng(seed)
    known = labels > 0
    same, diff = [], []
    ids = np.unique(labels[known])
    # Equal per-speaker sampling avoids letting speakers with many files win.
    for sid in ids:
        ix = np.flatnonzero(labels == sid)
        if len(ix) >= 2:
            take = min(8, len(ix))
            a = rng.choice(ix, take, replace=False)
            b = rng.choice(ix, take, replace=False)
            same.extend((emb[a] * emb[b]).sum(axis=1).tolist())
    # Random cross-speaker pairs.
    k = min(max_pairs, int(4 * len(same) + 1))
    a = rng.choice(np.flatnonzero(known), k, replace=True)
    b = rng.choice(np.flatnonzero(known), k, replace=True)
    mask = labels[a] != labels[b]
    diff = (emb[a[mask]] * emb[b[mask]]).sum(axis=1).tolist()
    same = np.asarray(same, dtype=np.float64)
    diff = np.asarray(diff, dtype=np.float64)
    pooled = np.sqrt((same.var() + diff.var()) / 2.0 + 1e-12)
    return {"same_mean": float(same.mean()), "same_std": float(same.std()),
            "different_mean": float(diff.mean()), "different_std": float(diff.std()),
            "d_prime": float((same.mean() - diff.mean()) / pooled),
            "same_pairs": int(len(same)), "different_pairs": int(len(diff))}


def leave_one_out_scores(emb, labels):
    known_ids = np.arange(1, NUM_CLASSES)
    sums = np.zeros((NUM_CLASSES, emb.shape[1]), dtype=np.float32)
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for sid in known_ids:
        m = labels == sid
        if m.any():
            sums[sid] = emb[m].sum(axis=0)
            counts[sid] = int(m.sum())
    full_cent = norm(sums[1:] / np.maximum(counts[1:, None], 1))
    known_ix = np.flatnonzero(labels > 0)
    sims = emb[known_ix] @ full_cent.T
    # Remove each query from its own centroid (the only acceptable known-side
    # evaluation when the entire clean set is used).
    for row, ix in enumerate(known_ix):
        sid = int(labels[ix])
        n = counts[sid]
        if n > 1:
            loo = norm(((sums[sid] - emb[ix]) / (n - 1))[None, :])[0]
            sims[row] = emb[ix] @ full_cent.T
            sims[row, sid - 1] = emb[ix] @ loo
        else:
            sims[row, sid - 1] = -1.0
    order = np.argsort(sims, axis=1)[:, ::-1]
    top1 = order[:, 0] + 1
    top5 = order[:, :5] + 1
    return known_ix, sims, top1, top5, full_cent


def evaluate_encoder(enc):
    tr = norm(np.load(DATA / f"embeddings_train_{enc}.npy").astype(np.float32))
    va = norm(np.load(DATA / f"embeddings_val_{enc}.npy").astype(np.float32))
    ytr = np.load(DATA / f"embeddings_train_{enc}_labels.npy").astype(np.int64)
    yva = np.load(DATA / f"embeddings_val_{enc}_labels.npy").astype(np.int64)
    emb = np.vstack([tr, va])
    y = np.concatenate([ytr, yva])
    known_ix, known_sims, top1, top5, cent = leave_one_out_scores(emb, y)
    known_y = y[known_ix]
    known_dist = 1.0 - known_sims.max(axis=1)
    unknown_ix = np.flatnonzero(y == 0)
    unknown_dist = 1.0 - (emb[unknown_ix] @ cent.T).max(axis=1)
    ood_y = np.r_[np.zeros(len(known_dist)), np.ones(len(unknown_dist))]
    ood_score = np.r_[known_dist, unknown_dist]
    auc = float(roc_auc_score(ood_y, ood_score))
    pred_known = top1.copy()
    best = {"macro_f1": -1.0, "threshold": None}
    for t in np.arange(0.02, 0.981, 0.01):
        pred = pred_known.copy()
        pred[known_dist >= t] = 0
        # Add unknown queries to the prediction vector for competition-space F1.
        all_pred = np.r_[pred, np.zeros(len(unknown_dist), dtype=np.int64)]
        all_true = np.r_[known_y, np.zeros(len(unknown_dist), dtype=np.int64)]
        # Unknown predictions are only unknown when rejected; otherwise use
        # their nearest known identity.
        u_sim = emb[unknown_ix] @ cent.T
        u_pred = u_sim.argmax(axis=1) + 1
        u_pred[unknown_dist >= t] = 0
        all_pred[len(pred):] = u_pred
        score = float(macro_f1_score(all_true, all_pred, NUM_CLASSES))
        if score > best["macro_f1"]:
            best = {"macro_f1": score, "threshold": float(t)}
    return {
        "n_files": int(len(y)), "n_known_files": int((y > 0).sum()),
        "n_unknown_files": int((y == 0).sum()),
        "embedding_dim": int(emb.shape[1]),
        "known_loo_top1": float((top1 == known_y).mean()),
        "known_loo_top5": float((top5 == known_y[:, None]).any(axis=1).mean()),
        "ood_auc": auc,
        "best_macro_f1": best,
        "pair_separation": pair_stats(emb, y),
    }


def main():
    out = {"scope": "All cached clean files; 70 corrupted files excluded; known scores are leave-one-out.",
           "encoders": {enc: evaluate_encoder(enc) for enc in ENCODERS}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
