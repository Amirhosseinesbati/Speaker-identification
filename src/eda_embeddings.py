"""
Phase 3 (unbiased) — Embedding-Space & OOD Separability EDA
for IAAA Competition 2026: Open-Set Speaker Identification

Goal
----
Quantify how separable the speaker identities are **in embedding space** using
the frozen ECAPA-TDNN encoder, **without the in-sample bias** of the original
Phase 3 report:

  1. Drop corrupted (< 1 s) and MD5-duplicate files **before** embedding.
  2. Extract multi-window (8 s sliding, 50% overlap) embeddings per file —
     matching what the training/inference pipeline actually sees.
  3. **Leave-one-out (LOO)** centroid evaluation → honest recognition ceiling.
  4. OOD feasibility via nearest-known-centroid distance (LOO for known files).
  5. Unknown internal structure via KMeans (pseudo-labelling feasibility).
  6. **Macro-F1 simulation** — the only number that actually matters for the
     competition metric (macro-F1 over all 447 classes).

Outputs (into eda/):
  - phase3_similarity_histograms.png
  - phase3_ood_score_histograms.png
  - phase3_roc_curve.png
  - phase3_tsne_known_unknown.png
  - phase3_tsne_top_speakers.png
  - phase3_embedding_summary.json   (unbiased numbers, labelled "unbiased")
  - Phase3_Embedding_EDA_Report.md  (markdown report)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

# ────────────────────────────────────────────────────────────────
#  Paths / config
# ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
EDA_DIR = PROJECT_ROOT / "eda"

LABELS_PATH = DATA_PROCESSED / "audio_wav_labels.csv"
WAV_AUDIO_DIR = DATA_PROCESSED / "audio_wav"

PLOT_SIM = EDA_DIR / "phase3_similarity_histograms.png"
PLOT_OOD = EDA_DIR / "phase3_ood_score_histograms.png"
PLOT_ROC = EDA_DIR / "phase3_roc_curve.png"
PLOT_T1 = EDA_DIR / "phase3_tsne_known_unknown.png"
PLOT_T2 = EDA_DIR / "phase3_tsne_top_speakers.png"
JSON_OUT = EDA_DIR / "phase3_embedding_summary.json"
REPORT = EDA_DIR / "Phase3_Embedding_EDA_Report.md"

TARGET_SR = 16000
DURATION_SECONDS = 8.0
EVAL_HOP_RATIO = 0.5
MAX_EVAL_WINDOWS = 8
BATCH_SIZE = 32
MIN_VALID_DURATION = 1.0
TSNE_SAMPLE = 1800          # points for t-SNE
MAX_CROSS_PAIRS = 50_000    # cap for cross-speaker similarity pairs
MAX_SAME_PAIRS = 50_000     # cap for same-speaker pairs
TOP_SPEAKERS_TSNE = 12
RANDOM_SEED = 42

sys.path.insert(0, str(PROJECT_ROOT))


# ────────────────────────────────────────────────────────────────
#  1. Load labels & drop corrupted / duplicate files
# ────────────────────────────────────────────────────────────────

def load_labels(path: Path) -> pd.DataFrame:
    """Load processed labels and add an `is_unknown` boolean column."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["speaker_id"] = df["speaker_id"].astype(str).str.strip()
    df["audio_file"] = df["audio_file"].astype(str).str.strip()
    df["is_unknown"] = df["speaker_id"].str.lower() == "unknown"
    return df


def clean_labels(df: pd.DataFrame, wav_dir: Path) -> tuple:
    """
    Drop corrupted (< min_valid_duration) and MD5-duplicate files.

    Returns:
        (clean_df, stats) where stats records how many files were dropped.
    """
    from src.data_pipeline import find_corrupted_files, find_duplicate_groups

    corrupted = find_corrupted_files(df, str(wav_dir), MIN_VALID_DURATION)
    dup_groups = find_duplicate_groups(df, str(wav_dir))

    drop = set(corrupted)
    dup_files = set(f for files in dup_groups.values() for f in files)
    n_corr_known = int(df[df["audio_file"].isin(corrupted) & ~df["is_unknown"]].shape[0])
    n_corr_unknown = int(df[df["audio_file"].isin(corrupted) & df["is_unknown"]].shape[0])

    clean = df[~df["audio_file"].isin(drop)].reset_index(drop=True)

    stats = {
        "n_raw_files": int(len(df)),
        "n_corrupted_dropped": len(corrupted),
        "n_corrupted_known": n_corr_known,
        "n_corrupted_unknown": n_corr_unknown,
        "n_duplicate_groups": len(dup_groups),
        "n_duplicate_files": len(dup_files),
        "n_files_after_clean": int(len(clean)),
    }
    return clean, stats


# ────────────────────────────────────────────────────────────────
#  2. Multi-window embedding extraction (matches pipeline inference)
# ────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(df: pd.DataFrame, wav_dir: Path, device: torch.device) -> np.ndarray:
    """Extract one 192-d ECAPA embedding per file (multi-window TTA).

    Windows of a batch are forwarded **one window index at a time** so peak
    VRAM stays at (batch_size, 1, T) instead of (batch_size * W, 1, T) — the
    GTX 1660 Ti (6 GB) OOMs on the full reshape for large W.
    """
    from torch.utils.data import DataLoader
    from src.encoders import ECAPAEncoder
    from src.data_pipeline import SpeakerDataset, create_class_mapping

    class_map = create_class_mapping(df)
    df = df.copy()
    df["label"] = df["speaker_id"].map(class_map)

    encoder = ECAPAEncoder(
        source="speechbrain/spkrec-ecapa-voxceleb",
        freeze_encoder=True,
    ).to(device)
    encoder.eval()

    ds = SpeakerDataset(
        df, str(wav_dir), sample_rate=TARGET_SR, duration_seconds=DURATION_SECONDS,
        augment=False, eval_hop_ratio=EVAL_HOP_RATIO, max_eval_windows=MAX_EVAL_WINDOWS,
    )
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    out_dim = encoder.output_dim
    embs = np.zeros((len(ds), out_dim), dtype=np.float32)
    start = 0
    for windows, _labels in tqdm(dl, desc="  Embedding extraction"):
        B = windows.shape[0]
        W = windows.shape[1]
        w_sum = torch.zeros(B, out_dim, device=device)
        for w_i in range(W):
            hidden, _ = encoder(windows[:, w_i].to(device))  # (B, 1, 192)
            w_sum += hidden.squeeze(1)
        embs[start:start + B] = (w_sum / W).cpu().numpy()
        start += B
    return embs


# ────────────────────────────────────────────────────────────────
#  3. Similarity analysis
# ────────────────────────────────────────────────────────────────

def cosine_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return A @ B.T


def cosine_pairwise(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Element-wise cosine similarity between corresponding rows."""
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return (A * B).sum(axis=1)


def sample_similarities(embs: np.ndarray, labels: np.ndarray, df: pd.DataFrame) -> dict:
    """Sample same-speaker and cross-speaker cosine similarities (known files)."""
    rng = np.random.default_rng(RANDOM_SEED)
    known_mask = labels > 0
    idx = np.where(known_mask)[0]
    spk = labels[idx]

    same_sims = []
    for sid in np.unique(spk):
        ii = idx[spk == sid]
        if len(ii) < 2:
            continue
        pairs = np.array([(ii[i], ii[j]) for i in range(len(ii)) for j in range(i + 1, len(ii))])
        if len(pairs) > 200:
            pairs = pairs[rng.choice(len(pairs), 200, replace=False)]
        same_sims.extend(cosine_pairwise(embs[pairs[:, 0]], embs[pairs[:, 1]]).tolist())
        if len(same_sims) >= MAX_SAME_PAIRS:
            break
    same_sims = np.array(same_sims[:MAX_SAME_PAIRS])

    n_pairs = min(MAX_CROSS_PAIRS, len(idx) * (len(idx) - 1) // 2)
    cross_sims = []
    while len(cross_sims) < n_pairs:
        a = rng.integers(0, len(idx), size=min(n_pairs, 100_000))
        b = rng.integers(0, len(idx), size=min(n_pairs, 100_000))
        valid = spk[a] != spk[b]
        if valid.any():
            aa, bb = idx[a[valid]], idx[b[valid]]
            cross_sims.extend(cosine_pairwise(embs[aa], embs[bb]).tolist())
    cross_sims = np.array(cross_sims[:n_pairs])

    return {"same": same_sims, "cross": cross_sims}


# ────────────────────────────────────────────────────────────────
#  4. Unbiased (LOO) centroid evaluation + OOD detection
# ────────────────────────────────────────────────────────────────

def _l2norm_rows(M: np.ndarray) -> np.ndarray:
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)


def unbiased_centroid_eval(embs: np.ndarray, labels: np.ndarray) -> dict:
    """
    Leave-one-out centroid evaluation — the honest recognition ceiling.

    - Known files: centroid of their speaker built WITHOUT that file (LOO).
    - Unknown files: scored against full known-speaker centroids.
    - OOD score = 1 − max cosine similarity to (LOO) known centroids.

    Returns dict with LOO known accuracy, top-5, OOD AUC and ROC arrays.
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    known_mask = labels > 0
    known_embs = embs[known_mask]
    known_ids = labels[known_mask]
    speakers = np.unique(known_ids)
    speaker_to_idx = {int(s): i for i, s in enumerate(speakers)}

    # Full centroids per known speaker (L2-normalised)
    centroids = {}
    for sid in speakers:
        centroids[int(sid)] = known_embs[known_ids == sid].mean(axis=0)
    centroids_np = np.stack([centroids[s] for s in speakers])  # (S, D)
    CN = _l2norm_rows(centroids_np)

    # LOO similarity matrix for known files
    n_known = len(known_embs)
    loo_sims = np.zeros((n_known, len(speakers)), dtype=np.float32)
    norm_known = _l2norm_rows(known_embs)
    for i, sid in enumerate(known_ids):
        loo_sims[i] = norm_known[i] @ CN.T
        # Replace this file's own centroid row with the LOO centroid
        c_full = centroids[int(sid)]
        n_sid = int((known_ids == sid).sum())
        if n_sid > 1:
            c_loo = (c_full * n_sid - known_embs[i]) / (n_sid - 1)
            c_loo = c_loo / (np.linalg.norm(c_loo) + 1e-12)
            loo_sims[i, speaker_to_idx[int(sid)]] = float(norm_known[i] @ c_loo)
        else:
            loo_sims[i, speaker_to_idx[int(sid)]] = -1.0  # no LOO centroid possible

    pred_speaker_idx = loo_sims.argmax(axis=1)
    true_idx = np.array([speaker_to_idx[int(s)] for s in known_ids])
    known_acc = float((pred_speaker_idx == true_idx).mean())
    topk = np.argsort(-loo_sims, axis=1)[:, :5]
    known_top5 = float((topk == true_idx[:, None]).any(axis=1).mean())

    # OOD score (unbiased): 1 − max LOO sim for known; 1 − max full-centroid sim for unknown
    ood_scores = np.zeros(len(embs), dtype=np.float64)
    ood_scores[known_mask] = 1.0 - loo_sims.max(axis=1)
    if (~known_mask).any():
        unk_sims = _l2norm_rows(embs[~known_mask]) @ CN.T
        ood_scores[~known_mask] = 1.0 - unk_sims.max(axis=1)

    y_true = (~known_mask).astype(int)
    auc = float(roc_auc_score(y_true, ood_scores))
    fpr, tpr, thr = roc_curve(y_true, ood_scores)
    youden = tpr - fpr
    best_i = int(np.argmax(youden))

    return {
        "known_acc_loo": known_acc,
        "known_top5_loo": known_top5,
        "ood_auc": auc,
        "best_threshold_youden": float(thr[best_i]),
        "best_tpr_youden": float(tpr[best_i]),
        "best_fpr_youden": float(fpr[best_i]),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "ood_scores": ood_scores,
        "y_true_ood": y_true,
    }


# ────────────────────────────────────────────────────────────────
#  5. Unknown internal structure (KMeans compactness)
# ────────────────────────────────────────────────────────────────

def unknown_structure_analysis(embs: np.ndarray, labels: np.ndarray, n_clusters: int = 8) -> dict:
    """KMeans on unknown embeddings → compactness (pseudo-labelling feasibility)."""
    from sklearn.cluster import KMeans

    unk_mask = labels == 0
    if unk_mask.sum() < n_clusters:
        return {"n_unknown_embedded": int(unk_mask.sum()), "note": "too few unknown files"}
    unk_embs = _l2norm_rows(embs[unk_mask])
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10).fit(unk_embs)
    centers = _l2norm_rows(km.cluster_centers_)
    sims = unk_embs @ centers.T  # (N, K)
    assigned = sims.max(axis=1)
    return {
        "n_unknown_embedded": int(unk_mask.sum()),
        "kmeans_n_clusters": n_clusters,
        "kmeans_inertia": float(km.inertia_),
        "mean_cos_to_cluster_centroid": float(assigned.mean()),
        "median_cos_to_cluster_centroid": float(np.median(assigned)),
        "frac_cos_gt_0_5": float((assigned > 0.5).mean()),
    }


# ────────────────────────────────────────────────────────────────
#  6. Macro-F1 simulation (the number that matters)
# ────────────────────────────────────────────────────────────────

def estimate_macro_f1(
    known_acc: float,
    tpr: float,
    fpr: float,
    unknown_prior: float = 0.5,
    num_known: int = 446,
) -> dict:
    """
    Analytic Macro-F1 estimate at a given OOD operating point.

    Assumes ~50/50 eval mix, each known class has balanced precision/recall
    ≈ known_acc, and the unknown class has recall=tpr / precision derived from
    the FPR.

        F1_unknown = 2·P·R / (P+R),  P = tpr·π / (tpr·π + fpr·(1−π))
        Macro-F1 ≈ (F1_unknown + num_known·known_acc) / (num_known + 1)
    """
    precision_u = (tpr * unknown_prior) / (
        tpr * unknown_prior + fpr * (1 - unknown_prior) + 1e-12)
    f1_unknown = 2 * precision_u * tpr / (precision_u + tpr + 1e-12)
    macro = (f1_unknown + num_known * known_acc) / (num_known + 1)
    return {
        "known_acc": float(known_acc),
        "ood_tpr": float(tpr),
        "ood_fpr": float(fpr),
        "unknown_precision": float(precision_u),
        "unknown_f1": float(f1_unknown),
        "estimated_macro_f1": float(macro),
    }


def macro_f1_simulation(
    embs: np.ndarray,
    labels: np.ndarray,
    ood_scores: np.ndarray | None = None,
    num_classes: int = 447,
) -> dict:
    """
    Simulate the competition metric directly on the labelled set with the
    centroid classifier + OOD threshold:

        ood_score > thr  → predict class 0 (unknown)
        otherwise        → predict argmax speaker (LOO for known / full for unknown)

    Sweeps thresholds and reports the best Macro-F1 (plain competition argmax
    semantics with a hard unknown gate, purely for local operating-point study).

    If `ood_scores` is None, the LOO-based OOD scores from
    `unbiased_centroid_eval` are computed internally.
    """
    from src.metrics import macro_f1_score

    if ood_scores is None:
        ood_scores = unbiased_centroid_eval(embs, labels)["ood_scores"]

    known_mask = labels > 0
    known_embs = embs[known_mask]
    known_ids = labels[known_mask]
    speakers = np.unique(known_ids)
    speaker_to_idx = {int(s): i for i, s in enumerate(speakers)}

    centroids = {}
    for sid in speakers:
        centroids[int(sid)] = known_embs[known_ids == sid].mean(axis=0)
    centroids_np = np.stack([centroids[s] for s in speakers])
    CN = _l2norm_rows(centroids_np)

    # predicted speaker (argmax over centroids)
    norm_all = _l2norm_rows(embs)
    all_sims = norm_all @ CN.T
    pred_speaker_idx = all_sims.argmax(axis=1)
    pred_speaker_global = np.array([int(speakers[i]) for i in pred_speaker_idx])

    best = {"threshold": 0.5, "macro_f1": 0.0}
    table = []
    for thr in np.arange(0.02, 0.98, 0.02):
        preds = np.where(ood_scores > thr, 0, pred_speaker_global)
        mf1 = macro_f1_score(labels, preds, num_classes=num_classes)
        table.append({"threshold": round(float(thr), 3), "macro_f1": round(float(mf1), 6)})
        if mf1 > best["macro_f1"]:
            best = {"threshold": float(thr), "macro_f1": float(mf1)}

    # argmax-only (no OOD gate) — the pure competition prediction
    pure_argmax_mf1 = macro_f1_score(labels, pred_speaker_global, num_classes=num_classes)
    return {
        "best_threshold_macro_f1": best["threshold"],
        "best_macro_f1": best["macro_f1"],
        "pure_argmax_macro_f1": float(pure_argmax_mf1),
        "threshold_table": table,
    }


# ────────────────────────────────────────────────────────────────
#  7. t-SNE
# ────────────────────────────────────────────────────────────────

def tsne_projection(embs: np.ndarray, df: pd.DataFrame, n: int = TSNE_SAMPLE) -> tuple:
    from sklearn.manifold import TSNE
    rng = np.random.default_rng(RANDOM_SEED)
    sel = rng.choice(len(embs), min(n, len(embs)), replace=False)
    X = embs[sel]
    proj = TSNE(n_components=2, perplexity=40, random_state=RANDOM_SEED,
                init="pca", learning_rate="auto", n_jobs=-1).fit_transform(X)
    return proj, df.iloc[sel].copy().reset_index(drop=True)


def plot_tsne_known_unknown(proj: np.ndarray, sub_df: pd.DataFrame, save_path: Path):
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = sub_df["is_unknown"].map({True: "#e74c3c", False: "#2ecc71"})
    ax.scatter(proj[:, 0], proj[:, 1], c=colors, s=10, alpha=0.55)
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=l)
               for c, l in [("#2ecc71", "Known (446 speakers)"), ("#e74c3c", "Unknown (OOD)")]]
    ax.legend(handles=handles, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("t-SNE of ECAPA Embeddings — Known vs Unknown (cleaned data)",
                 fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_tsne_top_speakers(proj: np.ndarray, sub_df: pd.DataFrame, save_path: Path):
    top = sub_df[~sub_df["is_unknown"]]["speaker_id"].value_counts().head(TOP_SPEAKERS_TSNE).index
    palette = sns.color_palette("husl", len(top))
    spk_to_color = {s: c for s, c in zip(top, palette)}
    others = sub_df[~sub_df["speaker_id"].isin(top)]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(proj[others.index, 0], proj[others.index, 1], c="#cccccc", s=6, alpha=0.35,
               label="other known")
    for sid in top:
        m = sub_df["speaker_id"] == sid
        ax.scatter(proj[m.index, 0], proj[m.index, 1], c=[spk_to_color[sid]], s=18,
                   alpha=0.8, label=f"{sid[:8]}…", edgecolors="white", linewidth=0.3)
    ax.legend(fontsize=8, ncol=2, markerscale=1.2)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"t-SNE of ECAPA Embeddings — Top {TOP_SPEAKERS_TSNE} Known Speakers",
                 fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ────────────────────────────────────────────────────────────────
#  8. Report
# ────────────────────────────────────────────────────────────────

def generate_report(stats: dict, sim: dict, cent: dict, sim_macro: dict,
                    est_macro: dict, unknown_struct: dict) -> str:
    same, cross = sim["same"], sim["cross"]
    lo = max(same.min(), cross.min())
    hi = min(same.max(), cross.max())
    grid = np.linspace(lo, hi, 200)

    def rate(vals):
        return np.array([(vals >= t).mean() for t in grid])

    tar = rate(same)
    far = rate(cross)
    eer_i = int(np.argmin(np.abs(tar - (1 - far))))
    eer = grid[eer_i]
    dprime = (np.mean(same) - np.mean(cross)) / np.sqrt(0.5 * (np.var(same) + np.var(cross)))

    return f"""# Phase 3 (Unbiased) — Embedding-Space & OOD Separability EDA Report

**Project:** IAAA Competition 2026 — Open-Set Speaker Identification
**Module:** `src/eda_embeddings.py` · **Date:** 2026-08-08
**Label:** all numbers below are **out-of-sample (LOO) / unbiased**

---

## 1. Setup & Data Cleaning

- Encoder: **ECAPA-TDNN** (SpeechBrain `spkrec-ecapa-voxceleb`, frozen) — 192-d embeddings.
- Input: **{DURATION_SECONDS:.0f}s sliding windows, {EVAL_HOP_RATIO:.0%} overlap, ≤ {MAX_EVAL_WINDOWS} per file** (multi-window TTA, matching the pipeline).
- Files embedded: **{stats['n_files_after_clean']:,}** after dropping:
  - **{stats['n_corrupted_dropped']} corrupted** (< {MIN_VALID_DURATION:.0f}s) — {stats['n_corrupted_known']} known + {stats['n_corrupted_unknown']} unknown
  - **{stats['n_duplicate_files']} MD5-duplicate** files ({stats['n_duplicate_groups']} groups)
- Device: {stats['device']}

---

## 2. Verification Calibration (same vs cross speaker, cleaned data)

| Metric | Value |
|--------|------:|
| Same-speaker pairs sampled | {len(sim['same']):,} |
| Cross-speaker pairs sampled | {len(sim['cross']):,} |
| Mean cosine — same speaker | {np.mean(sim['same']):.4f} |
| Mean cosine — cross speaker | {np.mean(sim['cross']):.4f} |
| **d′ (separability)** | **{dprime:.2f}** |
| **EER (cosine verification)** | **{eer:.3f}** |

![Similarity Histograms](phase3_similarity_histograms.png)

---

## 3. Recognition Ceiling — **Leave-One-Out** (unbiased)

Centroid of a speaker is built **without** the file being scored (LOO), so a
file never contributes to its own centroid (~20% self-contribution removed).

| Metric | Value |
|--------|------:|
| Known speakers (centroids) | {stats['n_known_speakers']} |
| **Argmax-centroid accuracy (LOO, known files)** | **{cent['known_acc_loo']*100:.2f}%** |
| Top-5 centroid accuracy (LOO) | {cent['known_top5_loo']*100:.2f}% |

> The original Phase-3 reported **~95.5% in-sample**; the honest LOO ceiling is
> **{cent['known_acc_loo']*100:.1f}%**. The learned heads / fine-tuning must push
> beyond this to reach the 0.97 Macro-F1 target.

---

## 4. OOD Feasibility (nearest-known-centroid distance, unbiased)

OOD score = `1 − max cosine similarity to known-speaker centroids` (LOO for known files).

| Metric | Value |
|--------|------:|
| **OOD detection AUC** | **{cent['ood_auc']:.4f}** |
| Best operating threshold (Youden) | {cent['best_threshold_youden']:.3f} |
| TPR at threshold | {cent['best_tpr_youden']:.3f} |
| FPR at threshold | {cent['best_fpr_youden']:.3f} |

![OOD Score Histograms](phase3_ood_score_histograms.png)

![ROC Curve](phase3_roc_curve.png)

---

## 5. Unknown Internal Structure (KMeans, pseudo-labelling feasibility)

| Metric | Value |
|--------|------:|
| Unknown files embedded | {unknown_struct.get('n_unknown_embedded', 0):,} |
| KMeans clusters | {unknown_struct.get('kmeans_n_clusters', '-')} |
| Mean cosine to cluster centroid | {unknown_struct.get('mean_cos_to_cluster_centroid', '-')} |
| Fraction with cos > 0.5 | {unknown_struct.get('frac_cos_gt_0_5', '-')} |

---

## 6. Macro-F1 Simulation (**the number that matters**)

Centroid classifier + OOD threshold gate on the full labelled set
(competition semantics: 447-class macro-F1).

| Metric | Value |
|--------|------:|
| **Macro-F1 (pure argmax, no OOD gate)** | **{sim_macro['pure_argmax_macro_f1']:.4f}** |
| **Macro-F1 (best OOD threshold)** | **{sim_macro['best_macro_f1']:.4f}** (thr={sim_macro['best_threshold_macro_f1']:.3f}) |

Analytic estimate at the Youden operating point (assumes ~50/50 eval mix,
balanced per-class known precision/recall ≈ known LOO accuracy):

| Metric | Value |
|--------|------:|
| Estimated known-class F1 | {est_macro['known_acc']:.4f} |
| Unknown-class recall (TPR) | {est_macro['ood_tpr']:.3f} |
| Unknown-class FPR | {est_macro['ood_fpr']:.3f} |
| **Estimated Macro-F1** | **{est_macro['estimated_macro_f1']:.4f}** |

> This is the only number that decides whether the centroid route alone can hit
> **0.97**. If it is well below 0.97, fine-tuning the encoder (Step 7) and/or an
> ensemble (Step 9) is required.

---

## 7. Embedding Geometry

### 7.1 Known vs unknown (cleaned data)

![t-SNE known vs unknown](phase3_tsne_known_unknown.png)

### 7.2 Top known speakers

![t-SNE top speakers](phase3_tsne_top_speakers.png)

---

## 8. Key Numbers (JSON)

```json
{json.dumps({k: v for k, v in {**stats, **cent, **sim_macro, **est_macro, **unknown_struct}.items()
             if not isinstance(v, (np.ndarray, list)) and k not in ("fpr", "tpr")}, indent=2, default=str)[:1800]}
```

---

*Report generated programmatically via `src/eda_embeddings.py` (unbiased / LOO).*
"""


# ────────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Phase 3 (Unbiased) — Embedding-Space & OOD Separability EDA")
    print("=" * 60)

    print("\n[1/7] Loading labels & dropping corrupted/duplicate files...")
    df = load_labels(LABELS_PATH)
    clean_df, stats = clean_labels(df, WAV_AUDIO_DIR)
    print(f"  Raw: {stats['n_raw_files']:,} → Clean: {stats['n_files_after_clean']:,} "
          f"(corrupted={stats['n_corrupted_dropped']}, dup files={stats['n_duplicate_files']})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    stats["device"] = str(device)

    print("\n[2/7] Extracting multi-window ECAPA embeddings...")
    embs = extract_embeddings(clean_df, WAV_AUDIO_DIR, device)
    known_ids = clean_df[~clean_df["is_unknown"]]["speaker_id"].unique()
    known_lookup = {s: i for i, s in enumerate(known_ids)}
    labels = clean_df["speaker_id"].map(lambda s: known_lookup.get(s, -1)).values.astype(int)
    labels = labels + 1              # known → 1..446
    labels[clean_df["is_unknown"].values] = 0  # unknown → 0
    stats["n_known_speakers"] = len(known_ids)
    np.save(EDA_DIR / "phase3_embeddings.npy", embs)
    print(f"  Embeddings: {embs.shape} saved (cleaned set)")

    print("\n[3/7] Similarity distributions (cleaned)...")
    sim = sample_similarities(embs, labels, clean_df)
    print(f"  Same: n={len(sim['same']):,} μ={np.mean(sim['same']):.4f} | "
          f"Cross: n={len(sim['cross']):,} μ={np.mean(sim['cross']):.4f}")

    print("\n[4/7] Unbiased (LOO) centroid analysis...")
    cent = unbiased_centroid_eval(embs, labels)
    print(f"  LOO argmax-centroid acc (known): {cent['known_acc_loo']*100:.2f}%")
    print(f"  OOD AUC (LOO centroid distance): {cent['ood_auc']:.4f}")

    print("\n[5/7] Unknown internal structure (KMeans)...")
    unknown_struct = unknown_structure_analysis(embs, labels)
    print(f"  Mean cos to cluster centroid: {unknown_struct.get('mean_cos_to_cluster_centroid', '-')}")

    print("\n[6/7] Macro-F1 simulation...")
    sim_macro = macro_f1_simulation(embs, labels, cent["ood_scores"])
    print(f"  Pure argmax Macro-F1: {sim_macro['pure_argmax_macro_f1']:.4f} | "
          f"Best-threshold Macro-F1: {sim_macro['best_macro_f1']:.4f} "
          f"(thr={sim_macro['best_threshold_macro_f1']:.3f})")
    est_macro = estimate_macro_f1(
        cent["known_acc_loo"], cent["best_tpr_youden"], cent["best_fpr_youden"],
    )
    print(f"  Analytic Macro-F1 estimate: {est_macro['estimated_macro_f1']:.4f}")

    print("\n[7/7] t-SNE, plots & report...")
    sns.set_theme(style="whitegrid", font_scale=1.05)

    proj, sub_df = tsne_projection(embs, clean_df)
    plot_tsne_known_unknown(proj, sub_df, PLOT_T1)
    plot_tsne_top_speakers(proj, sub_df, PLOT_T2)
    print("  [SAVED] 2 t-SNE charts")

    # similarity histograms
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(sim["same"], ax=ax, label="Same speaker", color="#2ecc71", fill=True, alpha=0.35)
    sns.kdeplot(sim["cross"], ax=ax, label="Cross speaker", color="#e74c3c", fill=True, alpha=0.35)
    ax.axvline(np.mean(sim["same"]), color="#2ecc71", linestyle="--", linewidth=1.2)
    ax.axvline(np.mean(sim["cross"]), color="#e74c3c", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Cosine similarity of ECAPA embeddings", fontsize=13, weight="bold")
    ax.set_ylabel("Density", fontsize=13, weight="bold")
    ax.set_title("Verification Calibration — Same vs Cross Speaker (cleaned)", fontsize=15, weight="bold")
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(PLOT_SIM, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # OOD score histograms
    y = cent["y_true_ood"]
    scores = cent["ood_scores"]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(scores[y == 0], ax=ax, label="Known files", color="#2ecc71", fill=True, alpha=0.35)
    sns.kdeplot(scores[y == 1], ax=ax, label="Unknown files", color="#e74c3c", fill=True, alpha=0.35)
    ax.axvline(cent["best_threshold_youden"], color="#2c3e50", linestyle="--", linewidth=1.5,
               label=f"Youden threshold = {cent['best_threshold_youden']:.3f}")
    ax.set_xlabel("OOD score (1 − max centroid cosine)", fontsize=13, weight="bold")
    ax.set_ylabel("Density", fontsize=13, weight="bold")
    ax.set_title("Centroid-Distance OOD Score — Known vs Unknown (LOO)", fontsize=15, weight="bold")
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOT_OOD, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ROC
    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.plot(cent["fpr"], cent["tpr"], color="#8e44ad", linewidth=2.5,
            label=f"AUC = {cent['ood_auc']:.4f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False positive rate (unknown flagged as known)", fontsize=12, weight="bold")
    ax.set_ylabel("True positive rate (unknown detected)", fontsize=12, weight="bold")
    ax.set_title("OOD Detection ROC — Centroid Distance (LOO)", fontsize=14, weight="bold")
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_ROC, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    summary = {
        **stats,
        **{k: v for k, v in cent.items() if k not in ("fpr", "tpr", "ood_scores", "y_true_ood")},
        **sim_macro,
        **est_macro,
        **unknown_struct,
    }
    summary.pop("threshold_table", None)
    JSON_OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    REPORT.write_text(generate_report(stats, sim, cent, sim_macro, est_macro, unknown_struct), encoding="utf-8")
    print(f"  [SAVED] {JSON_OUT.name} | {REPORT.name}")

    print("\n✅ Phase 3 (unbiased) EDA complete.")


if __name__ == "__main__":
    main()

