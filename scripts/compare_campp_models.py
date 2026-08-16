"""
Side-by-side comparison of two CAM++ checkpoints on the SAME val split.

The old submission (campp_best.pt, single-model, LB f1_macro 0.9505) was tuned
on the "single" leak-free split (val_per_known=1, unknown_val_ratio=0.2,
seed 42): 891 files covering all 446 speakers + 445 unknown.  The new
checkpoint (campp_best (2).pt) embeds a kfold split in its config, so its own
checkpoint macro-F1 is NOT comparable to the old one — this script evaluates
both models with the EXACT submission decision layer (predict_proba_and_embed
+ cosine centroids + alpha/kappa/tau/lambda gate) on the common reference
split, and reports head-only + decision-layer Macro-F1 for each.

Usage:
    uv run --no-sync python scripts/compare_campp_models.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.cli_utils import setup_utf8_stdio  # noqa: E402
setup_utf8_stdio()

from src.data_pipeline import (  # noqa: E402
    prepare_clean_split, SpeakerDataset,
)
from src.model_factory import create_model_from_config  # noqa: E402
from src.metrics import macro_f1_score  # noqa: E402
from submission.inference import centroid_probs_matrix  # noqa: E402

DATA = ROOT / "data" / "processed"
LABELS_PATH = DATA / "audio_wav_labels.csv"
AUDIO_DIR = DATA / "audio_wav"
PROCESSED_LABELS = DATA / "cleaned_labels.csv"

NUM_CLASSES = 447

OLD_CKPT = ROOT / "checkpoints" / "campp_best.pt"
NEW_CKPT = ROOT / "checkpoints" / "campp_best (2).pt"

# Decision params of the CURRENT leaderboard submission (α=0.2, κ=24, τ=0, λ=1.05)
SUBMITTED_DECISION = {"alpha": 0.2, "kappa": 24.0, "tau": 0.0,
                      "lambda_unknown": 1.05, "temperature": 1.0}


# ────────────────────────────────────────────────────────────
# 1. Common leak-free reference split (single scheme — matches old tuning)
# ────────────────────────────────────────────────────────────
def get_reference_split():
    train_df, val_df, class_map = prepare_clean_split(
        labels_path=str(LABELS_PATH),
        audio_dir=str(AUDIO_DIR),
        processed_labels=str(PROCESSED_LABELS),
        val_per_known=1,
        unknown_val_ratio=0.2,
        random_seed=42,
    )
    return train_df, val_df, class_map


# ────────────────────────────────────────────────────────────
# 2. Model loading (per-checkpoint embedded config)
# ────────────────────────────────────────────────────────────
def load_model(ckpt_path: Path, device: torch.device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ck["config"]
    class_map = ck["class_map"]
    num_known = config.get("model", {}).get(
        "competition_num_known", len(class_map) - 1)
    model = create_model_from_config(config, num_known_speakers=num_known)
    model.load_state_dict(ck["model_state_dict"])
    model.to(device).eval()
    return model, config


# ────────────────────────────────────────────────────────────
# 3. Val probs + embeddings (same forward as submission)
# ────────────────────────────────────────────────────────────
@torch.inference_mode()
def dump_val(model, val_df, config, device) -> tuple:
    audio_cfg = config["audio"]
    ds = SpeakerDataset(
        val_df, str(AUDIO_DIR), sample_rate=audio_cfg["sample_rate"],
        duration_seconds=audio_cfg["duration_seconds"], augment=False,
        num_train_windows=audio_cfg.get("num_train_windows", 1),
        eval_hop_ratio=audio_cfg.get("eval_hop_ratio", 0.5),
        max_eval_windows=audio_cfg.get("max_eval_windows", 8),
    )
    emb_dim = getattr(model.head_speaker, "embedding_dim", 192)
    probs = np.zeros((len(ds), NUM_CLASSES), dtype=np.float32)
    embs = np.zeros((len(ds), emb_dim), dtype=np.float32)
    labels = np.zeros((len(ds),), dtype=np.int64)
    for i in tqdm(range(len(ds)), desc="  val forward", leave=False):
        windows, lab = ds[i]
        p, e = model.predict_proba_and_embed(windows.to(device), temperature=1.0)
        probs[i] = p.cpu().numpy()
        embs[i] = e.cpu().numpy()
        labels[i] = int(lab.item())
    return probs, embs, labels


# ────────────────────────────────────────────────────────────
# 4. Centroids on the reference train split (same as submission)
# ────────────────────────────────────────────────────────────
@torch.inference_mode()
def build_centroids(model, train_df, config, device,
                    checkpoint_class_map: dict) -> tuple:
    """Centroids for the reference train split, rows aligned to the
    checkpoint's class_map (global id 1..num_known)."""
    from torch.utils.data import DataLoader
    audio_cfg = config["audio"]
    num_known = config.get("model", {}).get("competition_num_known", 446)

    # Align train labels to the checkpoint's class_map (speaker_id -> global id)
    train_df = train_df.copy()
    train_df["label"] = train_df["speaker_id"].map(checkpoint_class_map).astype(int)
    # Only known speakers contribute to centroids
    train_df = train_df[train_df["label"] > 0].reset_index(drop=True)

    ds = SpeakerDataset(
        train_df, str(AUDIO_DIR), sample_rate=audio_cfg["sample_rate"],
        duration_seconds=audio_cfg["duration_seconds"], augment=False,
        num_train_windows=audio_cfg.get("num_train_windows", 1),
        eval_hop_ratio=audio_cfg.get("eval_hop_ratio", 0.5),
        max_eval_windows=audio_cfg.get("max_eval_windows", 8),
    )
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    embs, labels = [], []
    for windows, lab in tqdm(dl, desc="  train embeddings", leave=False):
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
    for sid in speakers:
        m = known_ids == sid
        if m.sum() == 0:
            raise RuntimeError(f"No train embeddings for speaker {sid}")
        centroids[sid - 1] = known_embs[m].mean(axis=0)
    centroids = centroids / (np.linalg.norm(centroids, axis=1,
                                            keepdims=True) + 1e-12)
    return centroids.astype(np.float32), speakers.astype(np.int64), int(len(known_embs))


# ────────────────────────────────────────────────────────────
# 5. Decision-layer evaluation
# ────────────────────────────────────────────────────────────
def evaluate_decision(probs, emb, centroids, speaker_ids, labels, params):
    """Exact math of submission.inference.score_ensemble decision path."""
    alpha = float(params["alpha"])
    kappa = float(params["kappa"])
    tau = float(params["tau"])
    lambda_unknown = float(params["lambda_unknown"])

    cp, mc = centroid_probs_matrix(emb, centroids, speaker_ids,
                                   NUM_CLASSES, kappa)
    fused = alpha * probs + (1.0 - alpha) * cp
    fused = fused.copy()
    fused[:, 0] *= lambda_unknown
    fused /= (fused.sum(axis=1, keepdims=True) + 1e-12)
    pred = fused.argmax(axis=1).astype(np.int64)
    pred[mc < tau] = 0
    return macro_f1_score(labels, pred, num_classes=NUM_CLASSES), pred


def tune_decision(probs, emb, centroids, speaker_ids, labels):
    """Coordinate-descent sweep (mirrors src.decision_engine.tune_decision_bundle)."""
    grids = {
        "alpha": np.round(np.arange(0.0, 1.001, 0.05), 2),
        "kappa": np.array([0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0]),
        "tau": np.round(np.arange(0.0, 0.60, 0.02), 3),
        "lambda_unknown": np.round(np.arange(0.5, 1.601, 0.05), 2),
    }
    order = ["alpha", "kappa", "tau", "lambda_unknown"]
    best = {"alpha": 1.0, "kappa": 8.0, "tau": 0.0, "lambda_unknown": 1.0}
    best_score = evaluate_decision(probs, emb, centroids, speaker_ids,
                                   labels, best)[0]
    baseline = best_score
    for rnd in range(3):
        for key in order:
            improved = False
            for v in grids[key]:
                cand = dict(best)
                cand[key] = float(v)
                score = evaluate_decision(probs, emb, centroids, speaker_ids,
                                          labels, cand)[0]
                if score > best_score:
                    best_score = score
                    best = cand
                    improved = True
            if improved:
                print(f"    [round {rnd + 1}] {key} -> {best[key]} "
                      f"(Macro-F1 {best_score:.4f})")
    return best, best_score, baseline


def head_only_macro_f1(probs, labels):
    pred = probs.argmax(axis=1).astype(np.int64)
    return macro_f1_score(labels, pred, num_classes=NUM_CLASSES)


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 72)
    print("  CAM++ model comparison on the reference (single) val split")
    print("=" * 72)
    print(f"  Device: {device}")

    train_df, val_df, class_map = get_reference_split()
    val_labels = val_df["label"].values.astype(np.int64)
    print(f"  Val files: {len(val_df):,} "
          f"(known={int((val_labels > 0).sum())}, "
          f"unknown={int((val_labels == 0).sum())})")
    print(f"  Train files: {len(train_df):,} "
          f"(known={int((train_df['label'] > 0).sum())})")

    # Class-map sanity: the reference split's mapping must equal the
    # checkpoints' embedded mapping (both derive from the same labels file).
    for name, ckpt_path in [("old", OLD_CKPT), ("new", NEW_CKPT)]:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        embedded = ck["class_map"]
        assert embedded == class_map, (
            f"[{name}] embedded class_map differs from the reference split's! "
            f"Check label data drift."
        )
    print("  ✓ Checkpoint class_maps match the reference split.")

    results = {}
    for name, ckpt_path in [("old", OLD_CKPT), ("new", NEW_CKPT)]:
        print(f"\n{'─' * 72}\n  [{name}] {ckpt_path.name}\n{'─' * 72}")
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model, config = load_model(ckpt_path, device)
        checkpoint_class_map = ck["class_map"]  # speaker_id (uuid/unknown) -> global id

        # val dump
        probs, embs, labels = dump_val(model, val_df, config, device)
        assert (labels == val_labels).all(), \
            f"[{name}] val order/labels mismatch with reference split!"
        head_f1 = head_only_macro_f1(probs, val_labels)

        # centroids
        centroids, speaker_ids, n_train = build_centroids(
            model, train_df, config, device, checkpoint_class_map)

        # head-only
        print(f"  Head-only Macro-F1:           {head_f1:.4f}")

        # tuned decision
        print("  Tuning decision params...")
        best, best_score, baseline = tune_decision(
            probs, embs, centroids, speaker_ids, val_labels)
        print(f"  Baseline (plain head argmax): {baseline:.4f}")
        print(f"  Best decision Macro-F1:       {best_score:.4f}  params={best}")

        # submitted decision params applied
        sub_f1, sub_pred = evaluate_decision(
            probs, embs, centroids, speaker_ids, val_labels, SUBMITTED_DECISION)
        print(f"  Submitted params (α=.2 κ=24 τ=0 λ=1.05): {sub_f1:.4f}")

        # unknown-specific stats under submitted params
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(val_labels, sub_pred, labels=list(range(NUM_CLASSES)))
        tp_u = cm[0, 0]
        n_u = int((val_labels == 0).sum())
        print(f"  Unknown recall (submitted params): {tp_u}/{n_u} = {tp_u / max(n_u, 1):.3f}")

        results[name] = {
            "checkpoint": str(ckpt_path),
            "head_only_macro_f1": head_f1,
            "baseline_macro_f1": baseline,
            "best_macro_f1": best_score,
            "best_params": best,
            "submitted_params_macro_f1": sub_f1,
            "n_train_files": n_train,
        }

    print(f"\n{'=' * 72}\n  SUMMARY\n{'=' * 72}")
    for name, r in results.items():
        print(f"  [{name}] head-only {r['head_only_macro_f1']:.4f} | "
              f"baseline {r['baseline_macro_f1']:.4f} | "
              f"best-decision {r['best_macro_f1']:.4f} | "
              f"submitted-params {r['submitted_params_macro_f1']:.4f}")

    out = DATA / "campp_model_comparison.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\n  ✓ Saved to {out}")


if __name__ == "__main__":
    main()
