"""
Shared decision-layer engine (Audit §17.3).

Houses the reusable pieces that the Phase-1 CLI scripts (``dump_val_artifacts``,
``tune_decision``) and the new pipeline steps (``src/pipelines/steps.py``) both
need. Keeping them in ``src/`` (a proper package) rather than ``scripts/`` avoids
a namespace-package collision: a third-party ``scripts`` package ships in some
site-packages, which silently shadows a ``scripts/`` dir that has no
``__init__.py``.

Functions:
    dump_val_checkpoint(ckpt, device)      → writes val_probs/val_emb/val_labels
    load_decision_artifacts(data_dir=None) → dict consumed by the tuner
    tune_decision_bundle(artifacts)        → decision bundle dict (α/κ/τ/λ)

The centroid builder stays in ``src/centroid_baseline.build_checkpoint_centroids``
(single source of truth); callers save the returned matrix to ``.npz`` themselves.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data" / "processed"
FUSION_JSON = DATA / "ensemble_fusion_weights.json"
NUM_CLASSES = 447


def encoder_key(checkpoint_path: str) -> str:
    """'checkpoints/campp_best.pt' → 'campp'."""
    return Path(checkpoint_path).name.replace("_best.pt", "")


def dump_val_checkpoint(checkpoint_path: str, device: torch.device) -> dict:
    """Dump inference-consistent val probs + ArcFace embeddings for one checkpoint.

    Reuses the EXACT forward path the submission uses
    (``model.predict_proba_and_embed`` — prob-averaged head probs + mean-then-L2
    normalised embedding). Respects the checkpoint's embedded ``data.split``
    (scheme/fold/seed) so a kfold-trained checkpoint validates on ITS fold.

    Writes ``data/processed/val_probs_<enc>.npy``, ``val_emb_<enc>.npy`` and
    ``val_labels.npy``.
    """
    from src.data_pipeline import (
        prepare_clean_split, SpeakerDataset, split_args_from_config,
    )
    from src.model_factory import create_model_from_config

    key = encoder_key(checkpoint_path)
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
    )

    ds = SpeakerDataset(
        val_df, data_cfg["audio_dir"], sample_rate=audio_cfg["sample_rate"],
        duration_seconds=audio_cfg["duration_seconds"], augment=False,
        num_train_windows=audio_cfg.get("num_train_windows", 1),
        eval_hop_ratio=audio_cfg.get("eval_hop_ratio", 0.5),
        max_eval_windows=audio_cfg.get("max_eval_windows", 8),
    )

    # Competition output width (predict_proba_and_embed already collapses any
    # pseudo-identity cluster columns into unknown).
    num_unknown_clusters = int(model.num_unknown_clusters)
    num_classes = len(class_map) - num_unknown_clusters
    emb_dim = getattr(model.head_speaker, "embedding_dim", 192)
    probs = np.zeros((len(ds), num_classes), dtype=np.float32)
    embs = np.zeros((len(ds), emb_dim), dtype=np.float32)
    labels = np.zeros((len(ds),), dtype=np.int64)

    with torch.no_grad():
        for i in tqdm(range(len(ds)), desc=f"  [{key}] val forward", leave=False):
            windows, lab = ds[i]  # (W, 1, T), scalar
            p, e = model.predict_proba_and_embed(windows.to(device), temperature=1.0)
            probs[i] = p.cpu().numpy()
            embs[i] = e.cpu().numpy()
            labels[i] = int(lab.item())

    DATA.mkdir(parents=True, exist_ok=True)
    np.save(DATA / f"val_probs_{key}.npy", probs)
    np.save(DATA / f"val_emb_{key}.npy", embs)
    np.save(DATA / "val_labels.npy", labels)
    return {"encoder": key, "probs_shape": list(probs.shape),
            "emb_shape": list(embs.shape)}


def load_decision_artifacts(data_dir: Optional[Path] = None) -> dict:
    """Load val probs/embeddings + centroids for every active (non-zero-weight)
    encoder, driven by ``ensemble_fusion_weights.json``."""
    data_dir = data_dir or DATA
    fw = json.loads((data_dir / "ensemble_fusion_weights.json").read_text(encoding="utf-8"))
    encoder_names: List[str] = fw["encoder_names"]
    weights = np.asarray(fw["weights"], dtype=np.float64)
    active = [i for i, w in enumerate(weights) if w > 1e-8]
    encoder_names = [encoder_names[i] for i in active]
    weights = weights[active]
    weights = weights / weights.sum()

    probs, emb, cent, sids = [], [], [], []
    for enc in encoder_names:
        probs.append(np.load(data_dir / f"val_probs_{enc}.npy").astype(np.float64))
        emb.append(np.load(data_dir / f"val_emb_{enc}.npy").astype(np.float32))
        z = np.load(data_dir / f"centroids_{enc}.npz")
        cent.append(z["centroids"].astype(np.float32))
        sids.append(z["speaker_ids"].astype(np.int64))

    labels = np.load(data_dir / "val_labels.npy").astype(np.int64)
    return {
        "encoder_names": encoder_names,
        "weights": weights,
        "probs": probs,
        "emb": emb,
        "centroids": cent,
        "speaker_ids": sids,
        "labels": labels,
    }


def tune_decision_bundle(artifacts: dict, num_classes: int = NUM_CLASSES) -> dict:
    """Coordinate-descent sweep of (alpha, kappa, tau, lambda_unknown).

    Returns the decision-bundle dict written to ``decision_config.json``.
    """
    from src.metrics import macro_f1_score
    from submission.inference import centroid_probs_matrix

    weights = artifacts["weights"]
    n_models = len(artifacts["encoder_names"])
    labels = artifacts["labels"]
    print("=" * 60)
    print("  Decision-layer tuning (Macro-F1)")
    print("=" * 60)
    print(f"  Encoders: {artifacts['encoder_names']} "
          f"weights={weights.round(3).tolist()}")
    print(f"  Val samples: {len(labels):,}")

    head_ens = np.tensordot(weights, np.stack(artifacts["probs"]), axes=(0, 0))

    def ensemble_centroid(kappa: float):
        per_probs, per_mc = [], []
        for i in range(n_models):
            cp, mc = centroid_probs_matrix(
                artifacts["emb"][i], artifacts["centroids"][i],
                artifacts["speaker_ids"][i], num_classes, kappa,
            )
            per_probs.append(cp)
            per_mc.append(mc)
        ens_probs = np.tensordot(weights, np.stack(per_probs), axes=(0, 0))
        ens_mc = np.tensordot(weights, np.stack(per_mc), axes=(0, 0))
        return ens_probs, ens_mc

    def evaluate(params: Dict[str, float]) -> float:
        cent, mc = ensemble_centroid(params["kappa"])
        fused = params["alpha"] * head_ens + (1.0 - params["alpha"]) * cent
        fused = fused.copy()
        fused[:, 0] *= params["lambda_unknown"]
        fused /= (fused.sum(axis=1, keepdims=True) + 1e-12)
        pred = fused.argmax(axis=1).astype(np.int64)
        pred[mc < params["tau"]] = 0
        return macro_f1_score(labels, pred, num_classes)

    baseline_params = {"alpha": 1.0, "kappa": 8.0, "tau": 0.0, "lambda_unknown": 1.0}
    baseline = evaluate(baseline_params)
    print(f"\n  Baseline (plain argmax head ensemble): Macro-F1 = {baseline:.4f}")

    grids = {
        "alpha": np.round(np.arange(0.0, 1.001, 0.05), 2),
        "kappa": np.array([0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0]),
        "tau": np.round(np.arange(0.0, 0.60, 0.02), 3),
        "lambda_unknown": np.round(np.arange(0.5, 1.601, 0.05), 2),
    }
    order = ["alpha", "kappa", "tau", "lambda_unknown"]
    best = dict(baseline_params)
    best_score = baseline

    for rnd in range(3):
        for key in order:
            improved = False
            for v in grids[key]:
                cand = dict(best)
                cand[key] = float(v)
                score = evaluate(cand)
                if score > best_score:
                    best_score = score
                    best = cand
                    improved = True
            if improved:
                print(f"  [round {rnd+1}] {key:>15s} -> {best[key]} "
                      f"(Macro-F1 {best_score:.4f})")

    print(f"\n  ── Result ──")
    print(f"  Best Macro-F1: {best_score:.4f}  (baseline {baseline:.4f}, "
          f"Δ = {best_score - baseline:+.4f})")
    print(f"  Params: {best}")

    decision_params = {**best, "temperature": 1.0}
    return {
        "decision_params": decision_params,
        "val_macro_f1": float(best_score),
        "baseline_val_macro_f1": float(baseline),
        "delta": float(best_score - baseline),
        "encoder_names": artifacts["encoder_names"],
        "fusion_weights": weights.tolist(),
        "num_classes": num_classes,
        "note": "Tuned on the leak-free val split with inference-consistent "
                "prob-averaged head probs.",
    }
