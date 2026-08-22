"""Exp 2 — VAD-aware eval-window aggregation.

The current inference averages ALL eval windows (uniform mean) for both the
head probs and the embedding. For files with sparse speech (low VAD ratio)
the silent windows dilute the speaker signal and pull the embedding toward
the shared 'silence region' — the main known->unknown error driver.

This script recomputes the campp val probs + embeddings with
speech-energy-weighted window averaging and evaluates the fused decision
layer (campp_vad + titanet, current decision_config.json) on val.

Usage:
    uv run --no-sync python scripts/eval_vad_aggregation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.cli_utils import setup_utf8_stdio  # noqa: E402
setup_utf8_stdio()

from src.data_pipeline import (  # noqa: E402
    prepare_clean_split, SpeakerDataset, split_args_from_config,
)
from src.model_factory import create_model_from_config  # noqa: E402

DATA = ROOT / "data" / "processed"
CKPT = ROOT / "checkpoints" / "campp_best.pt"


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
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
        unknown_cluster_map=None,          # native split, apples-to-apples
    )
    ds = SpeakerDataset(
        val_df, data_cfg["audio_dir"], sample_rate=audio_cfg["sample_rate"],
        duration_seconds=audio_cfg["duration_seconds"], augment=False,
        num_train_windows=audio_cfg.get("num_train_windows", 1),
        eval_hop_ratio=audio_cfg.get("eval_hop_ratio", 0.5),
        max_eval_windows=audio_cfg.get("max_eval_windows", 8),
    )
    num_classes = int(model.num_output_classes)
    emb_dim = getattr(model.head_speaker, "embedding_dim", 192)
    files = list(val_df["audio_file"])

    # per-window RMS helper on the raw windowed waveform
    def window_rms_db(windows: torch.Tensor) -> torch.Tensor:
        x = windows.reshape(windows.shape[0], -1)          # (W, T)
        rms = x.pow(2).mean(dim=1).sqrt().clamp_min(1e-12)
        return 20 * torch.log10(rms + 1e-9)

    def weights_from_rms(rms_db: torch.Tensor, mode: str) -> torch.Tensor:
        if rms_db.numel() == 0:
            return torch.ones(1)
        floor_db = max(float(rms_db.quantile(0.15)), -55.0)
        w = (rms_db - floor_db).clamp_min(0.0)
        if mode == "sq":
            w = w * w
        if w.sum() <= 1e-9:                     # fully silent file → uniform
            return torch.ones_like(rms_db)
        return w / w.sum()

    outs = {m: {"probs": np.zeros((len(ds), num_classes), np.float32),
                "emb": np.zeros((len(ds), emb_dim), np.float32)}
            for m in ["uniform", "lin", "sq"]}

    with torch.no_grad():
        for i in range(len(ds)):
            windows, _ = ds[i]                              # (W, 1, T)
            w_t = windows.to(device)
            rms_db = window_rms_db(windows)
            hidden, _ = model.encoder(w_t)
            pooled = model.pooling(hidden)
            ood_logit = model.head_ood(pooled)
            speaker_logits = model.head_speaker(pooled)
            if hasattr(model.head_speaker, "embedding_proj"):
                raw_emb = model.head_speaker.embedding_proj(pooled)
            else:
                raw_emb = pooled
            p_unknown = torch.sigmoid(ood_logit)
            p_known = torch.softmax(speaker_logits / 1.0, dim=1)
            p_known_scaled = (1.0 - p_unknown.expand(-1, model.num_known_speakers)) * p_known
            probs_w = model._collapse_probs(p_unknown, p_known_scaled)   # (W, 447)
            probs_w = torch.clamp(probs_w, min=1e-7, max=1.0 - 1e-7)
            probs_w = probs_w / probs_w.sum(dim=1, keepdim=True)

            for m in outs:
                if m == "uniform":
                    wt = torch.ones_like(rms_db) / max(len(rms_db), 1)
                else:
                    wt = weights_from_rms(rms_db, "sq" if m == "sq" else "lin")
                p = (wt.to(device).unsqueeze(1) * probs_w).sum(dim=0)
                p = p / p.sum()
                e = torch.nn.functional.normalize(
                    (wt.to(device).unsqueeze(1) * raw_emb).sum(dim=0, keepdim=True), p=2, dim=1)[0]
                outs[m]["probs"][i] = p.cpu().numpy()
                outs[m]["emb"][i] = e.cpu().numpy()
            if (i + 1) % 200 == 0:
                print(f"  ...{i+1}/{len(ds)}")

    np.save(DATA / "val_probs_campp_vad.npy", outs["lin"]["probs"])
    np.save(DATA / "val_emb_campp_vad.npy", outs["lin"]["emb"])
    (DATA / "val_campp_vad_files.json").write_text(
        json.dumps(files), encoding="utf-8")

    # ---------- evaluate fused decision (campp_vad + titanet) ----------
    from sklearn.metrics import f1_score
    from submission.inference import centroid_probs_matrix, _collapse_centroid_probs

    per = pd.read_csv(DATA / "model_compare_legacy_vs_cluster" / "per_file_predictions.csv")
    idx = {f: i for i, f in enumerate(per["audio_file"])}
    order = [idx[f] for f in files]
    lbl = per["true_label"].values[order]

    cents_c = np.load(DATA / "centroids_campp.npz")["centroids"].astype(np.float32)
    cents_t = np.load(DATA / "centroids_titanet.npz")["centroids"].astype(np.float32)
    uc = np.load(DATA / "centroids_unknown_campp.npz")["centroids"].astype(np.float32)
    allc = np.vstack([cents_c, uc])
    sids = np.concatenate([np.arange(1, 447), np.arange(447, 447 + len(uc))])
    probs_t = np.load(DATA / "val_probs_titanet.npy").astype(np.float64)[order]
    emb_t = np.load(DATA / "val_emb_titanet.npy").astype(np.float32)[order]
    w_c, w_t = 0.8333, 0.1667

    def mf1(p):
        return float(f1_score(lbl, p, labels=list(range(447)), average="macro",
                              zero_division=0))

    def evaluate(probs_c, emb_c, cfg):
        head = w_c * probs_c + w_t * probs_t
        cp_c, mc_c = centroid_probs_matrix(emb_c, allc, sids, 1001, cfg["kappa"])
        cp_c = _collapse_centroid_probs(cp_c, 447)
        cp_t, mc_t = centroid_probs_matrix(emb_t, cents_t, np.arange(1, 447), 447, cfg["kappa"])
        cent = w_c * cp_c + w_t * cp_t
        mc = w_c * mc_c + w_t * mc_t
        fused = cfg["alpha"] * head + (1 - cfg["alpha"]) * cent
        fused[:, 0] *= cfg["lambda_unknown"]
        fused /= fused.sum(1, keepdims=True)
        pred = fused.argmax(1)
        pred[mc < cfg["tau"]] = 0
        return pred, mf1(pred)

    cfg = dict(alpha=0.3, kappa=16.0, tau=0.0, lambda_unknown=0.65)

    print("\n=== Exp 2 results (fused campp_vad + titanet, decision_config) ===")
    for m in ["uniform", "lin", "sq"]:
        pred, sc = evaluate(outs[m]["probs"], outs[m]["emb"], cfg)
        print(f"  {m:>8s} weighting: macro-F1 = {sc:.4f}  "
              f"known-err {(pred[lbl>0]!=lbl[lbl>0]).sum()}  "
              f"unk->known {((pred>0)&(lbl==0)).sum()}")

    pred_v, sc_v = evaluate(outs["lin"]["probs"], outs["lin"]["emb"], cfg)
    print("\n=== remaining errors (lin weighting) ===")
    errs = per.iloc[order].copy()
    errs["pred"] = pred_v
    e2 = errs[errs["pred"] != errs["true_label"]]
    print(e2[["audio_file", "true_label", "pred"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())