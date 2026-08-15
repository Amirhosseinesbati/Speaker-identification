"""
Q3 — Offline decision-layer tuning (no GPU, uses dumped val artifacts).

Sweeps the centroid + OOD-gate decision knobs — ``alpha`` (head↔centroid
fusion), ``kappa`` (centroid softmax scale), ``tau`` (OOD gate on max cosine)
and ``lambda_unknown`` (unknown-prob bias) — directly against the competition
Macro-F1 on the leak-free val split, using the inference-consistent artifacts
dumped by ``scripts/dump_val_artifacts.py`` (Q2) and the centroids built by
``scripts/build_centroids.py`` (Q4).

Writes ``data/processed/decision_config.json`` (shipped by build_submission.py)
and prints the tuned Macro-F1 vs the plain-argmax head baseline.

Usage:
    uv run --no-sync python scripts/tune_decision.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.cli_utils import setup_utf8_stdio  # noqa: E402
setup_utf8_stdio()

from src.metrics import macro_f1_score  # noqa: E402
from submission.inference import centroid_probs_matrix  # noqa: E402

DATA = ROOT / "data" / "processed"
FUSION_JSON = DATA / "ensemble_fusion_weights.json"
OUT_JSON = DATA / "decision_config.json"
NUM_CLASSES = 447


def load_artifacts() -> dict:
    fw = json.loads(FUSION_JSON.read_text(encoding="utf-8"))
    encoder_names: List[str] = fw["encoder_names"]
    weights = np.asarray(fw["weights"], dtype=np.float64)
    active = [i for i, w in enumerate(weights) if w > 1e-8]
    encoder_names = [encoder_names[i] for i in active]
    weights = weights[active]
    weights = weights / weights.sum()

    probs, emb, cent, sids = [], [], [], []
    for enc in encoder_names:
        probs.append(np.load(DATA / f"val_probs_{enc}.npy").astype(np.float64))
        emb.append(np.load(DATA / f"val_emb_{enc}.npy").astype(np.float32))
        z = np.load(DATA / f"centroids_{enc}.npz")
        cent.append(z["centroids"].astype(np.float32))
        sids.append(z["speaker_ids"].astype(np.int64))

    labels = np.load(DATA / "val_labels.npy").astype(np.int64)
    return {
        "encoder_names": encoder_names,
        "weights": weights,
        "probs": probs,
        "emb": emb,
        "centroids": cent,
        "speaker_ids": sids,
        "labels": labels,
    }


def main() -> int:
    a = load_artifacts()
    weights = a["weights"]
    n_models = len(a["encoder_names"])
    labels = a["labels"]
    print("=" * 60)
    print("  Q3 — Offline decision-layer tuning (Macro-F1)")
    print("=" * 60)
    print(f"  Encoders: {a['encoder_names']} weights={weights.round(3).tolist()}")
    print(f"  Val samples: {len(labels):,}")

    # Head ensemble probs (prob-averaged at T=1 — exactly what inference uses).
    head_ens = np.tensordot(weights, np.stack(a["probs"]), axes=(0, 0))  # (N, 447)

    def ensemble_centroid(kappa: float):
        per_probs, per_mc = [], []
        for i in range(n_models):
            cp, mc = centroid_probs_matrix(
                a["emb"][i], a["centroids"][i], a["speaker_ids"][i],
                NUM_CLASSES, kappa,
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
        return macro_f1_score(labels, pred, NUM_CLASSES)

    # ── Baseline: plain argmax head ensemble (no gate, no centroid) ──
    baseline_params = {"alpha": 1.0, "kappa": 8.0, "tau": 0.0, "lambda_unknown": 1.0}
    baseline = evaluate(baseline_params)
    print(f"\n  Baseline (plain argmax head ensemble): Macro-F1 = {baseline:.4f}")

    # ── Coordinate descent over the 4 knobs (3 rounds) ──
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

    # Match the exact decision_config schema read by submission.submission.
    decision_params = {**best, "temperature": 1.0}
    output = {
        "decision_params": decision_params,
        "val_macro_f1": float(best_score),
        "baseline_val_macro_f1": float(baseline),
        "delta": float(best_score - baseline),
        "encoder_names": a["encoder_names"],
        "fusion_weights": weights.tolist(),
        "num_classes": NUM_CLASSES,
        "note": "Tuned on the leak-free val split (val_per_known=1) with "
                "inference-consistent prob-averaged head probs.",
    }
    OUT_JSON.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"\n  ✓ Saved to {OUT_JSON}")
    print("\n✅ Decision tuning complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
