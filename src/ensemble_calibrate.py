"""
Ensemble + temperature calibration report (Step 9 — run after training).

Loads N trained checkpoints, evaluates each on the leak-free val split and
reports:
  - per-model Macro-F1 (the competition metric)
  - Macro-F1 for 6 fusion strategies:
      * average (equal weights)        — baseline
      * weighted_average (grid search) — optimised per-model weights
      * geometric_mean                 — dampens weak models
      * rank_average                   — scale-invariant
      * max_prob                       — element-wise max
      * learned_mlp                    — trained MLP fusion
  - best speaker-softmax temperature for each ensemble (via
    src.metrics.calibrate_temperature)
  - saves best fusion weights to data/processed/ensemble_fusion_weights.json

Usage (after training several models, e.g. different seeds/encoders):
    uv run --no-sync python -m src.ensemble_calibrate \
        --checkpoints checkpoints/ecapa_best.pt checkpoints/campp_best.pt \
        checkpoints/eres2net_best.pt checkpoints/titanet_best.pt

Only heavy when a checkpoint needs its encoder loaded — this script itself
only runs validation forwards on the val split.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Windows cp1252 fix: force UTF-8 stdio so emoji output never crashes.
from src.cli_utils import setup_utf8_stdio
setup_utf8_stdio()

RESULTS_JSON = PROJECT_ROOT / "data" / "processed" / "ensemble_fusion_results.json"
WEIGHTS_JSON = PROJECT_ROOT / "data" / "processed" / "ensemble_fusion_weights.json"


# ────────────────────────────────────────────────────────────────
#  Val split + logit collection
# ────────────────────────────────────────────────────────────────

def collect_val_logits(
    model: torch.nn.Module,
    device: torch.device,
    config: dict,
    batch_size: int = 16,
) -> dict:
    """Run multi-window validation forward (no margin) over the val split.

    Args:
        config: the effective config (first checkpoint's embedded one or the
                --config-path fallback) that defines audio TTA + data paths.
    """
    from torch.utils.data import DataLoader
    from src.data_pipeline import (
        prepare_clean_split, SpeakerDataset, split_args_from_config,
    )
    from src.train import forward_multi_window

    audio_cfg = config["audio"]
    data_cfg = config["data"]

    _, val_df, class_map = prepare_clean_split(
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
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    all_ood, all_spk, all_lbl = [], [], []
    model.eval()
    with torch.no_grad():
        for windows, labels in tqdm(dl, desc="  Val logits", leave=False):
            ood, spk = forward_multi_window(model, windows.to(device), labels=None)
            all_ood.append(ood.cpu())
            all_spk.append(spk.cpu())
            all_lbl.append(labels.cpu())

    return {
        "ood": torch.cat(all_ood),
        "spk": torch.cat(all_spk),
        "labels": torch.cat(all_lbl),
        "num_classes": len(class_map),
    }


# ────────────────────────────────────────────────────────────────
#  Checkpoint loading
# ────────────────────────────────────────────────────────────────

def load_checkpoint_model(
    checkpoint_path: str,
    device: torch.device,
    fallback_config_path: Optional[str],
) -> Tuple[torch.nn.Module, dict, Dict[str, int]]:
    """Build a model from the config EMBEDDED in the checkpoint.

    Mirrors submission/inference.load_model — each checkpoint carries its own
    config (encoder type, pooling, embedding dims), so the 5-encoder ensemble
    must NOT be built from one shared config file (that breaks load_state_dict
    for every non-ECAPA encoder).

    Returns:
        model, config, class_map
    """
    from src.data_pipeline import load_config
    from src.model_factory import create_model_from_config

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    config = checkpoint.get("config")
    if config is None:
        if fallback_config_path is None:
            raise RuntimeError(
                f"Checkpoint {checkpoint_path} has no embedded config — pass "
                "--config-path as the fallback."
            )
        config = load_config(fallback_config_path)
        print(f"  ⚠ {checkpoint_path}: no embedded config — used {fallback_config_path}")

    class_map = checkpoint.get("class_map")
    if class_map is None:
        labels_path = config["data"]["labels_path"]
        df = pd.read_csv(labels_path)
        df.columns = df.columns.str.strip()
        from src.data_pipeline import create_class_mapping
        class_map = create_class_mapping(df)
        print(f"  ⚠ class_map not in checkpoint — rebuilt from {labels_path}")

    num_known = len(class_map) - 1
    model = create_model_from_config(config, num_known_speakers=num_known)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, config, class_map


# ────────────────────────────────────────────────────────────────
#  Helper: convert logits → 447-way probability vector
# ────────────────────────────────────────────────────────────────

def logits_to_probs(
    ood_logits: torch.Tensor,
    spk_logits: torch.Tensor,
    num_unknown_clusters: int = 0,
) -> np.ndarray:
    """Fuse OOD + speaker logits into a 447-way probability vector (numpy)."""
    from src.metrics import fused_probs_from_logits
    return fused_probs_from_logits(
        ood_logits, spk_logits, temperature=1.0,
        num_unknown_clusters=num_unknown_clusters,
    ).numpy()


# ────────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────────

def main(checkpoints: List[str], config_path: str, batch_size: int,
         skip_learned_mlp: bool = False) -> None:
    from src.metrics import evaluate_macro_f1, calibrate_temperature, macro_f1_score
    from src.data_pipeline import load_config
    from src.ensemble import (
        weighted_average_fusion,
        geometric_mean_fusion,
        rank_average_fusion,
        max_prob_fusion,
        grid_search_weights,
        train_learned_fusion,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_models = len(checkpoints)

    print("=" * 60)
    print("  Ensemble + Smart Fusion Calibration")
    print("=" * 60)
    print(f"  Device: {device} | Checkpoints: {n_models}")

    # TTA + data params come from the FIRST checkpoint's embedded config
    first_ckpt = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
    first_cfg = first_ckpt.get("config")
    if first_cfg is None:
        first_cfg = load_config(config_path)
        print(f"  ⚠ {checkpoints[0]}: no embedded config — used {config_path}")

    # ── Phase A: collect per-model logits + probs ──
    per_model = []
    all_probs: List[np.ndarray] = []
    all_oods: List[torch.Tensor] = []
    all_spks: List[torch.Tensor] = []
    encoder_names: List[str] = []
    labels: Optional[np.ndarray] = None
    num_classes = 447

    for ckpt in checkpoints:
        ckpt_name = Path(ckpt).name
        print(f"\n  Loading {ckpt_name}...")
        model, _, class_map = load_checkpoint_model(ckpt, device, config_path)
        num_unknown_clusters = int(model.num_unknown_clusters)
        num_classes = len(class_map) - num_unknown_clusters
        enc_name = getattr(model, "encoder_name", Path(ckpt).stem)
        encoder_names.append(enc_name)

        val = collect_val_logits(model, device, first_cfg, batch_size=batch_size)
        num_classes = val["num_classes"] - num_unknown_clusters
        m = evaluate_macro_f1(
            val["ood"], val["spk"], val["labels"], num_classes=num_classes,
            num_unknown_clusters=num_unknown_clusters,
        )
        per_model.append({"checkpoint": ckpt_name, "encoder": enc_name, **m})
        print(f"  Model Macro-F1: {m['macro_f1']:.4f} "
              f"(OOD-F1 {m['ood_f1']:.4f}, known_acc {m['known_acc']:.4f}) "
              f"[{enc_name}]")

        # Convert logits to probability vectors
        probs = logits_to_probs(val["ood"], val["spk"],
                                 num_unknown_clusters=num_unknown_clusters)
        all_probs.append(probs)
        all_oods.append(val["ood"])
        all_spks.append(val["spk"])
        labels = val["labels"].numpy()

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    assert labels is not None and len(all_probs) == n_models

    # ── Phase B: try all fusion methods ──
    print(f"\n{'─' * 60}")
    print("  Fusion Method Comparison")
    print(f"{'─' * 60}")

    fusion_results = {}

    # ── B1: Simple average (baseline) ──
    t0 = time.time()
    avg_probs = weighted_average_fusion(all_probs, weights=None)
    avg_preds = avg_probs.argmax(axis=1)
    avg_mf1 = macro_f1_score(labels, avg_preds, num_classes=num_classes)
    fusion_results["average"] = {
        "method": "average",
        "macro_f1": float(avg_mf1),
        "weights": [round(1.0 / n_models, 4)] * n_models,
        "time_s": round(time.time() - t0, 2),
    }
    print(f"  📊 average (equal weights):        Macro-F1 = {avg_mf1:.4f}")

    # ── B2: Weighted average (grid search) ──
    t0 = time.time()
    gs = grid_search_weights(all_probs, labels, num_classes=num_classes, step=0.05)
    fusion_results["weighted_average"] = {
        "method": "weighted_average (grid search)",
        "macro_f1": float(gs["best_macro_f1"]),
        "weights": gs["best_weights"],
        "time_s": round(time.time() - t0, 2),
    }
    w_str = ", ".join(f"{w:.2f}" for w in gs["best_weights"])
    print(f"  📊 weighted_average (grid search):  Macro-F1 = {gs['best_macro_f1']:.4f}  "
          f"weights=[{w_str}]")

    # ── B3: Geometric mean ──
    t0 = time.time()
    geo_probs = geometric_mean_fusion(all_probs)
    geo_preds = geo_probs.argmax(axis=1)
    geo_mf1 = macro_f1_score(labels, geo_preds, num_classes=num_classes)
    fusion_results["geometric_mean"] = {
        "method": "geometric_mean",
        "macro_f1": float(geo_mf1),
        "time_s": round(time.time() - t0, 2),
    }
    print(f"  📊 geometric_mean:                  Macro-F1 = {geo_mf1:.4f}")

    # ── B4: Rank average ──
    t0 = time.time()
    rank_probs = rank_average_fusion(all_probs)
    rank_preds = rank_probs.argmax(axis=1)
    rank_mf1 = macro_f1_score(labels, rank_preds, num_classes=num_classes)
    fusion_results["rank_average"] = {
        "method": "rank_average",
        "macro_f1": float(rank_mf1),
        "time_s": round(time.time() - t0, 2),
    }
    print(f"  📊 rank_average:                    Macro-F1 = {rank_mf1:.4f}")

    # ── B5: Max prob ──
    t0 = time.time()
    max_probs = max_prob_fusion(all_probs)
    max_preds = max_probs.argmax(axis=1)
    max_mf1 = macro_f1_score(labels, max_preds, num_classes=num_classes)
    fusion_results["max_prob"] = {
        "method": "max_prob",
        "macro_f1": float(max_mf1),
        "time_s": round(time.time() - t0, 2),
    }
    print(f"  📊 max_prob:                        Macro-F1 = {max_mf1:.4f}")

    # ── B6: Learned MLP fusion ──
    if not skip_learned_mlp and n_models >= 2:
        print(f"\n  🧠 Training LearnedFusion MLP (this may take a minute)...")
        t0 = time.time()
        mlp_fusion, mlp_info = train_learned_fusion(
            all_probs, labels, num_classes=num_classes,
            hidden_dim=512, dropout=0.3, lr=1e-3, epochs=200,
            train_ratio=0.8, patience=20,
            device_str=str(device),
        )
        mlp_time = round(time.time() - t0, 2)

        # Evaluate on full val set
        import torch as _torch
        mlp_fusion.eval()
        X_full = np.concatenate(all_probs, axis=1)
        with _torch.no_grad():
            chunks = []
            for i in range(n_models):
                chunk = _torch.tensor(
                    X_full[:, i * num_classes:(i + 1) * num_classes],
                    dtype=_torch.float32,
                )
                chunks.append(chunk.to(device))
            mlp_output = mlp_fusion(chunks).cpu().numpy()
        mlp_preds = mlp_output.argmax(axis=1)
        mlp_mf1 = macro_f1_score(labels, mlp_preds, num_classes=num_classes)

        fusion_results["learned_mlp"] = {
            "method": "learned_mlp",
            "macro_f1": float(mlp_mf1),
            "val_macro_f1": float(mlp_info["best_val_macro_f1"]),
            "best_epoch": mlp_info["best_epoch"],
            "time_s": mlp_time,
        }
        print(f"  📊 learned_mlp:                     Macro-F1 = {mlp_mf1:.4f}  "
              f"(val={mlp_info['best_val_macro_f1']:.4f} @ epoch {mlp_info['best_epoch']})")

        # Save MLP state for submission
        mlp_path = PROJECT_ROOT / "data" / "processed" / "learned_fusion_mlp.pt"
        _torch.save(mlp_fusion.state_dict(), mlp_path)
        fusion_results["learned_mlp"]["state_path"] = str(mlp_path)
        print(f"  💾 Saved LearnedFusion state to {mlp_path}")

    # ── Phase C: Temperature calibration on the best fusion ──
    print(f"\n{'─' * 60}")
    print("  Temperature Calibration (on averaged logits)")
    print(f"{'─' * 60}")

    # Average logits (for temperature calibration — per-model logit averaging)
    ens_ood = sum(all_oods) / n_models  # type: ignore[operator]
    ens_spk = sum(all_spks) / n_models  # type: ignore[operator]
    cal = calibrate_temperature(
        ens_ood, ens_spk, torch.from_numpy(labels), num_classes=num_classes,
    )
    print(f"  Best temperature: {cal['best_temperature']:.2f} "
          f"(Macro-F1 {cal['macro_f1_at_best_t']:.4f} vs "
          f"T=1.0 {cal['macro_f1_at_t1']:.4f})")
    fusion_results["temperature_calibration"] = cal

    # ── Phase D: Summary & save ──
    print(f"\n{'=' * 60}")
    print("  🏆 Fusion Results Summary")
    print(f"{'=' * 60}")

    # Sort by Macro-F1 descending
    sorted_methods = sorted(
        [(k, v) for k, v in fusion_results.items()
         if "macro_f1" in v and k != "temperature_calibration"],
        key=lambda x: x[1]["macro_f1"],
        reverse=True,
    )

    for rank, (method, result) in enumerate(sorted_methods, 1):
        marker = "👑" if rank == 1 else "  "
        extra = ""
        if method == "weighted_average":
            extra = f" weights={result.get('weights', '?')}"
        elif method == "learned_mlp":
            extra = f" (val={result.get('val_macro_f1', 0):.4f})"
        print(f"  {marker} #{rank}: {result['method']:<35s} "
              f"Macro-F1 = {result['macro_f1']:.4f}{extra}")

    # Save results JSON
    output = {
        "checkpoints": [str(Path(c).name) for c in checkpoints],
        "encoder_names": encoder_names,
        "num_models": n_models,
        "num_classes": num_classes,
        "per_model": per_model,
        "fusion_results": fusion_results,
        "best_method": sorted_methods[0][0] if sorted_methods else "average",
        "best_macro_f1": sorted_methods[0][1]["macro_f1"] if sorted_methods else 0.0,
    }
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(
        json.dumps(output, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\n  ✓ Full results saved to {RESULTS_JSON}")

    # Save best weights for submission
    best_method = sorted_methods[0][0] if sorted_methods else "average"
    weights_data = {
        "fusion_method": best_method,
        "best_macro_f1": output["best_macro_f1"],
        # Encoder order the weights apply to (checkpoint names). The leaderboard
        # entrypoint uses this to align checkpoints with weights exactly.
        "encoder_names": encoder_names,
        "checkpoints": output["checkpoints"],
        # The weighted-average weights are ALWAYS shipped, even when a different
        # method (e.g. max_prob) scored marginally higher. Both the decision
        # layer (decision_tune → load_decision_artifacts) and the leaderboard
        # entrypoint (submission.py) fuse via weighted_average, so `weights` must
        # be present or the next stage crashes with a KeyError.
        "weights": fusion_results["weighted_average"]["weights"],
    }
    if best_method == "learned_mlp":
        weights_data["state_path"] = fusion_results["learned_mlp"].get("state_path", "")
    WEIGHTS_JSON.write_text(
        json.dumps(weights_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  ✓ Best fusion config saved to {WEIGHTS_JSON}")

    print(f"\n✅ Ensemble calibration complete.")
    print(f"   Best fusion: {best_method} (Macro-F1 = {output['best_macro_f1']:.4f})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ensemble + smart fusion calibration report (Step 9).")
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="Trained checkpoint paths (2+ for an ensemble).")
    parser.add_argument("--config-path", default="configs/default_config.yaml")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--skip-learned-mlp", action="store_true",
                        help="Skip the LearnedFusion MLP training (saves time).")
    args = parser.parse_args()
    main(args.checkpoints, args.config_path, args.batch_size,
         skip_learned_mlp=args.skip_learned_mlp)