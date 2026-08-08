"""
Ensemble + temperature calibration report (Step 9 — run after training).

Loads N trained checkpoints, evaluates each on the leak-free val split and
reports:
  - per-model Macro-F1 (the competition metric)
  - average-fusion ensemble Macro-F1
  - best speaker-softmax temperature for the ensemble (via
    src.metrics.calibrate_temperature)

Usage (after training several models, e.g. different seeds/encoders):
    uv run --no-sync python -m src.ensemble_calibrate \
        --checkpoints checkpoints/best_seed42.pt checkpoints/best_seed7.pt

Only heavy when a checkpoint needs its encoder loaded — this script itself
only runs validation forwards on the val split.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ────────────────────────────────────────────────────────────────
#  Val split + logit collection
# ────────────────────────────────────────────────────────────────

def collect_val_logits(
    model: torch.nn.Module,
    device: torch.device,
    config_path: str,
    batch_size: int = 16,
) -> dict:
    """Run multi-window validation forward (no margin) over the val split."""
    from torch.utils.data import DataLoader
    from src.data_pipeline import load_config, prepare_clean_split, SpeakerDataset
    from src.train import forward_multi_window

    config = load_config(config_path)
    audio_cfg = config["audio"]
    data_cfg = config["data"]

    _, val_df, class_map = prepare_clean_split(
        labels_path=data_cfg["labels_path"],
        audio_dir=data_cfg["audio_dir"],
        processed_labels=data_cfg["processed_labels"],
        val_per_known=1,
        unknown_val_ratio=0.2,
        min_valid_duration=audio_cfg.get("min_valid_duration", 1.0),
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
#  Main
# ────────────────────────────────────────────────────────────────

def main(checkpoints: List[str], config_path: str, batch_size: int) -> None:
    from src.metrics import evaluate_macro_f1, calibrate_temperature
    from src.model_factory import create_model_from_config
    from src.data_pipeline import load_config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("  Ensemble + Temperature Calibration")
    print("=" * 60)
    print(f"  Device: {device} | Checkpoints: {len(checkpoints)}")

    config = load_config(config_path)
    num_known = config["model"].get("competition_num_known", 446)

    per_model = []
    probs_sum: Optional[np.ndarray] = None
    ood_sum: Optional[torch.Tensor] = None
    spk_sum: Optional[torch.Tensor] = None
    labels: Optional[np.ndarray] = None
    num_classes = 447

    for ckpt in checkpoints:
        print(f"\n  Loading {ckpt}...")
        model = create_model_from_config(config, num_known_speakers=num_known)
        checkpoint = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        val = collect_val_logits(model, device, config_path, batch_size=batch_size)
        num_classes = val["num_classes"]
        m = evaluate_macro_f1(
            val["ood"], val["spk"], val["labels"], num_classes=num_classes,
        )
        per_model.append({"checkpoint": ckpt, **m})
        print(f"  Model Macro-F1: {m['macro_f1']:.4f} "
              f"(OOD-F1 {m['ood_f1']:.4f}, known_acc {m['known_acc']:.4f})")

        # accumulate for the ensemble (average probs + average logits)
        probs = torch.cat(
            [torch.sigmoid(val["ood"]),
             (1 - torch.sigmoid(val["ood"])) * torch.softmax(val["spk"], dim=1)],
            dim=1,
        ).numpy()
        probs_sum = probs if probs_sum is None else probs_sum + probs
        ood_sum = val["ood"] if ood_sum is None else ood_sum + val["ood"]
        spk_sum = val["spk"] if spk_sum is None else spk_sum + val["spk"]
        labels = val["labels"].numpy()

    assert probs_sum is not None and labels is not None
    ensemble_probs = probs_sum / len(checkpoints)
    ensemble_preds = ensemble_probs.argmax(axis=1)

    from src.metrics import macro_f1_score
    ens_mf1 = macro_f1_score(labels, ensemble_preds, num_classes=num_classes)
    print(f"\n  Ensemble (average probs) Macro-F1: {ens_mf1:.4f}")

    # Temperature calibration on the AVERAGED logits (clean semantics)
    ens_ood = ood_sum / len(checkpoints)
    ens_spk = spk_sum / len(checkpoints)
    cal = calibrate_temperature(
        ens_ood, ens_spk, torch.from_numpy(labels), num_classes=num_classes,
    )
    print(f"  Best temperature: {cal['best_temperature']:.2f} "
          f"(Macro-F1 {cal['macro_f1_at_best_t']:.4f} vs "
          f"T=1.0 {cal['macro_f1_at_t1']:.4f})")

    print("\n✅ Ensemble calibration complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ensemble + temperature calibration report (Step 9).")
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="Trained checkpoint paths (2+ for an ensemble).")
    parser.add_argument("--config-path", default="configs/default_config.yaml")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    main(args.checkpoints, args.config_path, args.batch_size)
