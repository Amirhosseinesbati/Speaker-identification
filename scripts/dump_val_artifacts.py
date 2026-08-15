"""
Q2 — Dump per-model val artifacts for the offline decision tuner.

For each ``checkpoints/<enc>_best.pt``, runs the EXACT inference forward path
(``model.predict_proba_and_embed`` — probability-averaged head probs + mean-then
L2-normalised ArcFace embedding, single encoder pass) over the leak-free val
split and saves:

    data/processed/val_probs_<enc>.npy  (N, 447)  head probs (T=1, prob-avg)
    data/processed/val_emb_<enc>.npy    (N, 192)  L2-normalised embedding
    data/processed/val_labels.npy       (N,)      ground-truth global ids

This is kept SEPARATE from ``ensemble_calibrate.py`` (which averages LOGITS for
the fusion report) so the decision tuner operates on exactly the same
probability semantics the submission uses (avoids R9.④ prob-avg vs logit-avg
mismatch).

Usage:
    uv run --no-sync python scripts/dump_val_artifacts.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.cli_utils import setup_utf8_stdio  # noqa: E402
setup_utf8_stdio()

DATA = ROOT / "data" / "processed"
CKPT_DIR = ROOT / "checkpoints"


def encoder_key(checkpoint_path: str) -> str:
    return Path(checkpoint_path).name.replace("_best.pt", "")


def dump_checkpoint(checkpoint_path: str, device: torch.device) -> None:
    from src.data_pipeline import prepare_clean_split, SpeakerDataset
    from src.model_factory import create_model_from_config

    key = encoder_key(checkpoint_path)
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ck["config"]
    class_map = ck["class_map"]
    num_known = config.get("model", {}).get("competition_num_known", len(class_map) - 1)

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
    )

    ds = SpeakerDataset(
        val_df, data_cfg["audio_dir"], sample_rate=audio_cfg["sample_rate"],
        duration_seconds=audio_cfg["duration_seconds"], augment=False,
        num_train_windows=audio_cfg.get("num_train_windows", 1),
        eval_hop_ratio=audio_cfg.get("eval_hop_ratio", 0.5),
        max_eval_windows=audio_cfg.get("max_eval_windows", 8),
    )

    num_classes = len(class_map)
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
    print(f"  ✓ {key}: probs {probs.shape} | emb {embs.shape}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Dump inference-consistent val artifacts")
    parser.add_argument("--checkpoints", nargs="*", default=None,
                        help="Checkpoint paths (default: all checkpoints/*_best.pt)")
    args = parser.parse_args()

    checkpoints = args.checkpoints or sorted(str(p) for p in CKPT_DIR.glob("*_best.pt"))
    if not checkpoints:
        print("  ⚠ No checkpoints found.")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("  Q2 — Dump inference-consistent val artifacts")
    print("=" * 60)
    print(f"  Device: {device} | Checkpoints: {len(checkpoints)}")

    for ckpt in checkpoints:
        dump_checkpoint(ckpt, device)

    print("\n✅ Val artifacts dumped (prob-avg probs + mean_then_l2norm embeddings).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
