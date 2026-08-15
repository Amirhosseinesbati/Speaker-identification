"""
Q2 — Dump per-model val artifacts for the offline decision tuner.

Thin CLI over ``src/decision_engine.dump_val_checkpoint`` (single source of
truth shared with the pipeline's ``build_embeddings`` step). For each
``checkpoints/<enc>_best.pt`` it runs the EXACT inference forward path and
saves:

    data/processed/val_probs_<enc>.npy  (N, 447)  head probs (T=1, prob-avg)
    data/processed/val_emb_<enc>.npy    (N, 192)  L2-normalised embedding
    data/processed/val_labels.npy       (N,)      ground-truth global ids

Usage:
    uv run --no-sync python scripts/dump_val_artifacts.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.cli_utils import setup_utf8_stdio  # noqa: E402
setup_utf8_stdio()

from src.decision_engine import dump_val_checkpoint  # noqa: E402

DATA = ROOT / "data" / "processed"
CKPT_DIR = ROOT / "checkpoints"


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
        info = dump_val_checkpoint(ckpt, device)
        print(f"  ✓ {info['encoder']}: probs {info['probs_shape']} | "
              f"emb {info['emb_shape']}")

    print("\n✅ Val artifacts dumped (prob-avg probs + mean_then_l2norm embeddings).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
