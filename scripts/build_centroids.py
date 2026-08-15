"""
Q4 — Build per-encoder speaker centroids from trained checkpoints.

For each ``checkpoints/<enc>_best.pt``, extracts the L2-normalised ArcFace
embeddings of the leak-free TRAIN split via ``model.embed`` and saves a
per-speaker centroid matrix to ``data/processed/centroids_<enc>.npz``:

    centroids    — (446, 192) float32  (row i = global class i+1)
    speaker_ids  — (446,) int64        [1 .. 446]
    embedding_dim, encoder, n_train_files

These centroids are shipped inside the submission zip (the server has no train
data) and used at inference for the cosine centroid + OOD-gate decision layer.

Usage:
    uv run --no-sync python scripts/build_centroids.py
    uv run --no-sync python scripts/build_centroids.py --checkpoints checkpoints/campp_best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.cli_utils import setup_utf8_stdio  # noqa: E402
setup_utf8_stdio()

from src.centroid_baseline import build_checkpoint_centroids  # noqa: E402

DATA_PROCESSED = ROOT / "data" / "processed"
CKPT_DIR = ROOT / "checkpoints"


def discover_checkpoints() -> list:
    if not CKPT_DIR.exists():
        return []
    return sorted(str(p) for p in CKPT_DIR.glob("*_best.pt"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build centroids from checkpoints")
    parser.add_argument("--checkpoints", nargs="*", default=None,
                        help="Checkpoint paths (default: all checkpoints/*_best.pt)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-eval-windows", type=int, default=None,
                        help="Override max windows per file (default: checkpoint config)")
    args = parser.parse_args()

    checkpoints = args.checkpoints or discover_checkpoints()
    if not checkpoints:
        print("  ⚠ No *_best.pt checkpoints found.")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("  Q4 — Build speaker centroids from checkpoints")
    print("=" * 60)
    print(f"  Device: {device} | Checkpoints: {len(checkpoints)}")

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    manifest = []
    for ckpt in checkpoints:
        name = Path(ckpt).name.replace("_best.pt", "")
        print(f"\n  [{name}] Building centroids...")
        out = build_checkpoint_centroids(
            ckpt, device, batch_size=args.batch_size,
            max_eval_windows=args.max_eval_windows,
        )
        out_path = DATA_PROCESSED / f"centroids_{name}.npz"
        np.savez_compressed(
            out_path,
            centroids=out["centroids"],
            speaker_ids=out["speaker_ids"],
            embedding_dim=np.array(out["embedding_dim"]),
        )
        size_kb = out_path.stat().st_size / 1024
        print(f"  ✓ {out_path.name}: centroids={out['centroids'].shape} "
              f"({size_kb:.0f} KB), {out['n_train_files']} train files")
        manifest.append({
            "encoder": name,
            "centroids": out_path.name,
            "shape": list(out["centroids"].shape),
            "embedding_dim": out["embedding_dim"],
            "n_train_files": out["n_train_files"],
            "size_kb": round(size_kb, 1),
        })

    manifest_path = DATA_PROCESSED / "centroids_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"\n  ✓ Manifest saved to {manifest_path}")
    print("\n✅ Centroids built for all checkpoints.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
