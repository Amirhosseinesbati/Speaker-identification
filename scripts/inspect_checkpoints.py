"""
Inspect all checkpoints — verify they contain required keys and report metrics.

Usage:
    uv run --no-sync python scripts/inspect_checkpoints.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = PROJECT_ROOT / "checkpoints"


def inspect(path: Path) -> dict:
    """Load a checkpoint and report its contents without building the model."""
    print(f"\n  📦 {path.name} ({path.stat().st_size / 1e6:.0f} MB)")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    # Required keys
    required = ["model_state_dict", "config", "class_map"]
    for k in required:
        if k not in ckpt:
            print(f"     ❌ MISSING KEY: {k}")
        else:
            v = ckpt[k]
            if k == "model_state_dict":
                n_params = sum(p.numel() for p in v.values())
                print(f"     ✅ {k}: {len(v)} tensors, {n_params/1e6:.1f}M params")
            elif k == "config":
                enc = v.get("model", {}).get("encoder_type", "?")
                head = v.get("model", {}).get("speaker_head_type", "?")
                print(f"     ✅ {k}: encoder={enc}, speaker_head={head}")
            elif k == "class_map":
                print(f"     ✅ {k}: {len(v)} classes")

    # Optional metrics
    metrics = {k: v for k, v in ckpt.items() if k not in required}
    if metrics:
        print(f"     📊 Other keys: {list(metrics.keys())}")
        for k in ["epoch", "val_macro_f1", "val_loss", "val_ood_acc",
                   "val_speaker_acc", "ood_threshold"]:
            if k in ckpt:
                print(f"       {k} = {ckpt[k]}")

    return ckpt


def main():
    print("=" * 60)
    print("  Checkpoint Inspector")
    print("=" * 60)

    if not CKPT_DIR.exists():
        print(f"  ❌ {CKPT_DIR} not found")
        sys.exit(1)

    # Find all .pt files
    ckpts = sorted(CKPT_DIR.glob("*.pt"))
    if not ckpts:
        print("  ❌ No .pt files found in checkpoints/")
        sys.exit(1)

    print(f"  Found {len(ckpts)} checkpoint files\n")

    # Categorize
    best = [c for c in ckpts if "_best.pt" in c.name]
    latest = [c for c in ckpts if "_latest.pt" in c.name]
    other = [c for c in ckpts if c not in best and c not in latest]

    if best:
        print(f"\n── Best checkpoints ({len(best)}) ──")
        for c in best:
            inspect(c)

    if latest:
        print(f"\n── Latest checkpoints ({len(latest)}) ──")
        for c in latest:
            inspect(c)

    if other:
        print(f"\n── Other checkpoints ({len(other)}) ──")
        for c in other:
            inspect(c)

    # Summary table
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  {'Name':<25} {'Encoder':<12} {'Head':<10} {'Epoch':>5} "
          f"{'Macro-F1':>10} {'Params(M)':>10} {'Size(MB)':>10}")
    print("  " + "-" * 85)
    for c in ckpts:
        try:
            ckpt = torch.load(c, map_location="cpu", weights_only=False)
            cfg = ckpt.get("config", {})
            enc = cfg.get("model", {}).get("encoder_type", "?")
            head = cfg.get("model", {}).get("speaker_head_type", "?")
            epoch = ckpt.get("epoch", "?")
            mf1 = ckpt.get("val_macro_f1")
            mf1_str = f"{mf1:.4f}" if mf1 is not None else "N/A"
            n_params = sum(p.numel() for p in ckpt.get("model_state_dict", {}).values())
            size = c.stat().st_size / 1e6
            print(f"  {c.name:<25} {enc:<12} {head:<10} {str(epoch):>5} "
                  f"{mf1_str:>10} {n_params/1e6:>10.1f} {size:>10.0f}")
        except Exception as e:
            print(f"  {c.name:<25} ERROR: {e}")

    print("\n✅ Inspection complete.")


if __name__ == "__main__":
    main()
