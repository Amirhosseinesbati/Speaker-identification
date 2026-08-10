"""
Assemble the self-contained submission package.

Produces this structure under ``submission/``:

    submission/
    ├── src/                      (all project code)
    ├── configs/
    │   └── inference_config.yaml (local_path only, allow_hub_download: false)
    ├── weights/
    │   ├── ecapa/                (speechbrain savedir format)
    │   ├── campp/                (modelscope cache format)
    │   ├── eres2net/             (eres2netv2.ckpt)
    │   ├── titanet/              (titanet_large.nemo)
    │   └── wavlm_large/          (HF weights)
    ├── checkpoints/              (trained <enc>_best.pt — copied when present)
    ├── inference.py              (competition entrypoint)
    └── README.md                 (how to run, what each file is)

Usage:
    uv run --no-sync python scripts/build_submission.py [--skip-weights]

--skip-weights copies code + config only (fast iteration); the weights are
added when building the final zip.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUB = ROOT / "submission"

# Dirs to copy from the project root into the package.
SRC_DIRS = ["src"]
WEIGHT_DIRS = ["ecapa", "campp", "eres2net", "titanet", "wavlm_large"]


def _rm_artifacts(dst: Path) -> None:
    """Remove non-portable artifacts (pycache, deploy UI, logs)."""
    for p in dst.rglob("__pycache__"):
        shutil.rmtree(p, ignore_errors=True)


def build(skip_weights: bool) -> None:
    print(f"Building submission package in {SUB}")
    SUB.mkdir(parents=True, exist_ok=True)

    # ── src ──
    for d in SRC_DIRS:
        src = ROOT / d
        dst = SUB / d
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        _rm_artifacts(dst)
        print(f"  ✓ {d}/")
    # The deploy UI (Vast.ai/Streamlit) is not needed in the submission zip.
    shutil.rmtree(SUB / "src" / "deploy", ignore_errors=True)
    print("  ✓ src/deploy/ removed (not needed for inference)")

    # ── configs ──
    cfg_src = ROOT / "configs" / "inference_config.yaml"
    (SUB / "configs").mkdir(parents=True, exist_ok=True)
    shutil.copy2(cfg_src, SUB / "configs" / "inference_config.yaml")
    print("  ✓ configs/inference_config.yaml")

    # ── weights (idempotent — skip dirs already copied) ──
    if not skip_weights:
        (SUB / "weights").mkdir(parents=True, exist_ok=True)
        for w in WEIGHT_DIRS:
            src = ROOT / "weights" / w
            if not src.exists():
                print(f"  ⚠ weights/{w} missing — run scripts/download_all_weights.py")
                continue
            dst = SUB / "weights" / w
            if dst.exists() and any(dst.iterdir()):
                print(f"  ⏭  weights/{w}/ already present — skipping")
                continue
            shutil.copytree(src, dst)
            size = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file())
            print(f"  ✓ weights/{w}/ ({size/1e6:.0f} MB)")
    else:
        print("  ⏭  weights/ skipped (--skip-weights)")

    # ── checkpoints (trained best_*.pt) ──
    ckpt_src = ROOT / "checkpoints"
    if ckpt_src.exists():
        candidates = sorted(ckpt_src.glob("*_best.pt"))
        if candidates:
            (SUB / "checkpoints").mkdir(parents=True, exist_ok=True)
            for c in candidates:
                shutil.copy2(c, SUB / "checkpoints" / c.name)
                print(f"  ✓ checkpoints/{c.name}")
        else:
            print("  ⚠ no *_best.pt checkpoints found — train the 5 models first")
    else:
        print("  ⚠ checkpoints/ not found")

    # ── entrypoint ──
    for f in ("inference.py", "__init__.py"):
        src = ROOT / "submission" / f
        dst = SUB / f
        if src.resolve() == dst.resolve():
            print(f"  ✓ {f} (already in place)")
        else:
            shutil.copy2(src, dst)
            print(f"  ✓ {f}")

    # ── README ──
    readme = SUB / "README.md"
    if not readme.exists():
        shutil.copy2(ROOT / "submission" / "README.template.md", readme)
        print("  ✓ README.md")
    else:
        print("  ✓ README.md (existing kept)")

    print("\n✅ Submission package ready.")
    if not skip_weights:
        total = sum(f.stat().st_size for f in SUB.rglob("*") if f.is_file())
        print(f"   Total size: {total/1e6:.0f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build submission package")
    parser.add_argument("--skip-weights", action="store_true",
                        help="copy code+config only (fast iteration)")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    build(skip_weights=args.skip_weights)


if __name__ == "__main__":
    main()
