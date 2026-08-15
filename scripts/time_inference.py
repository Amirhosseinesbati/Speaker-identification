"""
Q7 — Inference timing harness (anti-timeout guard).

Runs the real ``score_ensemble`` path (centroid + OOD-gate decision layer) on a
sample of real audio files, times it, and extrapolates to the full 3,604-file
leaderboard test set (×36). Any feature that pushes the extrapolation past the
~20-minute budget must be rejected or made faster BEFORE uploading.

Usage:
    # sample 100 files (default) from data/raw, extrapolate ×36
    uv run --no-sync python scripts/time_inference.py

    # custom sample + the leaderboard venv python (to mirror the eval env)
    uv run --no-sync python scripts/time_inference.py --n-files 100 --audio-dir data/raw
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.cli_utils import setup_utf8_stdio  # noqa: E402
setup_utf8_stdio()

FUSION_JSON = ROOT / "data" / "processed" / "ensemble_fusion_weights.json"
DECISION_JSON = ROOT / "data" / "processed" / "decision_config.json"
CENTROIDS_DIR = ROOT / "data" / "processed"
CKPT_DIR = ROOT / "checkpoints"
FULL_TEST_SIZE = 3604
BUDGET_SECONDS = 20 * 60


def discover_checkpoints() -> list:
    order = None
    if FUSION_JSON.exists():
        try:
            order = json.loads(FUSION_JSON.read_text(encoding="utf-8")).get("encoder_names")
        except Exception:
            order = None
    if order:
        return [str(CKPT_DIR / f"{enc}_best.pt") for enc in order
                if (CKPT_DIR / f"{enc}_best.pt").exists()]
    return sorted(str(p) for p in CKPT_DIR.glob("*_best.pt"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Time the submission inference path")
    parser.add_argument("--n-files", type=int, default=100)
    parser.add_argument("--audio-dir", default=str(ROOT / "data" / "raw"))
    args = parser.parse_args()

    import numpy as np

    from submission.inference import score_ensemble, load_centroids

    checkpoints = discover_checkpoints()
    fusion_weights = json.loads(FUSION_JSON.read_text(encoding="utf-8"))["weights"]

    encoder_names = [Path(c).name.replace("_best.pt", "") for c in checkpoints]
    centroids = load_centroids(str(CENTROIDS_DIR), encoder_names) or None
    decision_params = None
    if DECISION_JSON.exists():
        decision_params = json.loads(DECISION_JSON.read_text(encoding="utf-8")).get(
            "decision_params")

    audio_dir = Path(args.audio_dir)
    files = sorted(p for p in audio_dir.iterdir()
                   if p.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg", ".m4a"})
    if len(files) > args.n_files:
        step = max(1, len(files) // args.n_files)
        files = files[::step][:args.n_files]
    if not files:
        print(f"  ⚠ No audio files in {audio_dir}")
        return 1

    # Copy the sample into a scratch dir (score_ensemble reads a directory).
    import tempfile, shutil
    scratch = Path(tempfile.mkdtemp(prefix="time_inf_"))
    for f in files:
        shutil.copy2(f, scratch / f.name)

    print("=" * 60)
    print("  Q7 — Inference timing (decision layer enabled)")
    print("=" * 60)
    print(f"  Files: {len(files)} | checkpoints: {len(checkpoints)}")
    print(f"  centroids: {'yes' if centroids else 'no'} | "
          f"decision_params: {'yes' if decision_params else 'no'}")

    # Warmup + fixed-overhead measurement on a single file: model loading,
    # CUDA context / cudnn-benchmark warmup are one-time costs that must NOT be
    # amortised into the per-file estimate.
    warm_dir = Path(tempfile.mkdtemp(prefix="time_inf_warm_"))
    shutil.copy2(files[0], warm_dir / files[0].name)
    t0 = time.time()
    score_ensemble(data_dir=str(warm_dir), checkpoint_path=checkpoints,
                   fusion_method="weighted_average", fusion_weights=fusion_weights,
                   centroids=centroids, decision_params=decision_params)
    overhead = time.time() - t0
    shutil.rmtree(warm_dir, ignore_errors=True)

    t0 = time.time()
    result = score_ensemble(
        data_dir=str(scratch),
        checkpoint_path=checkpoints,
        fusion_method="weighted_average",
        fusion_weights=fusion_weights,
        centroids=centroids,
        decision_params=decision_params,
    )
    elapsed = time.time() - t0

    # Per-file cost = (second run − warmup run) / (n_files − 1). The warmup run
    # also processes 1 file, so subtracting it removes both loading AND one
    # file's inference from the estimate.
    per_file = (elapsed - overhead) / (len(files) - 1)
    extrapolated = per_file * FULL_TEST_SIZE + overhead
    print(f"\n  Warmup (loading + 1 file): {overhead:.1f}s")
    print(f"  Elapsed: {elapsed:.1f}s for {len(files)} files "
          f"({per_file * 1000:.0f} ms/file, clean)")
    print(f"  Extrapolated to {FULL_TEST_SIZE} files: {extrapolated / 60:.1f} min "
          f"(budget {BUDGET_SECONDS / 60:.0f} min)")
    verdict = "✅ OK" if extrapolated <= BUDGET_SECONDS else "❌ OVER BUDGET"
    print(f"  Verdict: {verdict}")

    shutil.rmtree(scratch, ignore_errors=True)
    return 0 if extrapolated <= BUDGET_SECONDS else 2


if __name__ == "__main__":
    sys.exit(main())
