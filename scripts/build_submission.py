"""
Assemble the self-contained submission package.

Produces this structure under ``submission/`` (which becomes the ZIP ROOT):

    submission/                 (zip root)
    ├── submission.py           (competition entry point — required name)
    ├── inference.py            (full-featured implementation)
    ├── __init__.py
    ├── src/                    (all project code, no deploy/ UI)
    ├── configs/
    │   └── inference_config.yaml (local_path only, allow_hub_download: false)
    ├── weights/                (ONLY encoders that have a trained checkpoint)
    │   ├── ecapa/              (speechbrain savedir format)
    │   ├── campp/              (modelscope cache format)
    │   ├── eres2net/           (eres2netv2.ckpt)
    │   └── titanet/            (titanet_large.nemo)
    ├── checkpoints/            (trained <enc>_best.pt — one per used encoder)
    ├── ensemble_fusion_weights.json  (best fusion config for the leaderboard)
    └── README.md               (how to run, what each file is)

Rules:
  - Only weights whose encoder has a ``*_best.pt`` checkpoint are shipped
    (so the untrained 1.2 GB wavlm_large weights are excluded automatically).
  - Stale demo files (sample_predictions.csv / .class_map.json) are removed.
  - ``__pycache__`` is removed everywhere.

Usage:
    uv run --no-sync python scripts/build_submission.py [--skip-weights]

--skip-weights copies code + config + checkpoints only (fast iteration); the
weights are added when building the final package.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUB = ROOT / "submission"

SRC_DIRS = ["src"]
ALL_WEIGHT_DIRS = ["ecapa", "campp", "eres2net", "titanet", "wavlm_large"]
ENTRYPOINT_FILES = ["submission.py", "inference.py", "__init__.py"]


def _rm_artifacts(dst: Path) -> None:
    """Remove non-portable artifacts (pycache, deploy UI, logs, stale demos)."""
    for p in dst.rglob("__pycache__"):
        shutil.rmtree(p, ignore_errors=True)
    # Stale demo outputs from earlier iterations — never ship these.
    for stale in ("sample_predictions.csv", "sample_predictions.class_map.json"):
        (dst / stale).unlink(missing_ok=True)


def _encoder_from_checkpoint_name(name: str) -> str:
    """'ecapa_best.pt' → 'ecapa'."""
    return name.replace("_best.pt", "")


# ModelScope 1.39+ declares REDUCED wheel metadata: its runtime imports
# (addict, easydict, simplejson, yapf) are NOT auto-installed and are NOT in
# the leaderboard package list. Vendor them (pure-Python, no compiled ext) so
# modelscope imports work in the evaluation env.
VENDORED_DEPS = ["addict", "easydict", "simplejson", "yapf"]


def _vendor_deps() -> None:
    import importlib

    dst = SUB / "vendor"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    for name in VENDORED_DEPS:
        try:
            mod = importlib.import_module(name)
        except ImportError:
            print(f"  ⚠ vendor/{name} not installed in the build env — "
                  f"modelscope will fail on the leaderboard. "
                  f"Run `uv sync` first.")
            continue
        src = Path(mod.__file__).resolve().parent
        out = dst / name
        out.mkdir(parents=True)
        n = 0
        for py in src.rglob("*.py"):
            rel = py.relative_to(src)
            target = out / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(py, target)
            n += 1
        print(f"  ✓ vendor/{name} ({n} py files)")

    compiled = list(dst.rglob("*.pyd")) + list(dst.rglob("*.so")) + list(dst.rglob("*.dll"))
    if compiled:
        for f in compiled:
            f.unlink()
        print(f"  🧹 removed {len(compiled)} compiled ext(s) from vendor/")


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
    # Non-inference modules (EDA, ZenML pipelines, MLflow) import heavy deps
    # that are NOT on the leaderboard (zenml/mlflow/streamlit/...) — prune them
    # so the shipped src/ can never trigger an import error at eval time.
    for dead in ("src/pipelines", "src/mlflow_helper.py"):
        p = SUB / dead
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)
        print(f"  ✓ {dead} removed (not needed for inference)")
    for f in list((SUB / "src").glob("eda*.py")):
        f.unlink(missing_ok=True)
        print(f"  ✓ {f.name} removed (EDA only)")
    # Training / calibration / offline-analysis modules are dead weight on the
    # leaderboard (and some import deps absent from the server, e.g. faiss).
    # Checkpoints embed config + class_map, so none of these run at eval time.
    for dead in (
        "src/audio_preprocessing.py",
        "src/centroid_baseline.py",
        "src/data_pipeline.py",
        "src/ensemble_calibrate.py",
        "src/metrics.py",
        "src/ood_detector.py",
        "src/train.py",
    ):
        (SUB / dead).unlink(missing_ok=True)
        print(f"  ✓ {dead} removed (training/calibration only)")
    # The repo src/ensemble.py carries training/calibration tooling; ship only
    # the stateless fusion functions the leaderboard actually calls.
    shutil.copy2(ROOT / "scripts" / "minimal_ensemble.py",
                 SUB / "src" / "ensemble.py")
    print("  ✓ src/ensemble.py replaced with minimal fusion-only version")

    # ── vendored deps (modelscope transitive imports missing on the leaderboard) ──
    _vendor_deps()

    # ── checkpoints ──
    ckpt_src = ROOT / "checkpoints"
    # Skip encoders whose fusion weight is ~0 — they contribute nothing to the
    # ensemble output, so shipping their checkpoint + weights only bloats the
    # package (e.g. a 129 MB ecapa_best.pt that is never loaded at eval time).
    fw_src = ROOT / "data" / "processed" / "ensemble_fusion_weights.json"
    active_encoders = None
    if fw_src.exists():
        try:
            fw = json.loads(fw_src.read_text(encoding="utf-8"))
            names = fw.get("encoder_names", [])
            weights = fw.get("weights", [])
            active_encoders = {
                names[i] for i, w in enumerate(weights)
                if i < len(names) and w > 1e-8
            }
        except Exception:
            active_encoders = None

    ckpt_files = sorted(ckpt_src.glob("*_best.pt")) if ckpt_src.exists() else []
    if active_encoders is not None:
        skipped = [c for c in ckpt_files
                   if _encoder_from_checkpoint_name(c.name) not in active_encoders]
        ckpt_files = [c for c in ckpt_files
                      if _encoder_from_checkpoint_name(c.name) in active_encoders]
        for c in skipped:
            print(f"  ⏭  checkpoints/{c.name} skipped (zero fusion weight)")
    # Rebuild checkpoints/ from scratch: remove stale *.pt from earlier builds
    # (e.g. eres2net/titanet left over after switching to a single-model
    # ensemble) so the package never ships dead weight.
    ckpt_dst = SUB / "checkpoints"
    ckpt_dst.mkdir(parents=True, exist_ok=True)
    wanted = {c.name for c in ckpt_files}
    for stale in list(ckpt_dst.glob("*.pt")):
        if stale.name not in wanted:
            stale.unlink(missing_ok=True)
            print(f"  🧹 checkpoints/{stale.name} removed (stale)")
    if ckpt_files:
        for c in ckpt_files:
            shutil.copy2(c, ckpt_dst / c.name)
            print(f"  ✓ checkpoints/{c.name}")
    else:
        print("  ⚠ no *_best.pt checkpoints found — train the models first")

    # ── weights (only for encoders that have a trained checkpoint) ──
    used_encoders = {_encoder_from_checkpoint_name(c.name) for c in ckpt_files}
    if not skip_weights:
        (SUB / "weights").mkdir(parents=True, exist_ok=True)
        # Prune stale weight dirs from earlier builds (e.g. wavlm_large with no
        # checkpoint) so the package never ships dead weight.
        for stale in list((SUB / "weights").iterdir()):
            if stale.is_dir() and stale.name not in used_encoders:
                shutil.rmtree(stale, ignore_errors=True)
                print(f"  🧹 weights/{stale.name}/ pruned (no {stale.name}_best.pt)")
        for w in ALL_WEIGHT_DIRS:
            if w not in used_encoders:
                print(f"  ⏭  weights/{w}/ skipped (no {w}_best.pt checkpoint)")
                continue
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

    # ── entrypoint ──
    for f in ENTRYPOINT_FILES:
        src = ROOT / "submission" / f
        dst = SUB / f
        if src.resolve() == dst.resolve():
            print(f"  ✓ {f} (already in place)")
        else:
            shutil.copy2(src, dst)
            print(f"  ✓ {f}")

    # ── fusion config (best weights for the leaderboard) ──
    fw_src = ROOT / "data" / "processed" / "ensemble_fusion_weights.json"
    if fw_src.exists():
        shutil.copy2(fw_src, SUB / "ensemble_fusion_weights.json")
        print("  ✓ ensemble_fusion_weights.json")
    else:
        print("  ⚠ data/processed/ensemble_fusion_weights.json missing — "
              "run ensemble_calibrate.py first")

    # ── centroids (cosine centroid + OOD-gate decision layer) ──
    cent_dst = SUB / "centroids"
    cent_dst.mkdir(parents=True, exist_ok=True)
    wanted_npz = {f"centroids_{enc}.npz" for enc in used_encoders}
    for stale in list(cent_dst.glob("*.npz")):
        if stale.name not in wanted_npz:
            stale.unlink(missing_ok=True)
            print(f"  🧹 centroids/{stale.name} removed (stale)")
    shipped_centroids = 0
    for enc in sorted(used_encoders):
        src = ROOT / "data" / "processed" / f"centroids_{enc}.npz"
        if src.exists():
            shutil.copy2(src, cent_dst / src.name)
            shipped_centroids += 1
        else:
            print(f"  ⚠ centroids_{enc}.npz missing — run scripts/build_centroids.py "
                  f"(decision layer falls back to plain argmax for {enc})")

    # ── closed-set 1000-class experiment artifacts ──
    # Ship EVERY pseudo-identity map + cluster centroid that exists. Maps and
    # centroids are k-locked (unknown_clusters_k<k>.json /
    # centroids_unknown_<enc>_k<k>.npz, plus the plain "active" aliases), so a
    # shipped checkpoint finds its exact k's map + centroid space at eval time
    # regardless of how many k experiments coexist in data/processed.
    shipped_maps = 0
    for cm_src in sorted((ROOT / "data" / "processed").glob("unknown_clusters*.json")):
        shutil.copy2(cm_src, SUB / cm_src.name)
        shipped_maps += 1
        print(f"  ✓ {cm_src.name} (1000-class experiment)")
    if shipped_maps == 0:
        print("  ℹ no pseudo-identity maps — legacy 447-way model")
    shipped_cluster_centroids = 0
    for enc in sorted(used_encoders):
        for src in sorted((ROOT / "data" / "processed").glob(
                f"centroids_unknown_{enc}*.npz")):
            shutil.copy2(src, cent_dst / src.name)
            shipped_cluster_centroids += 1
            print(f"  ✓ centroids/{src.name}")
    if shipped_cluster_centroids == 0:
        print("  ℹ no unknown-cluster centroids — legacy 447-way decision layer")
    if shipped_centroids == 0:
        print("  ⚠ no centroids shipped — decision layer disabled")
    else:
        print(f"  ✓ centroids/ ({shipped_centroids} encoder(s))")

    # ── decision config (τ/α/κ/λ/T tuned on val) ──
    dc_src = ROOT / "data" / "processed" / "decision_config.json"
    if dc_src.exists():
        shutil.copy2(dc_src, SUB / "decision_config.json")
        print("  ✓ decision_config.json")
    else:
        print("  ⚠ decision_config.json missing — run scripts/tune_decision.py "
              "(plain argmax fallback)")

    # ── README ──
    readme = SUB / "README.md"
    if readme.exists():
        print("  ✓ README.md (committed source, kept)")
    else:
        print("  ⚠ README.md missing in submission/ — add it (source file)")

    # ── cleanup + summary ──
    _rm_artifacts(SUB)
    print("\n✅ Submission package ready.")
    total = sum(f.stat().st_size for f in SUB.rglob("*") if f.is_file())
    print(f"   Total size: {total/1e6:.0f} MB")

    # Sanity checklist
    print("\n── Submission checklist ──")
    for f in ("submission.py", "inference.py", "src/encoders.py",
              "src/ensemble.py", "ensemble_fusion_weights.json",
              "checkpoints"):
        p = SUB / f
        print(f"  {'✓' if p.exists() else '✗ MISSING'} {f}")
    for w in used_encoders:
        p = SUB / "weights" / w
        print(f"  {'✓' if p.exists() and any(p.iterdir()) else '✗ MISSING'} weights/{w}/")
    cdir = SUB / "centroids"
    print(f"  {'✓' if cdir.exists() and any(cdir.iterdir()) else '✗ MISSING'} centroids/")
    dc = SUB / "decision_config.json"
    print(f"  {'✓' if dc.exists() else '⚠ absent (plain argmax)'} decision_config.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build submission package")
    parser.add_argument("--skip-weights", action="store_true",
                        help="copy code+config only (fast iteration)")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    build(skip_weights=args.skip_weights)


if __name__ == "__main__":
    main()
