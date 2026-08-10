"""
Download all 5 encoder weights into the local `weights/` directory (dev only).

This script runs on the DEV machine (full internet access). Each download is
idempotent — if the target already exists it is skipped, so re-running is safe.

Layout produced (matches the offline inference convention):

    weights/
    ├── ecapa/          SpeechBrain savedir: hyperparams.yaml + model.ckpt + normalizer.ckpt
    ├── campp/          ModelScope cache (pipeline-importable via MODELSCOPE_CACHE)
    ├── eres2net/       ModelScope cache (pipeline-importable via MODELSCOPE_CACHE)
    ├── titanet/        titanet_large.nemo
    └── wavlm_large/    HF safetensors: config.json + model.safetensors + preprocessor_config.json

Usage:
    uv run --no-sync python scripts/download_all_weights.py [--force]

Note: the venv currently ships CPU-only torch — model *weights* download fine,
but do NOT try to run inference in this venv; use the Vast.ai / GPU venv for
Phase 2 tests.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"


# ─────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────

def _marker(name: str) -> str:
    """Human-readable skip marker."""
    return f"  ⏭️  {name}: already present — skipping"


def _done(name: str, path: Path) -> None:
    """Human-readable completion marker."""
    print(f"  ✅ {name}: {path}")


# ─────────────────────────────────────────────────────────
#  Per-model downloaders (each is idempotent)
# ─────────────────────────────────────────────────────────

def download_ecapa(force: bool = False) -> Path:
    """
    SpeechBrain ECAPA-TDNN (VoxCeleb) — spkrec-ecapa-voxceleb.

    `EncoderClassifier.from_hparams(source=..., savedir=local_dir)` copies the
    full model (hyperparams.yaml + model.ckpt + normalizer.ckpt) into the local
    dir, so inference can later instantiate from that dir with no hub fetch.
    `local_strategy=COPY` avoids symlinks (Windows-compatible + zip-portable).
    """
    target = WEIGHTS_DIR / "ecapa"
    marker = target / "hyperparams.yaml"
    if marker.exists() and not force:
        print(_marker("ECAPA-TDNN"))
        return target

    print("  ⬇️  Downloading ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb)...")
    target.mkdir(parents=True, exist_ok=True)

    # Neutralise SpeechBrain lazy-module breakage (same patch as src/encoders.py).
    from src.encoders import _patch_speechbrain_lazy_modules

    import speechbrain  # noqa: F401  (must import first so lazy modules register)
    _patch_speechbrain_lazy_modules()
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy

    EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(target),
        run_opts={"device": "cpu"},
        local_strategy=LocalStrategy.COPY,
    )
    _done("ECAPA-TDNN", target)
    return target


def download_campp(force: bool = False) -> Path:
    """
    CAM++ speaker verification (ModelScope) — iic/speech_campplus_sv_en_voxceleb_16k.

    Downloads the full ModelScope model snapshot into the local cache. At
    inference we point MODELSCOPE_CACHE at this dir so the pipeline reads from
    cache with no network.
    """
    target = WEIGHTS_DIR / "campp"
    marker = target / "configuration.json"
    if marker.exists() and not force:
        print(_marker("CAM++"))
        return target

    print("  ⬇️  Downloading CAM++ (iic/speech_campplus_sv_en_voxceleb_16k)...")
    target.mkdir(parents=True, exist_ok=True)

    os.environ["MODELSCOPE_CACHE"] = str(target)
    from modelscope import snapshot_download

    snapshot_download(
        "iic/speech_campplus_sv_en_voxceleb_16k",
        revision="v1.0.2",
        cache_dir=str(target),
    )
    _done("CAM++", target)
    return target


def download_eres2net(force: bool = False) -> Path:
    """
    ERes2NetV2 speaker verification — official 192-dim VoxCeleb checkpoint.

    The architecture is VENDORED in src/sv_arch.py (torch + torchaudio only),
    so the model needs NO modelscope / 3dspeaker package. The official
    release is mirrored on HuggingFace (`bandad/eres2netv2_pretrained`) —
    verified in Phase 2c: strict state_dict load succeeds (17.86 M params).

    Offline layout: weights/eres2net/eres2netv2.ckpt
    """
    target = WEIGHTS_DIR / "eres2net"
    ckpt = target / "eres2netv2.ckpt"
    if ckpt.exists() and not force:
        print(_marker("ERes2NetV2"))
        return target

    print("  ⬇️  Downloading ERes2NetV2 (bandad/eres2netv2_pretrained)...")
    target.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import hf_hub_download

    src = hf_hub_download(
        "bandad/eres2netv2_pretrained",
        "pretrained_eres2netv2.ckpt",
        local_dir=str(target / ".src"),
    )
    import shutil
    shutil.copyfile(src, str(ckpt))
    _done("ERes2NetV2", ckpt)
    return target


def download_titanet(force: bool = False) -> Path:
    """
    NeMo TitaNet-Large — nvidia/speakerverification_en_titanet_large.

    Downloads the .nemo artifact and saves it as weights/titanet/titanet_large.nemo.
    At inference: EncDecSpeakerLabelModel.restore_from(<local>.nemo) — no hub.
    """
    target = WEIGHTS_DIR / "titanet"
    nemo_file = target / "titanet_large.nemo"
    if nemo_file.exists() and not force:
        print(_marker("TitaNet-Large"))
        return target

    print("  ⬇️  Downloading TitaNet-Large (nvidia/speakerverification_en_titanet_large)...")
    target.mkdir(parents=True, exist_ok=True)

    from nemo.collections.asr.models import EncDecSpeakerLabelModel

    model = EncDecSpeakerLabelModel.from_pretrained(
        "nvidia/speakerverification_en_titanet_large",
        map_location="cpu",
    )
    model.save_to(str(nemo_file))
    _done("TitaNet-Large", nemo_file)
    return target


def download_wavlm_large(force: bool = False) -> Path:
    """
    WavLM-Large (HuggingFace) — microsoft/wavlm-large.

    snapshots the model into weights/wavlm_large/. At inference:
    WavLMModel.from_pretrained(<dir>, local_files_only=True) — no hub fetch.
    """
    target = WEIGHTS_DIR / "wavlm_large"
    marker = target / "model.safetensors"
    if marker.exists() and not force:
        print(_marker("WavLM-Large"))
        return target

    print("  ⬇️  Downloading WavLM-Large (microsoft/wavlm-large)...")
    target.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download

    snapshot_download(
        "microsoft/wavlm-large",
        local_dir=str(target),
    )
    _done("WavLM-Large", target)
    return target


# ─────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download all 5 encoder weights into weights/ (idempotent)."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if a marker file already exists.",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(PROJECT_ROOT))
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Download all encoder weights (dev machine, online)")
    print("=" * 60)
    print(f"  Output dir: {WEIGHTS_DIR}")

    # Order matters: ECAPA first (fast, small), then the rest.
    download_ecapa(force=args.force)
    download_campp(force=args.force)
    download_eres2net(force=args.force)
    download_titanet(force=args.force)
    download_wavlm_large(force=args.force)

    print("\n  All weights downloaded. Verify with:")
    print(f"    find {WEIGHTS_DIR} -type f | head -50")


if __name__ == "__main__":
    main()
