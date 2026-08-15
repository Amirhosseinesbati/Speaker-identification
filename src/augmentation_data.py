"""
Auto-download MUSAN + RIR augmentation data (idempotent).

The domain augmentation (MUSAN noise/music, RIR reverb) needs large external
datasets that are NOT committed. Previously a run would silently SKIP them when
the data dir was missing (a "download first" warning deep in training). For an
automated remote run that is pure poison — the user toggles the option in the
UI and the job must fetch the data itself.

``ensure_augmentation_data(config)`` is called at the top of the data-prep
pipeline step on ANY machine (local or Vast.ai): for each enabled domain aug it
checks the dir and, if absent, downloads + extracts it. Idempotent and non-fatal
(download failure degrades to the old skip-with-warning behaviour).

Downloads:
    MUSAN  → https://www.openslr.org/resources/17/musan.tar.gz  (noise + music)
    RIRs   → https://www.openslr.org/resources/28/rirs_noises.zip (simulated RIRs)
"""

from __future__ import annotations

import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUG_ROOT = PROJECT_ROOT / "data" / "augmentation"
CACHE_DIR = AUG_ROOT / "_cache"

MUSAN_URL = "https://www.openslr.org/resources/17/musan.tar.gz"
RIRS_URL = "https://www.openslr.org/resources/28/rirs_noises.zip"


# ────────────────────────────────────────────────────────────────
#  Presence checks (idempotency)
# ────────────────────────────────────────────────────────────────

def musan_present(base: Path) -> bool:
    noise = base / "noise"
    music = base / "music"
    return (
        noise.is_dir() and any(noise.glob("*.wav"))
        and music.is_dir() and any(music.glob("*.wav"))
    )


def rirs_present(path: Path) -> bool:
    return path.is_dir() and any(path.glob("*.wav"))


# ────────────────────────────────────────────────────────────────
#  Download helpers
# ────────────────────────────────────────────────────────────────

def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    if dest.exists():
        print(f"  ✅ {dest.name} already cached — skipping download.")
        return dest

    print(f"  📥 Downloading {url}")
    last_pct = [-1]

    def _hook(blocks: int, block_size: int, total: int) -> None:
        if total <= 0:
            return
        pct = int(blocks * block_size * 100 / total)
        if pct != last_pct[0] and pct % 5 == 0:
            last_pct[0] = pct
            print(f"  ⏳ {dest.name}: {pct:3d}% "
                  f"({blocks * block_size / 1e9:.2f}/{total / 1e9:.2f} GB)")

    urllib.request.urlretrieve(url, tmp, reporthook=_hook)
    print()
    tmp.rename(dest)
    return dest


def _extract_musan(archive: Path, base: Path) -> None:
    base.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tf:
        members = [m for m in tf.getmembers()
                   if m.isfile()
                   and (m.name.startswith("musan/noise/")
                        or m.name.startswith("musan/music/"))]
        for m in members:
            rel = Path(m.name).relative_to("musan")   # noise/… or music/…
            dest = base / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            with tf.extractfile(m) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)
    print(f"  ✅ MUSAN noise/music extracted to {base} ({len(members)} files)")


def _extract_rirs(archive: Path, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(archive) as zf:
        names = [n for n in zf.namelist()
                 if n.startswith("RIRS_NOISES/simulated_rirs/")
                 and n.lower().endswith(".wav")]
        for name in names:
            # Flatten into the target dir — audiomentations AddImpulseResponse
            # reads a flat folder of .wav files.
            (path / Path(name).name).write_bytes(zf.read(name))
            n += 1
    print(f"  ✅ RIRs extracted to {path} ({n} files)")


# ────────────────────────────────────────────────────────────────
#  Public API
# ────────────────────────────────────────────────────────────────

def _prob(block, default: float = 0.0) -> float:
    if not isinstance(block, dict):
        return default
    try:
        return float(block.get("p", default) or 0)
    except (TypeError, ValueError):
        return default


def download_musan(base: Optional[Path] = None) -> Path:
    base = base or AUG_ROOT / "musan"
    archive = _download(MUSAN_URL, CACHE_DIR / "musan.tar.gz")
    _extract_musan(archive, base)
    return base


def download_rirs(path: Optional[Path] = None) -> Path:
    path = path or AUG_ROOT / "rirs"
    archive = _download(RIRS_URL, CACHE_DIR / "rirs_noises.zip")
    _extract_rirs(archive, path)
    return path


def ensure_augmentation_data(config: dict) -> dict:
    """Download any enabled-but-missing domain augmentation data.

    Returns a small summary dict (``musan`` / ``rirs`` → downloaded|present|off|failed).
    Never raises — a download failure degrades to the old skip-with-warning.
    """
    domain = (config.get("augmentation", {}) or {}).get("domain", {}) or {}
    summary = {}

    # ── MUSAN (noise_p / music_p) ──
    musan = domain.get("musan", {}) or {}
    musan_on = (float(musan.get("noise_p", 0) or 0) > 0
                or float(musan.get("music_p", 0) or 0) > 0)
    if musan_on:
        base = Path(musan.get("path", "data/augmentation/musan"))
        if musan_present(base):
            summary["musan"] = "present"
        else:
            try:
                print("  🔽 MUSAN enabled but missing — downloading…")
                download_musan(base)
                summary["musan"] = "downloaded"
            except Exception as e:
                print(f"  ⚠ MUSAN download failed ({e}) — noise/music aug will be skipped.")
                summary["musan"] = "failed"
    else:
        summary["musan"] = "off"

    # ── RIR reverb (p) ──
    rir = domain.get("rirs_reverb", {}) or {}
    if _prob(rir) > 0:
        path = Path(rir.get("path", "data/augmentation/rirs"))
        if rirs_present(path):
            summary["rirs"] = "present"
        else:
            try:
                print("  🔽 RIR enabled but missing — downloading…")
                download_rirs(path)
                summary["rirs"] = "downloaded"
            except Exception as e:
                print(f"  ⚠ RIR download failed ({e}) — reverb aug will be skipped.")
                summary["rirs"] = "failed"
    else:
        summary["rirs"] = "off"

    print(f"  Augmentation data: {summary}")
    return summary


# ────────────────────────────────────────────────────────────────
#  CLI
# ────────────────────────────────────────────────────────────────

def main() -> int:
    import argparse

    from src.cli_utils import setup_utf8_stdio
    setup_utf8_stdio()

    parser = argparse.ArgumentParser(description="Download MUSAN + RIR augmentation data")
    parser.add_argument("--musan", action="store_true", help="Download MUSAN only")
    parser.add_argument("--rirs", action="store_true", help="Download RIRs only")
    args = parser.parse_args()

    do_all = not args.musan and not args.rirs
    if do_all or args.musan:
        download_musan()
    if do_all or args.rirs:
        download_rirs()
    print("\n✅ Augmentation data ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
