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

import http.client
import shutil
import ssl
import sys
import tarfile
import time
import urllib.request
import urllib.error
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
        noise.is_dir() and any(noise.rglob("*.wav"))
        and music.is_dir() and any(music.rglob("*.wav"))
    )


def rirs_present(path: Path) -> bool:
    return path.is_dir() and any(path.glob("*.wav"))


# ────────────────────────────────────────────────────────────────
#  Download helpers
# ────────────────────────────────────────────────────────────────

def _fmt_eta(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _fmt_gb(n: float) -> str:
    return f"{n / 1e9:.2f} GB"


# Transient network failures worth an automatic retry: TLS handshake stalls,
# socket read timeouts, mid-transfer disconnects and HTTP 5xx — all common on
# openslr.org. ``HTTPError`` (4xx/5xx) must be matched BEFORE ``URLError``
# because it subclasses it.
_TRANSIENT = (
    urllib.error.URLError,          # DNS / connect / TLS handshake / reset
    http.client.IncompleteRead,     # server dropped mid-transfer
    http.client.RemoteDisconnected,
    ConnectionError,                # reset / aborted / broken pipe
    TimeoutError,                   # == socket.timeout on py3.10+
    ssl.SSLError,
)


def _download_once(url: str, tmp: Path, timeout: float, name: str) -> None:
    """One download attempt, resuming from the current ``.part`` size.

    Raises on failure; everything already received is kept in ``tmp`` so the
    retry loop can resume instead of restarting.
    """
    resume_from = tmp.stat().st_size if tmp.exists() else 0
    headers = {"User-Agent": "Mozilla/5.0"}
    if resume_from:
        headers["Range"] = f"bytes={resume_from}-"
        print(f"  ⏮ resuming {name} from {_fmt_gb(resume_from)}")

    print(f"  📥 Downloading {url} (socket timeout {timeout:.0f}s)")
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        # A server that ignores Range answers 200 → restart from scratch.
        if resp.status == 200 and resume_from:
            resume_from = 0
        # Full size: Content-Range "bytes X-Y/TOTAL" beats Content-Length
        # (which only covers the remaining bytes of a 206 response).
        content_range = resp.headers.get("Content-Range") or ""
        if "/" in content_range:
            try:
                total = int(content_range.rsplit("/", 1)[1])
            except ValueError:
                total = 0
        else:
            total = int(resp.headers.get("Content-Length") or 0)
            if resume_from:
                total += resume_from

        downloaded = resume_from
        start = time.monotonic()
        last_print = 0.0
        last_pct = -1
        with open(tmp, "ab" if resume_from else "wb") as out:
            while True:
                try:
                    chunk = resp.read(256 * 1024)
                except http.client.IncompleteRead as e:
                    # Server closed mid-transfer — keep the bytes we already
                    # got so the retry resumes from the newest position.
                    out.write(e.partial)
                    raise
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                now = time.monotonic()
                pct = int(downloaded * 100 / total) if total > 0 else -1
                if (now - last_print >= 2.0
                        or (total > 0 and pct != last_pct and pct % 5 == 0)):
                    last_print = now
                    last_pct = pct
                    speed = downloaded / max(now - start, 1e-9)
                    if total > 0:
                        line = (f"  ⏳ {name}: {pct:3d}% "
                                f"({downloaded / 1e9:.2f}/{total / 1e9:.2f} GB)")
                    else:
                        line = f"  ⏳ {name}: {downloaded / 1e9:.2f} GB"
                    line += f" · {speed / 1e6:.1f} MB/s"
                    if total > 0 and speed > 0:
                        line += f" · ETA {_fmt_eta((total - downloaded) / speed)}"
                    sys.stdout.write("\r" + line.ljust(78))
                    sys.stdout.flush()
            # ``read(amt)`` returns b"" instead of raising when the server
            # closes early, so an interrupted transfer would silently "finish"
            # with a truncated file. Catch it explicitly — the retry resumes.
            if total > 0 and downloaded < total:
                raise ConnectionError(
                    f"connection closed early: got {downloaded} of {total} "
                    f"bytes")


def _download(url: str, dest: Path, timeout: float = 60.0,
              max_attempts: int = 6) -> Path:
    """Download ``url`` to ``dest``, resilient to a flaky/slow link.

    - RESUMES from the ``.part`` file via Range on every attempt, so nothing
      already transferred is ever thrown away (openslr.org is flaky and MUSAN
      is a ~10 GB archive — it WILL stall at least once).
    - RETRIES transient failures (TLS handshake stalls, socket read timeouts,
      mid-transfer disconnects, HTTP 5xx) with exponential backoff. Only a
      4xx answer or exhausting ``max_attempts`` gives up.
    - Progress is a SINGLE ``\\r``-updated line (download % · size · speed ·
      ETA), so a slow or stalled link is visible WITHOUT spamming the log. The
      UI's LocalRunner / experiment queue split ``\\r`` updates from ``\\n``
      lines, so it renders as one live progress bar; in a plain terminal it is
      the classic overwriting progress line. ``\\n`` is only emitted for
      attempt / retry / completion messages.
    - On final failure the partial file is KEPT; ``ensure_augmentation_data``
      turns the exception into the usual skip-with-warning and a re-run
      resumes from the kept bytes.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    if dest.exists():
        print(f"  ✅ {dest.name} already cached — skipping download.")
        return dest

    attempt = 0
    last_err = None
    while True:
        attempt += 1
        try:
            _download_once(url, tmp, timeout, dest.name)
            print()
            tmp.rename(dest)
            return dest
        except urllib.error.HTTPError as e:
            if e.code < 500:  # 4xx — permanent, retrying won't help
                raise RuntimeError(
                    f"download of {dest.name} failed: HTTP {e.code} "
                    f"({e.reason})") from e
            last_err = e
        except _TRANSIENT as e:
            last_err = e
        # Any other exception propagates uncaught — no retry for local bugs.

        if attempt >= max_attempts:
            kept = tmp.stat().st_size if tmp.exists() else 0
            print(f"  ⚠ giving up after {attempt} attempts "
                  f"({_fmt_gb(kept)} kept at {tmp.name})")
            raise RuntimeError(
                f"download of {dest.name} failed after {attempt} attempts: "
                f"{last_err} (re-run resumes from {_fmt_gb(kept)})") from last_err

        kept = tmp.stat().st_size if tmp.exists() else 0
        delay = min(5 * 2 ** (attempt - 1), 60)   # 5, 10, 20, 40, 60, 60…
        print(f"  🔁 attempt {attempt}/{max_attempts} failed "
              f"({_fmt_gb(kept)} so far) — retrying in {delay}s", flush=True)
        time.sleep(delay)


def _extract_musan(archive: Path, base: Path) -> None:
    base.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tf:
        members = [m for m in tf.getmembers()
                   if m.isfile()
                   and (m.name.startswith("musan/noise/")
                        or m.name.startswith("musan/music/"))]
        total = len(members)
        start = time.monotonic()
        last_print = 0.0
        for i, m in enumerate(members, 1):
            rel = Path(m.name).relative_to("musan")   # noise/… or music/…
            dest = base / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            with tf.extractfile(m) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)
            if time.monotonic() - last_print >= 2.0 or i == total:
                last_print = time.monotonic()
                line = (f"  ⏳ extracting {dest.name} ({i}/{total}) "
                        f"· {i * 100 // total:3d}% · "
                        f"{i / max(time.monotonic() - start, 1e-9):.0f} files/s")
                sys.stdout.write("\r" + line.ljust(78))
                sys.stdout.flush()
    print(f"  ✅ MUSAN noise/music extracted to {base} ({total} files)")


def _extract_rirs(archive: Path, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    last_print = 0.0
    n = 0
    with zipfile.ZipFile(archive) as zf:
        names = [n for n in zf.namelist()
                 if n.startswith("RIRS_NOISES/simulated_rirs/")
                 and n.lower().endswith(".wav")]
        total = len(names)
        for name in names:
            # Flatten into the target dir — audiomentations AddImpulseResponse
            # reads a flat folder of .wav files.
            (path / Path(name).name).write_bytes(zf.read(name))
            n += 1
            if time.monotonic() - last_print >= 2.0 or n == total:
                last_print = time.monotonic()
                line = (f"  ⏳ extracting {Path(name).name} ({n}/{total}) "
                        f"· {n * 100 // total:3d}% · "
                        f"{n / max(time.monotonic() - start, 1e-9):.0f} files/s")
                sys.stdout.write("\r" + line.ljust(78))
                sys.stdout.flush()
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
