"""
Robust WavLM-large downloader with resume + retry.

The network is flaky; a single snapshot_download often dies mid-file. This
wraps `curl -C -` (byte-range resume) in a retry loop until the file reaches
its full size.

Usage: python scripts/_download_wavlm.py
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "weights" / "wavlm_large"
URL = "https://huggingface.co/microsoft/wavlm-large/resolve/main/pytorch_model.bin"
DST = TARGET / "pytorch_model.bin"

# Full size of pytorch_model.bin (bytes) — verified via HEAD on the CDN.
FULL_SIZE = 1_261_990_257


def main():
    TARGET.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 200):
        have = DST.stat().st_size if DST.exists() else 0
        if have >= FULL_SIZE:
            print(f"✅ WavLM pytorch_model.bin complete ({have} bytes)")
            return
        print(f"⏳ attempt {attempt}: {have}/{FULL_SIZE} bytes "
              f"({100*have/FULL_SIZE:.1f}%)", flush=True)
        # curl -C - resumes from the current size; cap each attempt at 90s so
        # a stalled connection doesn't hang forever.
        r = subprocess.run(
            ["curl", "-sL", "-C", "-", "--max-time", "90", "--retry", "2",
             "-o", str(DST), URL],
        )
        time.sleep(2)
    raise SystemExit("Gave up downloading WavLM-large")


if __name__ == "__main__":
    main()
