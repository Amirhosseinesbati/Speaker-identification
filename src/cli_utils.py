"""
Shared CLI helpers — UTF-8 stdio setup for Windows.

Windows consoles/pipes often use code page cp1252, which cannot encode the
emoji used across this project (⚠, ✅, …). When stdout is a pipe (e.g. when the
process is launched from the Streamlit UI with `capture_output=True`), Python
defaults to the locale encoding with strict errors, so any emoji raises
`UnicodeEncodeError` — and ZenML's logger wrapper (which re-emits stdout
writes) makes this surface even for a single `print()`.

Call `setup_utf8_stdio()` at the top of every entry point so output is always
UTF-8 with replace-on-error (never crashes), regardless of how the process was
launched.
"""

import sys


def setup_utf8_stdio() -> None:
    """Reconfigure stdout/stderr to UTF-8 with replace-on-error (no crashes).

    `line_buffering=True` also flushes every line immediately when stdout is a
    pipe, so Streamlit's LocalRunner can stream logs live (Python block-buffers
    piped output by default, which would hide all progress until the end).
    """
    for stream in (sys.stdout, sys.stderr):
        if stream is None:
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace",
                               line_buffering=True)
        except Exception:
            pass
