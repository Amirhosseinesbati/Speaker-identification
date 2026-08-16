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


def _decode_keep_tail(buf: bytes) -> tuple:
    """Decode as much of ``buf`` as possible; return ``(text, leftover)``.

    A UTF-8 multibyte char may be split across pipe chunks — keep the trailing
    incomplete sequence for the next chunk instead of corrupting it.
    """
    try:
        return buf.decode("utf-8"), b""
    except UnicodeDecodeError:
        for cut in range(len(buf) - 1, 0, -1):
            try:
                return buf[:cut].decode("utf-8"), buf[cut:]
            except UnicodeDecodeError:
                continue
        return buf.decode("utf-8", "replace"), b""


def pump_pipe(stream, on_line, on_progress) -> None:
    """Split a binary pipe into newline lines and carriage-return updates.

    Reads ``stream`` (e.g. ``Popen(...).stdout.buffer``) until EOF and calls:

    - ``on_line(text)`` for every newline-terminated segment (LF or Windows
      CRLF). ``text`` is never empty. A line also terminates any in-flight
      ``\\r`` progress stream, so consumers should clear their progress slot.
    - ``on_progress(text)`` for every standalone ``\\r`` segment — i.e. a
      single-line progress update that overwrites the previous one (download
      progress, tqdm bars). ``text`` is never empty.

    Both callbacks make subprocess output resilient to `\\r`-based progress
    bars: the reader never blocks on a bar that omits newlines, and the UI can
    render the latest bar as one live line instead of log spam. Windows CRLF is
    normalised to LF, and a ``\\r\\n`` split across chunk boundaries is handled
    by carrying a trailing ``\\r`` over to the next chunk.
    """
    buf = b""
    pending_cr = False
    seg = []
    while True:
        chunk = stream.read(4096)
        if not chunk:
            break
        buf += chunk
        text, buf = _decode_keep_tail(buf)
        if pending_cr:
            text = "\r" + text
            pending_cr = False
        if text.endswith("\r"):
            pending_cr = True
            text = text[:-1]
        text = text.replace("\r\n", "\n")   # Windows CRLF → LF
        for c in text:
            if c == "\n":
                s = "".join(seg)
                seg = []
                if s:
                    on_line(s)
            elif c == "\r":
                s = "".join(seg)
                seg = []
                if s:
                    on_progress(s)
            else:
                seg.append(c)
    # Flush a final unterminated segment as a line (matches readline() EOF).
    s = "".join(seg)
    if s:
        on_line(s)

