"""Pure selection logic shared by the GPU batch probe and its tests."""

from __future__ import annotations

from typing import Any


def select_recommended_batch(
    rows: list[dict[str, Any]], total_vram_gib: float, headroom_fraction: float,
) -> int | None:
    """Choose the fastest batch whose allocated *and reserved* VRAM are safe.

    Older probe reports did not record ``reserved_vram_gib``.  Falling back to
    the allocated peak keeps those reports readable, while new reports must
    respect the larger of the two measurements so allocator reservation and
    fragmentation cannot silently consume the declared headroom.
    """
    limit = total_vram_gib * (1.0 - headroom_fraction)
    eligible = [
        row for row in rows
        if row.get("status") == "ok"
        and max(
            float(row["peak_vram_gib"]),
            float(row.get("reserved_vram_gib", row["peak_vram_gib"])),
        ) <= limit
    ]
    if not eligible:
        return None
    return int(max(eligible, key=lambda row: row["files_per_second"])["batch_size"])
