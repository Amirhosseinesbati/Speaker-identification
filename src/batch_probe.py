"""Pure selection logic shared by the GPU batch probe and its tests."""

from __future__ import annotations

from typing import Any


def select_recommended_batch(
    rows: list[dict[str, Any]], total_vram_gib: float, headroom_fraction: float,
) -> int | None:
    """Choose measured max-throughput batch within the VRAM safety limit."""
    limit = total_vram_gib * (1.0 - headroom_fraction)
    eligible = [
        row for row in rows
        if row.get("status") == "ok" and row["peak_vram_gib"] <= limit
    ]
    if not eligible:
        return None
    return int(max(eligible, key=lambda row: row["files_per_second"])["batch_size"])
