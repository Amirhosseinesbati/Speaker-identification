"""
Q3 — Offline decision-layer tuning (no GPU, uses dumped val artifacts).

Thin CLI over ``src/decision_engine`` (single source of truth shared with the
pipeline's ``decision_tune`` step). Sweeps the centroid + OOD-gate decision
knobs (alpha / kappa / tau / lambda_unknown) against the competition Macro-F1
and writes ``data/processed/decision_config.json`` (shipped by
build_submission.py).

Usage:
    uv run --no-sync python scripts/tune_decision.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.cli_utils import setup_utf8_stdio  # noqa: E402
setup_utf8_stdio()

from src.decision_engine import load_decision_artifacts, tune_decision_bundle  # noqa: E402

DATA = ROOT / "data" / "processed"
OUT_JSON = DATA / "decision_config.json"


def main() -> int:
    artifacts = load_decision_artifacts()
    output = tune_decision_bundle(artifacts)
    OUT_JSON.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"\n  ✓ Saved to {OUT_JSON}")
    print("\n✅ Decision tuning complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
