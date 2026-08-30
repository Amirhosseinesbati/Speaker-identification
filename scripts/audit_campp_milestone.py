"""Create a machine-readable diagnostic receipt for a CAM++ milestone.

Milestones are monitoring evidence only.  This command validates checkpoint
profile binding, an exact contiguous history, finite metrics and the locked
rolling-window diagnostics.  It never selects a checkpoint or authorises a
later Run.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_campp_paired_consistency_lme20 import (  # noqa: E402
    milestone_diagnostic,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.epoch <= 0:
        raise SystemExit("--epoch must be positive")

    receipt = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision_role": (
            "diagnostic_only_cannot_select_checkpoint_or_authorize_run"
        ),
        "diagnostic": milestone_diagnostic(
            args.checkpoint,
            args.profile,
            expected_epoch=args.epoch,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
