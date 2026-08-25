"""Concatenate fold prediction bundles and compute one unbiased OOF score.

Usage:
    uv run --no-sync python scripts/aggregate_oof_results.py \
      checkpoints/run_f0_bundle/oof_predictions.npz \
      checkpoints/run_f1_bundle/oof_predictions.npz \
      checkpoints/run_f2_bundle/oof_predictions.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import evaluate_macro_f1


def aggregate(paths: list[str]) -> dict:
    records = [np.load(path, allow_pickle=False) for path in paths]
    widths = {r["speaker_logits"].shape[1] for r in records}
    clusters = {int(r["num_unknown_clusters"][0]) for r in records}
    if len(widths) != 1 or len(clusters) != 1:
        raise ValueError("All folds must use the same speaker-head width and cluster count")

    files = np.concatenate([r["files"] for r in records])
    if len(set(files.tolist())) != len(files):
        raise ValueError("OOF files overlap across folds; concatenation would leak/double count")
    labels = torch.from_numpy(np.concatenate([r["labels"] for r in records]))
    speaker = torch.from_numpy(np.concatenate([r["speaker_logits"] for r in records]))
    ood_arrays = [r["ood_logits"] for r in records]
    ood = (torch.from_numpy(np.concatenate(ood_arrays))
           if all(x.shape[1] == 1 for x in ood_arrays) else None)
    num_unknown_clusters = next(iter(clusters))
    competition_classes = speaker.shape[1] - num_unknown_clusters + 1
    metrics = evaluate_macro_f1(
        ood, speaker, labels,
        num_classes=competition_classes,
        num_unknown_clusters=num_unknown_clusters,
    )
    return {
        "folds": len(records),
        "samples": int(len(files)),
        "unique_files": int(len(set(files.tolist()))),
        "speaker_head_width": int(speaker.shape[1]),
        "num_unknown_clusters": num_unknown_clusters,
        "competition_classes": int(competition_classes),
        **metrics,
        "inputs": [str(Path(p)) for p in paths],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", nargs="+", help="fold oof_predictions.npz files")
    parser.add_argument(
        "--out", default="reports/generated/oof_aggregate.json",
        help="JSON report path",
    )
    args = parser.parse_args()
    report = aggregate(args.predictions)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
