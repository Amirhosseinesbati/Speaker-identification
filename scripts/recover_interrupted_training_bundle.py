"""Idempotently finalise the selected bundle after an interrupted run.

The command never trains, evaluates, or regenerates OOF predictions.  It only
materialises the human-readable sidecars and manifest around an already-bound
selected Raw checkpoint and existing OOF payload.  A valid existing manifest
is treated as success and left byte-for-byte untouched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_campp_channelrobust_lme20 import (  # noqa: E402
    sha256_file,
    validate_raw_bundle_binding,
)
from scripts.analyze_control_oof_centroid_crossfit import (  # noqa: E402
    rebuild_exact_splits,
)
from src.model_artifacts import create_training_bundle  # noqa: E402
from src.pipelines.steps import evaluate_model  # noqa: E402


def _same_model_state(left: dict, right: dict) -> bool:
    left_state = left.get("model_state_dict") or {}
    right_state = right.get("model_state_dict") or {}
    return left_state.keys() == right_state.keys() and all(
        torch.equal(left_state[key], right_state[key]) for key in left_state
    )


def _summary(history: list[dict]) -> dict:
    raw = np.asarray([float(row["val_macro_f1"]) for row in history])
    ema_values = [row.get("val_ema_macro_f1") for row in history]
    finite_ema = [float(value) for value in ema_values if value is not None]
    return {
        "best_val_macro_f1": float(np.max(raw)),
        "best_raw_val_macro_f1": float(np.max(raw)),
        "best_ema_val_macro_f1": max(finite_ema) if finite_ema else None,
        "selected_weight_variant": "raw",
        "total_epochs_run": len(history),
        "recovery": "interrupted_bundle_manifest_only",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument(
        "--checkpoint-root", type=Path, default=ROOT / "checkpoints"
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=ROOT / "data" / "processed" / "audio_wav_labels.csv",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "audio_wav",
    )
    parser.add_argument(
        "--regenerate-oof",
        action="store_true",
        help=(
            "Run only the canonical evaluation step when OOF predictions are "
            "missing. No training or parameter search is performed."
        ),
    )
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_root / args.profile
    selected_path = checkpoint_dir / "campp_best.pt"
    raw_path = checkpoint_dir / "campp_best_raw.pt"
    latest_path = checkpoint_dir / "campp_latest.pt"
    bundle_dir = checkpoint_dir / "campp_best_bundle"
    oof_path = bundle_dir / "oof_predictions.npz"
    required = (selected_path, raw_path, latest_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"required interrupted-run artifacts missing: {missing}")

    manifest_path = bundle_dir / "manifest.json"
    if manifest_path.is_file() and oof_path.is_file():
        binding = validate_raw_bundle_binding(
            checkpoint_dir, raw_path, oof_path
        )
        print(json.dumps({
            "status": "already_valid",
            "profile": args.profile,
            "binding": binding,
        }, sort_keys=True))
        return 0

    selected = torch.load(selected_path, map_location="cpu", weights_only=False)
    raw = torch.load(raw_path, map_location="cpu", weights_only=False)
    latest = torch.load(latest_path, map_location="cpu", weights_only=False)
    if selected.get("weight_variant") != "raw" or raw.get("weight_variant") != "raw":
        raise RuntimeError("selected and Raw checkpoints must both be Raw")
    if int(selected.get("epoch", -1)) != int(raw.get("epoch", -2)):
        raise RuntimeError("selected and Raw checkpoint epochs differ")
    if not np.isclose(
        float(selected.get("val_macro_f1", np.nan)),
        float(raw.get("val_macro_f1", np.nan)),
        atol=1e-12,
        rtol=0.0,
    ):
        raise RuntimeError("selected and Raw checkpoint metrics differ")
    if not _same_model_state(selected, raw):
        raise RuntimeError("selected and Raw model states differ")
    for key in ("config", "class_map"):
        if selected.get(key) != raw.get(key) or selected.get(key) != latest.get(key):
            raise RuntimeError(f"checkpoint {key} provenance differs")

    history = latest.get("training_history", latest.get("history", []))
    if not isinstance(history, list) or not history:
        raise RuntimeError("latest checkpoint has no training history")
    epochs = [int(row.get("epoch", -1)) for row in history]
    if epochs != list(range(1, len(history) + 1)):
        raise RuntimeError("latest history is not contiguous")
    if int(latest.get("epoch", -1)) != len(history):
        raise RuntimeError("latest checkpoint epoch/history mismatch")
    if int(selected.get("epoch", -1)) > len(history):
        raise RuntimeError("selected checkpoint lies after interrupted history")

    if not oof_path.is_file():
        if not args.regenerate_oof:
            raise RuntimeError(
                "OOF predictions are missing; pass --regenerate-oof to run "
                "the canonical evaluation-only recovery"
            )
        split = (selected["config"].get("data", {}) or {}).get("split", {}) or {}
        fold = int(split.get("fold", -1))
        splits, _ = rebuild_exact_splits(args.labels, args.audio_dir)
        if fold not in splits:
            raise RuntimeError(f"configured validation fold is unavailable: {fold}")
        _, validation_frame = splits[fold]
        evaluate_model.entrypoint(
            config=selected["config"],
            class_map=selected["class_map"],
            val_df=validation_frame,
            best_model_path=str(selected_path),
        )
        if not oof_path.is_file() or not manifest_path.is_file():
            raise RuntimeError("canonical evaluation did not materialise OOF bundle")
        binding = validate_raw_bundle_binding(
            checkpoint_dir, raw_path, oof_path
        )
        print(json.dumps({
            "status": "recovered_oof_and_bundle",
            "profile": args.profile,
            "history_points": len(history),
            "manifest_sha256": sha256_file(manifest_path),
            "binding": binding,
        }, sort_keys=True))
        return 0

    create_training_bundle(
        selected_path,
        selected["config"],
        selected["class_map"],
        history,
        _summary(history),
    )
    binding = validate_raw_bundle_binding(checkpoint_dir, raw_path, oof_path)
    print(json.dumps({
        "status": "recovered",
        "profile": args.profile,
        "history_points": len(history),
        "manifest_sha256": sha256_file(manifest_path),
        "binding": binding,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
