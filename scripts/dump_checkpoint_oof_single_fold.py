"""Create submission-consistent OOF probabilities for one immutable checkpoint.

The script rebuilds the checkpoint's exact leak-free fold through the normal
data pipeline, loads the project checkpoint with the restricted unpickler, and
uses the same per-window probability averaging as training and submission.
It performs no threshold, checkpoint, epoch, or fusion search.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_control_oof_centroid_crossfit import metric_bundle  # noqa: E402
from src.checkpoint_io import load_project_checkpoint_safe  # noqa: E402
from src.data_pipeline import get_dataloaders  # noqa: E402
from src.model_factory import create_model_from_config  # noqa: E402
from src.train import forward_multi_window_evaluation  # noqa: E402


NUM_CLASSES = 447


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collapse_labels(labels: np.ndarray) -> np.ndarray:
    values = np.asarray(labels, dtype=np.int64)
    return np.where(values >= NUM_CLASSES, 0, values).astype(np.int64)


def validate_probability_matrix(probabilities: np.ndarray, rows: int) -> None:
    values = np.asarray(probabilities)
    if values.shape != (rows, NUM_CLASSES):
        raise RuntimeError(
            "Unexpected probability shape: "
            f"expected={(rows, NUM_CLASSES)}, got={values.shape}"
        )
    if not np.isfinite(values).all():
        raise RuntimeError("Checkpoint OOF probabilities contain NaN/Inf")
    if np.any(values < -1e-7):
        raise RuntimeError("Checkpoint OOF probabilities contain negative values")
    if not np.allclose(values.sum(axis=1), 1.0, rtol=0, atol=2e-5):
        raise RuntimeError("Checkpoint OOF probability rows do not sum to one")


def validate_checkpoint_split(config: dict, expected_fold: int) -> dict:
    split = ((config.get("data", {}) or {}).get("split", {}) or {})
    observed = {
        "scheme": str(split.get("scheme", "")),
        "fold": int(split.get("fold", -1)),
        "folds": int(split.get("folds", -1)),
        "seed": int(split.get("seed", -1)),
    }
    expected = {
        "scheme": "kfold",
        "fold": int(expected_fold),
        "folds": 3,
        "seed": 42,
    }
    if observed != expected:
        raise RuntimeError(
            f"Checkpoint split mismatch: expected={expected}, observed={observed}"
        )
    return observed


@torch.inference_mode()
def collect_probabilities(model, loader, device: torch.device):
    probabilities: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    model.eval()
    for windows, targets in loader:
        windows = windows.to(device, non_blocking=True)
        _, _, competition_probs = forward_multi_window_evaluation(model, windows)
        probabilities.append(competition_probs.cpu())
        labels.append(targets.cpu())
    return torch.cat(probabilities).numpy(), torch.cat(labels).numpy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    actual_sha = sha256_file(checkpoint)
    if actual_sha != args.expected_checkpoint_sha256:
        raise RuntimeError(
            "Checkpoint SHA mismatch: "
            f"expected={args.expected_checkpoint_sha256}, actual={actual_sha}"
        )
    payload = load_project_checkpoint_safe(checkpoint)
    config = payload.get("config")
    class_map = payload.get("class_map")
    if not isinstance(config, dict) or not isinstance(class_map, dict):
        raise RuntimeError("Checkpoint is missing dictionary config/class_map metadata")
    split = validate_checkpoint_split(config, args.fold)

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    _, validation_loader, rebuilt_class_map = get_dataloaders(config=config)
    if rebuilt_class_map != class_map:
        raise RuntimeError("Rebuilt class_map does not match the checkpoint class_map")
    validation_frame = validation_loader.dataset.df
    files = validation_frame["audio_file"].astype(str).to_numpy()
    if len(set(files.tolist())) != len(files):
        raise RuntimeError("Validation fold contains duplicate file ids")

    model = create_model_from_config(
        config, num_known_speakers=len(class_map) - 1
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    probabilities, labels = collect_probabilities(model, validation_loader, device)
    validate_probability_matrix(probabilities, len(files))
    collapsed_labels = collapse_labels(labels)
    metrics = metric_bundle(collapsed_labels, probabilities.argmax(axis=1))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        files=files,
        labels=labels.astype(np.int64),
        competition_probs=probabilities.astype(np.float32),
        split_scheme=np.asarray([split["scheme"]]),
        split_fold=np.asarray([split["fold"]], dtype=np.int64),
        split_folds=np.asarray([split["folds"]], dtype=np.int64),
        split_seed=np.asarray([split["seed"]], dtype=np.int64),
        checkpoint_sha256=np.asarray([actual_sha]),
    )
    receipt = {
        "contract": {
            "selection": "none; immutable checkpoint and exact embedded split",
            "threshold_or_fusion_search": False,
            "submission_authorized": False,
        },
        "provenance": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": actual_sha,
            "checkpoint_epoch": int(payload.get("epoch", -1)),
            "split": split,
            "class_map_size": len(class_map),
            "files": len(files),
            "file_order_sha256": hashlib.sha256(
                "\n".join(files).encode("utf-8")
            ).hexdigest(),
            "oof": str(args.output.resolve()),
            "oof_sha256": sha256_file(args.output),
        },
        "metrics": metrics,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
