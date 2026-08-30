"""Deterministic checkpoint audit for the P6 inter-class mechanism gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train import exclusive_inter_class_angular_loss  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_energy(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Checkpoint is not a dictionary: {path}")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise RuntimeError(f"Checkpoint lacks model_state_dict: {path}")
    matches = [
        (key, value)
        for key, value in state.items()
        if str(key).endswith("head_speaker.weight")
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one head_speaker.weight in {path}, found "
            f"{len(matches)}"
        )
    key, weights = matches[0]
    if not isinstance(weights, torch.Tensor) or weights.ndim != 2:
        raise RuntimeError(
            f"{key} must be one 2-D single-center ArcFace tensor"
        )
    energy = float(
        exclusive_inter_class_angular_loss(weights).detach().cpu().item()
    )
    if not math.isfinite(energy):
        raise RuntimeError(f"Non-finite inter-class energy in {path}")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "epoch": checkpoint.get("epoch"),
        "weight_variant": checkpoint.get("weight_variant"),
        "state_key": str(key),
        "num_classes": int(weights.shape[0]),
        "embedding_dim": int(weights.shape[1]),
        "exclusive_inter_class_energy": energy,
    }


def audit_pair(
    control_checkpoint: Path,
    treatment_checkpoint: Path,
    *,
    maximum_energy_ratio: float = 0.95,
) -> dict:
    if not 0.0 < maximum_energy_ratio <= 1.0:
        raise ValueError("maximum_energy_ratio must be in (0, 1]")
    control = _checkpoint_energy(control_checkpoint)
    treatment = _checkpoint_energy(treatment_checkpoint)
    control_shape = (control["num_classes"], control["embedding_dim"])
    treatment_shape = (
        treatment["num_classes"],
        treatment["embedding_dim"],
    )
    if control_shape != treatment_shape:
        raise RuntimeError(
            f"Class-weight shape mismatch: {control_shape} vs {treatment_shape}"
        )
    control_energy = control["exclusive_inter_class_energy"]
    treatment_energy = treatment["exclusive_inter_class_energy"]
    if control_energy <= 0.0:
        raise RuntimeError(
            "Control inter-class energy must be positive for a ratio gate"
        )
    ratio = treatment_energy / control_energy
    return {
        "schema_version": 1,
        "formula": "mean_class_frobenius_square(relu(Wn@Wn.T)-I)",
        "control": control,
        "treatment": treatment,
        "mechanism_gate": {
            "maximum_energy_ratio": float(maximum_energy_ratio),
            "observed_energy_ratio": float(ratio),
            "absolute_energy_change": float(treatment_energy - control_energy),
            "passed": bool(ratio <= maximum_energy_ratio),
        },
    }


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-checkpoint", type=Path, required=True)
    parser.add_argument("--treatment-checkpoint", type=Path, required=True)
    parser.add_argument("--maximum-energy-ratio", type=float, default=0.95)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_pair(
        args.control_checkpoint,
        args.treatment_checkpoint,
        maximum_energy_ratio=args.maximum_energy_ratio,
    )
    if args.output is not None:
        _atomic_write(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
