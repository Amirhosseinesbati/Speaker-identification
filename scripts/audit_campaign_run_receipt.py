"""Verify one completed campaign run and its scientific receipt artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _one_completed_run(state: dict[str, Any], profile: str) -> dict[str, Any]:
    matches = [
        run for run in state.get("completed_runs", [])
        if run.get("profile") == profile
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one completed run for {profile!r}; "
            f"found {len(matches)}"
        )
    run = matches[0]
    if run.get("status") != "complete" or int(run.get("exit_code", -1)) != 0:
        raise RuntimeError(
            f"run is not a successful completion: status={run.get('status')!r}, "
            f"exit_code={run.get('exit_code')!r}"
        )
    return run


def _receipt_paths(
    project_root: Path,
    receipts: list[dict[str, Any]],
) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    root = project_root.resolve()
    resolved: dict[str, Path] = {}
    verified: list[dict[str, Any]] = []
    for receipt in receipts:
        relative = str(receipt.get("path", "")).replace("\\", "/")
        if not relative or relative in resolved:
            raise RuntimeError(f"missing or duplicate receipt path: {relative!r}")
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise RuntimeError(f"receipt escapes project root: {relative}")
        if not path.is_file():
            raise RuntimeError(f"receipt artifact is missing: {relative}")
        actual_size = path.stat().st_size
        actual_sha = sha256_file(path)
        if actual_size != int(receipt.get("size_bytes", -1)):
            raise RuntimeError(f"receipt size mismatch: {relative}")
        if actual_sha != str(receipt.get("sha256", "")):
            raise RuntimeError(f"receipt SHA256 mismatch: {relative}")
        resolved[relative] = path
        verified.append({
            "path": relative,
            "size_bytes": actual_size,
            "sha256": actual_sha,
        })
    return resolved, verified


def _audit_checkpoint(
    checkpoint: dict[str, Any],
    profile: str,
) -> dict[str, Any]:
    config = checkpoint.get("config", {}) or {}
    training = config.get("training", {}) or {}
    history = checkpoint.get("training_history", checkpoint.get("history"))
    if not isinstance(history, list) or not history:
        raise RuntimeError("canonical checkpoint has no training history")
    epochs = [int(row.get("epoch", -1)) for row in history]
    expected_epochs = int(training.get("epochs", len(epochs)))
    if epochs != list(range(1, len(epochs) + 1)):
        raise RuntimeError("training history is not contiguous")
    if len(epochs) > expected_epochs:
        raise RuntimeError("training history exceeds the configured horizon")

    early_stopping_patience = int(training.get("early_stopping_patience", 0))
    early_stopping_start_epoch = int(
        training.get("early_stopping_start_epoch", 1)
    )
    early_stopped = len(epochs) < expected_epochs
    if early_stopped:
        if early_stopping_patience <= 0:
            raise RuntimeError(
                "training history ended before the configured horizon without "
                "early stopping enabled"
            )
        earliest_legal_stop = (
            max(1, early_stopping_start_epoch) + early_stopping_patience - 1
        )
        if len(epochs) < earliest_legal_stop:
            raise RuntimeError(
                "training history ended before the earliest legal early stop"
            )

    class_map = checkpoint.get("class_map")
    if not isinstance(class_map, dict):
        raise RuntimeError("canonical checkpoint class_map is missing")
    model = config.get("model", {}) or {}
    expected_classes = (
        int(model.get("competition_num_known", 0))
        + int(model.get("num_unknown_clusters", 0))
        + 1
    )
    if expected_classes <= 1 or len(class_map) != expected_classes:
        raise RuntimeError(
            f"class_map size mismatch: {len(class_map)} != {expected_classes}"
        )

    split = (config.get("data", {}) or {}).get("split", {}) or {}
    required_split = {"scheme", "folds", "fold", "seed"}
    if not required_split.issubset(split):
        raise RuntimeError(f"checkpoint split provenance is incomplete: {split}")

    checkpoint_dir = str(
        (config.get("logging", {}) or {}).get("checkpoint_dir", "")
    ).replace("\\", "/")
    if not checkpoint_dir.endswith(f"checkpoints/{profile}"):
        raise RuntimeError("canonical checkpoint profile binding mismatch")

    requested_variant = str(training.get("selection_variant", "")).lower()
    actual_variant = str(checkpoint.get("weight_variant", "")).lower()
    if requested_variant and actual_variant != requested_variant:
        raise RuntimeError(
            f"selected weight variant mismatch: {actual_variant!r} "
            f"!= {requested_variant!r}"
        )
    return {
        "configured_epochs": expected_epochs,
        "history_points": len(history),
        "early_stopped": early_stopped,
        "early_stopping_start_epoch": early_stopping_start_epoch,
        "early_stopping_patience": early_stopping_patience,
        "selected_epoch": int(checkpoint.get("epoch", -1)),
        "weight_variant": actual_variant or None,
        "class_map_size": len(class_map),
        "competition_num_known": int(model["competition_num_known"]),
        "competition_class_count": int(model["competition_num_known"]) + 1,
        "split": split,
    }


def _audit_oof(path: Path, checkpoint_summary: dict[str, Any]) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "files", "labels", "speaker_logits", "ood_logits",
            "competition_probs", "embeddings", "split_scheme", "split_folds",
            "split_fold", "split_seed",
        }
        missing = required.difference(payload.files)
        if missing:
            raise RuntimeError(f"OOF bundle is missing arrays: {sorted(missing)}")
        files = payload["files"].astype(str)
        labels = payload["labels"]
        probs = payload["competition_probs"]
        embeddings = payload["embeddings"]
        speaker_logits = payload["speaker_logits"]
        ood_logits = payload["ood_logits"]
        n_rows = len(files)
        if n_rows == 0 or len(set(files.tolist())) != n_rows:
            raise RuntimeError("OOF files are empty or non-unique")
        arrays = (labels, probs, embeddings, speaker_logits, ood_logits)
        if any(array.shape[0] != n_rows for array in arrays):
            raise RuntimeError("OOF arrays have inconsistent row counts")
        for name, array in (
            ("competition_probs", probs),
            ("embeddings", embeddings),
            ("speaker_logits", speaker_logits),
            ("ood_logits", ood_logits),
        ):
            if not np.isfinite(array).all():
                raise RuntimeError(f"OOF {name} contains non-finite values")
        if probs.ndim != 2 or not np.allclose(
            probs.sum(axis=1), 1.0, rtol=0.0, atol=1e-5
        ):
            raise RuntimeError("OOF competition probabilities are not normalized")
        if probs.shape[1] != checkpoint_summary["competition_class_count"]:
            raise RuntimeError("OOF competition probability width is inconsistent")
        if (
            speaker_logits.ndim != 2
            or speaker_logits.shape[1]
            != checkpoint_summary["competition_num_known"]
        ):
            raise RuntimeError("OOF speaker-logit width is inconsistent")
        if ood_logits.ndim != 2 or ood_logits.shape[1] != 1:
            raise RuntimeError("OOF OOD-logit width is inconsistent")

        split = checkpoint_summary["split"]
        oof_split = {
            "scheme": str(payload["split_scheme"][0]),
            "folds": int(payload["split_folds"][0]),
            "fold": int(payload["split_fold"][0]),
            "seed": int(payload["split_seed"][0]),
        }
        expected_split = {
            "scheme": str(split["scheme"]),
            "folds": int(split["folds"]),
            "fold": int(split["fold"]),
            "seed": int(split["seed"]),
        }
        if oof_split != expected_split:
            raise RuntimeError(
                f"OOF split provenance mismatch: {oof_split} != {expected_split}"
            )
        return {
            "path": str(path),
            "rows": n_rows,
            "unique_files": n_rows,
            "competition_classes": int(probs.shape[1]),
            "embedding_dim": int(embeddings.shape[1]),
            "split": oof_split,
        }


def audit_campaign_run(
    project_root: Path,
    state_path: Path,
    profile: str,
) -> dict[str, Any]:
    root = project_root.resolve()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    run = _one_completed_run(state, profile)
    receipt_paths, verified = _receipt_paths(root, run.get("artifacts", []))

    config_receipt = f"configs/experiments/{profile}.yaml"
    if config_receipt not in receipt_paths:
        raise RuntimeError("experiment config is absent from campaign receipts")
    if sha256_file(receipt_paths[config_receipt]) != run.get("config_sha256"):
        raise RuntimeError("run config SHA256 does not match its receipt")

    checkpoint_prefix = f"checkpoints/{profile}/"
    pt_paths = [
        path for relative, path in receipt_paths.items()
        if relative.startswith(checkpoint_prefix) and path.suffix.lower() == ".pt"
    ]
    if not pt_paths:
        raise RuntimeError("campaign receipt contains no checkpoints")
    for path in pt_paths:
        torch.load(path, map_location="cpu", weights_only=False)

    canonical = [
        path for path in pt_paths
        if path.name.endswith("_best.pt")
        and not path.name.endswith(("_best_raw.pt", "_best_ema.pt"))
    ]
    if len(canonical) != 1:
        raise RuntimeError(f"expected one canonical best checkpoint; found {canonical}")
    checkpoint = torch.load(canonical[0], map_location="cpu", weights_only=False)
    checkpoint_summary = _audit_checkpoint(checkpoint, profile)

    oof_paths = [
        path for relative, path in receipt_paths.items()
        if relative.startswith(checkpoint_prefix)
        and relative.endswith("_best_bundle/oof_predictions.npz")
    ]
    if len(oof_paths) != 1:
        raise RuntimeError(f"expected one canonical OOF bundle; found {oof_paths}")
    oof_summary = _audit_oof(oof_paths[0], checkpoint_summary)

    return {
        "profile": profile,
        "git_commit": run.get("git_commit"),
        "config_sha256": run.get("config_sha256"),
        "started_at_utc": run.get("started_at_utc"),
        "finished_at_utc": run.get("finished_at_utc"),
        "exit_code": run.get("exit_code"),
        "receipt_artifacts": verified,
        "readable_checkpoints": len(pt_paths),
        "canonical_checkpoint": {
            "path": str(canonical[0]),
            "sha256": sha256_file(canonical[0]),
            **checkpoint_summary,
        },
        "oof": {"sha256": sha256_file(oof_paths[0]), **oof_summary},
        "passed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--state",
        type=Path,
        default=Path("data/experiments/campaign_state.json"),
    )
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    state_path = (
        args.state if args.state.is_absolute() else args.project_root / args.state
    )
    report = audit_campaign_run(args.project_root, state_path, args.profile)
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, args.output)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
