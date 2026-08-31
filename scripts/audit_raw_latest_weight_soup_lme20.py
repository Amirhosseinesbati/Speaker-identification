"""Fixed uniform Raw/latest weight-soup audit for CAM++ LME20.

The only changed variable relative to the locked Raw LME20 baseline is a
parameter-space average of the selected Raw checkpoint and the terminal latest
checkpoint from the same OOF fold.  The soup weight is fixed at 50/50 before
evaluation.  There is no epoch, threshold, Fold-specific or leaderboard
selection.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_control_oof_centroid_crossfit import (  # noqa: E402
    NUM_FOLDS,
    metric_bundle,
    metric_delta,
    sha256_file,
)
from scripts.analyze_unknown_cluster_hypotheses_crossfit import (  # noqa: E402
    load_fold_inputs,
)
from scripts.audit_raw_ema_lme20 import (  # noqa: E402
    build_or_load_ema_cache,
    decision_summary,
    final_decision,
    probability_evidence,
)
from scripts.audit_short_audio_repeat import (  # noqa: E402
    LOCKED_BASELINE_MACRO_F1,
    LOCKED_LME_BETA,
    acceptance_gate,
    digest_names,
)


SOUP_WEIGHT = 0.5
SOUP_VARIANT = "uniform_raw_latest_weight_soup"
REFERENCE_PAPER = "https://arxiv.org/abs/2203.05482"


def uniform_soup_state_dict(
    raw_state: dict[str, torch.Tensor],
    latest_state: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    """Average compatible floating tensors and preserve Raw integer buffers."""
    if raw_state.keys() != latest_state.keys():
        raise RuntimeError("Raw/latest state-dict keys differ")
    soup: dict[str, torch.Tensor] = {}
    floating_tensors = 0
    preserved_nonfloating_tensors = 0
    for key in raw_state:
        raw = raw_state[key].detach().cpu()
        latest = latest_state[key].detach().cpu()
        if raw.shape != latest.shape or raw.dtype != latest.dtype:
            raise RuntimeError(
                f"Raw/latest tensor mismatch for {key}: "
                f"{raw.shape}/{raw.dtype} != {latest.shape}/{latest.dtype}"
            )
        if torch.is_floating_point(raw) or torch.is_complex(raw):
            work_dtype = torch.float64 if raw.dtype == torch.float64 else torch.float32
            averaged = (
                SOUP_WEIGHT * raw.to(work_dtype)
                + SOUP_WEIGHT * latest.to(work_dtype)
            )
            soup[key] = averaged.to(raw.dtype)
            floating_tensors += 1
        else:
            soup[key] = raw.clone()
            preserved_nonfloating_tensors += 1
    return soup, {
        "averaged_floating_tensors": floating_tensors,
        "preserved_raw_nonfloating_tensors": preserved_nonfloating_tensors,
    }


def build_or_load_soup_checkpoint(
    *, fold: int, checkpoint_root: Path, cache_dir: Path
) -> tuple[Path, dict]:
    profile = f"p0-campp-known446-ood-control-oof-f{fold}"
    raw_path = checkpoint_root / profile / "campp_best_raw.pt"
    latest_path = checkpoint_root / profile / "campp_latest.pt"
    output_dir = cache_dir / "soup_checkpoints" / profile
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "campp_uniform_raw_latest.pt"
    metadata_path = output_path.with_suffix(".json")
    expected = {
        "schema_version": 1,
        "fold": fold,
        "variant": SOUP_VARIANT,
        "raw_weight": SOUP_WEIGHT,
        "latest_weight": SOUP_WEIGHT,
        "raw_checkpoint_sha256": sha256_file(raw_path),
        "latest_checkpoint_sha256": sha256_file(latest_path),
    }
    if output_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            all(metadata.get(key) == value for key, value in expected.items())
            and metadata.get("soup_checkpoint_sha256") == sha256_file(output_path)
        ):
            return output_path, {**metadata, "cache_status": "reused"}

    raw_checkpoint = torch.load(raw_path, map_location="cpu", weights_only=False)
    latest_checkpoint = torch.load(
        latest_path, map_location="cpu", weights_only=False
    )
    if raw_checkpoint["config"] != latest_checkpoint["config"]:
        raise RuntimeError(f"Fold {fold} Raw/latest configs differ")
    if raw_checkpoint["class_map"] != latest_checkpoint["class_map"]:
        raise RuntimeError(f"Fold {fold} Raw/latest class maps differ")
    soup_state, tensor_summary = uniform_soup_state_dict(
        raw_checkpoint["model_state_dict"], latest_checkpoint["model_state_dict"]
    )
    soup_checkpoint = {
        "config": raw_checkpoint["config"],
        "class_map": raw_checkpoint["class_map"],
        "model_state_dict": soup_state,
        "soup_contract": expected,
    }
    temporary_checkpoint = output_path.with_suffix(".pt.tmp")
    torch.save(soup_checkpoint, temporary_checkpoint)
    os.replace(temporary_checkpoint, output_path)
    metadata = {
        **expected,
        **tensor_summary,
        "soup_checkpoint_sha256": sha256_file(output_path),
        "reference_paper": REFERENCE_PAPER,
    }
    temporary_metadata = metadata_path.with_suffix(".json.tmp")
    temporary_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    os.replace(temporary_metadata, metadata_path)
    return output_path, {**metadata, "cache_status": "created"}


def link_or_copy(alias: Path, target: Path) -> None:
    alias.parent.mkdir(parents=True, exist_ok=True)
    if alias.exists() or alias.is_symlink():
        if alias.resolve() == target.resolve():
            return
        alias.unlink()
    try:
        alias.symlink_to(target.resolve())
    except OSError:
        shutil.copy2(target, alias)


def build_or_load_soup_cache(
    *,
    fold: int,
    raw_artifact: dict[str, np.ndarray],
    raw_metadata: dict,
    raw_oof: dict[str, np.ndarray],
    checkpoint_root: Path,
    cache_dir: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[dict[str, np.ndarray], dict]:
    profile = f"p0-campp-known446-ood-control-oof-f{fold}"
    soup_path, soup_metadata = build_or_load_soup_checkpoint(
        fold=fold, checkpoint_root=checkpoint_root, cache_dir=cache_dir
    )
    alias_root = cache_dir / "checkpoint_aliases"
    alias_profile = alias_root / profile
    link_or_copy(
        alias_profile / "campp_best_raw.pt",
        checkpoint_root / profile / "campp_best_raw.pt",
    )
    link_or_copy(alias_profile / "campp_best_ema.pt", soup_path)
    link_or_copy(
        alias_profile / "campp_best_bundle" / "oof_predictions.npz",
        checkpoint_root / profile / "campp_best_bundle" / "oof_predictions.npz",
    )
    arrays, inference_metadata = build_or_load_ema_cache(
        fold=fold,
        raw_artifact=raw_artifact,
        raw_metadata=raw_metadata,
        raw_oof=raw_oof,
        checkpoint_root=alias_root,
        cache_dir=cache_dir / "soup_inference",
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    if inference_metadata["ema_checkpoint_sha256"] != soup_metadata[
        "soup_checkpoint_sha256"
    ]:
        raise RuntimeError("Soup inference checkpoint hash mismatch")
    return arrays, {
        **inference_metadata,
        "variant": SOUP_VARIANT,
        "soup": soup_metadata,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-root", type=Path, default=ROOT / "checkpoints"
    )
    parser.add_argument(
        "--raw-cache-dir",
        type=Path,
        default=ROOT / "data" / "experiments" / "campp_control_centroid_crossfit",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "data" / "experiments" / "campp_weight_soup_lme20",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "generated" / "campp_weight_soup_lme20.json",
    )
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    raw_oofs, raw_artifacts, raw_metadata = load_fold_inputs(
        args.checkpoint_root, args.raw_cache_dir
    )
    soup_arrays: list[dict[str, np.ndarray]] = []
    soup_metadata: list[dict] = []
    for fold in range(NUM_FOLDS):
        arrays, metadata = build_or_load_soup_cache(
            fold=fold,
            raw_artifact=raw_artifacts[fold],
            raw_metadata=raw_metadata[fold],
            raw_oof=raw_oofs[fold],
            checkpoint_root=args.checkpoint_root,
            cache_dir=args.cache_dir,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        soup_arrays.append(arrays)
        soup_metadata.append(metadata)

    fold_rows: list[dict] = []
    all_files: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_raw_predictions: list[np.ndarray] = []
    all_soup_predictions: list[np.ndarray] = []
    for fold in range(NUM_FOLDS):
        raw_oof = raw_oofs[fold]
        raw_artifact = raw_artifacts[fold]
        soup = soup_arrays[fold]
        soup_artifact = {
            "train_embeddings": soup["train_embeddings"],
            "competition_labels": soup["competition_labels"],
            "unknown_cluster_ids": soup["unknown_cluster_ids"],
        }
        raw_evidence = probability_evidence(
            raw_artifact, raw_oof["embeddings"], raw_oof["competition_probs"]
        )
        soup_evidence = probability_evidence(
            soup_artifact,
            soup["validation_embeddings"],
            soup["validation_probabilities"],
        )
        _, raw_predictions = final_decision(*raw_evidence)
        _, soup_predictions = final_decision(*soup_evidence)
        labels = raw_oof["labels"].astype(np.int64)
        raw_metrics = metric_bundle(labels, raw_predictions)
        soup_metrics = metric_bundle(labels, soup_predictions)
        fold_rows.append({
            "fold": fold,
            "raw": raw_metrics,
            "soup": soup_metrics,
            "delta": metric_delta(soup_metrics, raw_metrics),
            **decision_summary(labels, raw_predictions, soup_predictions),
        })
        all_files.append(raw_oof["files"].astype(str))
        all_labels.append(labels)
        all_raw_predictions.append(raw_predictions)
        all_soup_predictions.append(soup_predictions)

    files = np.concatenate(all_files)
    if len(set(files.tolist())) != len(files):
        raise RuntimeError("OOF validation files overlap across folds")
    labels = np.concatenate(all_labels)
    raw_predictions = np.concatenate(all_raw_predictions)
    soup_predictions = np.concatenate(all_soup_predictions)
    raw_metrics = metric_bundle(labels, raw_predictions)
    soup_metrics = metric_bundle(labels, soup_predictions)
    if abs(raw_metrics["macro_f1"] - LOCKED_BASELINE_MACRO_F1) > 1e-10:
        raise RuntimeError("Locked Raw LME20 baseline mismatch")
    aggregate = {
        "raw": raw_metrics,
        "soup": soup_metrics,
        "delta": metric_delta(soup_metrics, raw_metrics),
        **decision_summary(labels, raw_predictions, soup_predictions),
    }
    gate = acceptance_gate(fold_rows, aggregate)
    report = {
        "contract": {
            "single_changed_variable": (
                "Raw checkpoint -> fixed 50/50 Raw/latest uniform weight soup"
            ),
            "raw_weight": SOUP_WEIGHT,
            "latest_weight": SOUP_WEIGHT,
            "weights_selected": False,
            "epochs_selected": False,
            "folds": "fixed kfold/folds3/seed42 OOF",
            "lme_beta": LOCKED_LME_BETA,
            "leaderboard_tuning": False,
            "reference_paper": REFERENCE_PAPER,
        },
        "provenance": {
            "raw_cache_metadata": raw_metadata,
            "soup_cache_metadata": soup_metadata,
            "unique_oof_files": int(len(files)),
            "oof_file_sha256": digest_names(files),
        },
        "folds": fold_rows,
        "aggregate": aggregate,
        "acceptance_gate": gate,
        "decision": "accept" if gate["passed"] else "reject",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({
        "output": str(args.output),
        "decision": report["decision"],
        "fold_deltas": [row["delta"] for row in fold_rows],
        "aggregate": aggregate,
        "gate": gate,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
