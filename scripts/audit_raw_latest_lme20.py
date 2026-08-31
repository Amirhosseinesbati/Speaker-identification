"""Fixed three-fold Raw/latest snapshot ensemble audit for CAM++ LME20.

This audit changes exactly one variable relative to the externally validated
Raw LME20 baseline: head, prototype and max-score evidence from the selected
Raw checkpoint is averaged 50/50 with evidence from the terminal ``latest``
checkpoint of the same OOF fold.  No epoch, weight, threshold or Fold-specific
parameter is selected.
"""

from __future__ import annotations

import argparse
import json
import os
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
    EQUIVALENCE_ATOL,
    build_or_load_ema_cache,
    decision_summary,
    final_decision,
    fixed_raw_ema_decision,
    probability_evidence,
)
from scripts.audit_short_audio_repeat import (  # noqa: E402
    LOCKED_BASELINE_MACRO_F1,
    LOCKED_LME_BETA,
    acceptance_gate,
    digest_names,
)


SNAPSHOT_VARIANT = "latest"
SNAPSHOT_WEIGHT = 0.5


def latest_checkpoint_path(checkpoint_root: Path, fold: int) -> Path:
    profile = f"p0-campp-known446-ood-control-oof-f{fold}"
    return checkpoint_root / profile / "campp_latest.pt"


def build_or_load_latest_cache(
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
    """Reuse the production extractor with a provenance-checked latest alias.

    ``build_or_load_ema_cache`` already owns the exact production batching,
    cache validation and atomic-write contract.  A temporary checkpoint-root
    view maps its expected ``campp_best_ema.pt`` name to the immutable latest
    checkpoint without altering either source artifact.
    """
    profile = f"p0-campp-known446-ood-control-oof-f{fold}"
    source = latest_checkpoint_path(checkpoint_root, fold)
    if not source.is_file():
        raise FileNotFoundError(source)

    alias_root = cache_dir / "checkpoint_aliases"
    alias_profile = alias_root / profile
    alias_profile.mkdir(parents=True, exist_ok=True)
    raw_source = checkpoint_root / profile / "campp_best_raw.pt"
    oof_source = (
        checkpoint_root / profile / "campp_best_bundle" / "oof_predictions.npz"
    )
    aliases = {
        alias_profile / "campp_best_raw.pt": raw_source,
        alias_profile / "campp_best_ema.pt": source,
        alias_profile / "campp_best_bundle" / "oof_predictions.npz": oof_source,
    }
    for alias, target in aliases.items():
        alias.parent.mkdir(parents=True, exist_ok=True)
        if alias.exists() or alias.is_symlink():
            if alias.resolve() != target.resolve():
                alias.unlink()
            else:
                continue
        try:
            alias.symlink_to(target.resolve())
        except OSError:
            # Windows developer environments may forbid symlinks.  The files
            # are read-only inputs; copying preserves the same hash contract.
            import shutil

            shutil.copy2(target, alias)

    arrays, metadata = build_or_load_ema_cache(
        fold=fold,
        raw_artifact=raw_artifact,
        raw_metadata=raw_metadata,
        raw_oof=raw_oof,
        checkpoint_root=alias_root,
        cache_dir=cache_dir / "latest_inference",
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    expected_latest_sha = sha256_file(source)
    if metadata["ema_checkpoint_sha256"] != expected_latest_sha:
        raise RuntimeError("Latest cache checkpoint hash mismatch")
    metadata = {
        **metadata,
        "variant": SNAPSHOT_VARIANT,
        "latest_checkpoint": str(source),
        "latest_checkpoint_sha256": expected_latest_sha,
    }
    return arrays, metadata


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
        default=ROOT / "data" / "experiments" / "campp_raw_latest_lme20",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "generated" / "campp_raw_latest_lme20.json",
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
    latest_arrays: list[dict[str, np.ndarray]] = []
    latest_metadata: list[dict] = []
    for fold in range(NUM_FOLDS):
        arrays, metadata = build_or_load_latest_cache(
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
        latest_arrays.append(arrays)
        latest_metadata.append(metadata)

    fold_rows: list[dict] = []
    all_files: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_raw_predictions: list[np.ndarray] = []
    all_latest_predictions: list[np.ndarray] = []
    all_ensemble_predictions: list[np.ndarray] = []
    for fold in range(NUM_FOLDS):
        raw_oof = raw_oofs[fold]
        raw_artifact = raw_artifacts[fold]
        latest = latest_arrays[fold]
        latest_artifact = {
            "train_embeddings": latest["train_embeddings"],
            "competition_labels": latest["competition_labels"],
            "unknown_cluster_ids": latest["unknown_cluster_ids"],
        }
        raw_evidence = probability_evidence(
            raw_artifact, raw_oof["embeddings"], raw_oof["competition_probs"]
        )
        latest_evidence = probability_evidence(
            latest_artifact,
            latest["validation_embeddings"],
            latest["validation_probabilities"],
        )
        raw_probabilities, raw_predictions = final_decision(*raw_evidence)
        latest_probabilities, latest_predictions = final_decision(*latest_evidence)
        ensemble_probabilities, ensemble_predictions = fixed_raw_ema_decision(
            raw_evidence, latest_evidence
        )
        self_probabilities, self_predictions = fixed_raw_ema_decision(
            raw_evidence, raw_evidence
        )
        if not np.array_equal(self_predictions, raw_predictions):
            raise RuntimeError(f"Fold {fold} self-ensemble prediction mismatch")
        if float(np.max(np.abs(self_probabilities - raw_probabilities))) > EQUIVALENCE_ATOL:
            raise RuntimeError(f"Fold {fold} self-ensemble probability mismatch")

        labels = raw_oof["labels"].astype(np.int64)
        raw_metrics = metric_bundle(labels, raw_predictions)
        latest_metrics = metric_bundle(labels, latest_predictions)
        ensemble_metrics = metric_bundle(labels, ensemble_predictions)
        fold_rows.append({
            "fold": fold,
            "raw": raw_metrics,
            "latest_diagnostic": latest_metrics,
            "ensemble": ensemble_metrics,
            "delta": metric_delta(ensemble_metrics, raw_metrics),
            "latest_delta": metric_delta(latest_metrics, raw_metrics),
            **decision_summary(labels, raw_predictions, ensemble_predictions),
            "raw_latest_probability_max_abs_diff": float(np.max(np.abs(
                raw_probabilities - latest_probabilities
            ))),
            "raw_latest_prediction_disagreements": int(np.sum(
                raw_predictions != latest_predictions
            )),
        })
        all_files.append(raw_oof["files"].astype(str))
        all_labels.append(labels)
        all_raw_predictions.append(raw_predictions)
        all_latest_predictions.append(latest_predictions)
        all_ensemble_predictions.append(ensemble_predictions)

    files = np.concatenate(all_files)
    if len(set(files.tolist())) != len(files):
        raise RuntimeError("OOF validation files overlap across folds")
    labels = np.concatenate(all_labels)
    raw_predictions = np.concatenate(all_raw_predictions)
    latest_predictions = np.concatenate(all_latest_predictions)
    ensemble_predictions = np.concatenate(all_ensemble_predictions)
    raw_metrics = metric_bundle(labels, raw_predictions)
    latest_metrics = metric_bundle(labels, latest_predictions)
    ensemble_metrics = metric_bundle(labels, ensemble_predictions)
    if abs(raw_metrics["macro_f1"] - LOCKED_BASELINE_MACRO_F1) > 1e-10:
        raise RuntimeError("Locked Raw LME20 baseline mismatch")
    aggregate = {
        "raw": raw_metrics,
        "latest_diagnostic": latest_metrics,
        "ensemble": ensemble_metrics,
        "delta": metric_delta(ensemble_metrics, raw_metrics),
        "latest_delta": metric_delta(latest_metrics, raw_metrics),
        **decision_summary(labels, raw_predictions, ensemble_predictions),
        "raw_latest_prediction_disagreements": int(np.sum(
            raw_predictions != latest_predictions
        )),
    }
    gate = acceptance_gate(fold_rows, aggregate)
    report = {
        "contract": {
            "single_changed_variable": (
                "Raw-only head/prototype/max-score evidence -> fixed 50/50 "
                "Raw/latest evidence average"
            ),
            "snapshot_variant": SNAPSHOT_VARIANT,
            "snapshot_weight": SNAPSHOT_WEIGHT,
            "weights_selected": False,
            "epochs_selected": False,
            "folds": "fixed kfold/folds3/seed42 OOF",
            "lme_beta": LOCKED_LME_BETA,
            "leaderboard_tuning": False,
        },
        "provenance": {
            "raw_cache_metadata": raw_metadata,
            "latest_cache_metadata": latest_metadata,
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
