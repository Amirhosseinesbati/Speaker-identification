"""Fixed three-fold audit of view-level LME20 speaker scoring.

The locked backend first averages every file's temporal-window embeddings and
then scores enrollment files.  This candidate retains each real window view
and applies the same beta-20 normalised log-mean-exp over all query/enrollment
view pairs belonging to a speaker group.  Repeated padding windows are excluded
through deterministic per-file view counts.  Head probabilities, weights,
folds, group membership and every decision parameter remain fixed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_control_oof_centroid_crossfit import (  # noqa: E402
    NUM_FOLDS,
    metric_bundle,
    metric_delta,
    sha256_file,
)
from scripts.analyze_lme20_asnorm_crossfit import (  # noqa: E402
    LOCKED_ALPHA,
    LOCKED_RAW_KAPPA,
    LOCKED_TAU,
    LOCKED_UNKNOWN_WEIGHT,
    decision_predictions,
)
from scripts.analyze_prototype_aggregation_crossfit import (  # noqa: E402
    group_indices,
)
from scripts.analyze_unknown_cluster_hypotheses_crossfit import (  # noqa: E402
    load_fold_inputs,
)
from scripts.audit_short_audio_repeat import (  # noqa: E402
    LOCKED_BASELINE_MACRO_F1,
    LOCKED_LME_BETA,
    LONG_EQUIVALENCE_ATOL,
    acceptance_gate,
    atomic_savez,
    digest_names,
    locked_lme_predictions,
    make_eval_dataset,
    resolve_repo_path,
)
from src.model_factory import create_model_from_config  # noqa: E402


VIEW_CACHE_SCHEMA = 1


def post_resample_length(path: Path, target_sample_rate: int) -> int:
    info = sf.info(str(path))
    return int(round(
        info.frames * float(target_sample_rate) / float(info.samplerate)
    ))


def unique_eval_window_count(
    sample_count: int,
    target_length: int,
    hop_ratio: float,
    max_windows: int,
) -> int:
    """Mirror ``make_eval_windows`` count before last-window repetition."""
    sample_count = int(sample_count)
    target_length = int(target_length)
    max_windows = max(1, int(max_windows))
    if sample_count <= target_length:
        return 1
    hop = max(1, int(target_length * float(hop_ratio)))
    starts = list(range(0, sample_count - target_length + 1, hop))
    if starts[-1] != sample_count - target_length:
        starts.append(sample_count - target_length)
    return min(len(starts), max_windows)


def view_counts_for_files(
    files: np.ndarray,
    audio_dir: Path,
    target_sample_rate: int,
    target_length: int,
    hop_ratio: float,
    max_windows: int,
) -> np.ndarray:
    counts = [
        unique_eval_window_count(
            post_resample_length(audio_dir / str(name), target_sample_rate),
            target_length,
            hop_ratio,
            max_windows,
        )
        for name in files.astype(str)
    ]
    return np.asarray(counts, dtype=np.int64)


def extract_view_embeddings(
    *,
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    description: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-window unit embeddings and mean-raw aggregate embeddings."""
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=device.type == "cuda",
    )
    view_chunks: list[np.ndarray] = []
    aggregate_chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for windows, _ in tqdm(loader, desc=description):
            windows = windows.to(device, non_blocking=True)
            views = windows.shape[1]
            # Match SpeakerModel.embed exactly: CAM++ is evaluated one window
            # position at a time across the file batch.  Flattening B*W changes
            # CUDA kernel/batch numerics enough to violate the locked aggregate
            # reproduction invariant, even in eval mode.
            raw = torch.stack(
                [model._embed_single(windows[:, view]) for view in range(views)],
                dim=1,
            )
            view_chunks.append(F.normalize(raw, p=2, dim=2).cpu().numpy())
            aggregate_chunks.append(
                F.normalize(raw.mean(dim=1), p=2, dim=1).cpu().numpy()
            )
    return (
        np.concatenate(view_chunks, axis=0).astype(np.float32),
        np.concatenate(aggregate_chunks, axis=0).astype(np.float32),
    )


def validate_view_cache(
    arrays: dict[str, np.ndarray],
    pad_artifact: dict[str, np.ndarray],
    pad_oof: dict[str, np.ndarray],
) -> None:
    train_files = pad_artifact["train_files"].astype(str)
    validation_files = pad_oof["files"].astype(str)
    embedding_dim = int(pad_artifact["train_embeddings"].shape[1])
    max_windows = int(arrays["train_view_embeddings"].shape[1])
    expected = {
        "train_files": (len(train_files),),
        "train_view_embeddings": (len(train_files), max_windows, embedding_dim),
        "train_view_counts": (len(train_files),),
        "validation_files": (len(validation_files),),
        "validation_view_embeddings": (
            len(validation_files), max_windows, embedding_dim
        ),
        "validation_view_counts": (len(validation_files),),
    }
    for key, shape in expected.items():
        if key not in arrays or arrays[key].shape != shape:
            actual = None if key not in arrays else arrays[key].shape
            raise RuntimeError(f"{key} shape {actual} != {shape}")
    if not np.array_equal(arrays["train_files"].astype(str), train_files):
        raise RuntimeError("Train view cache order mismatch")
    if not np.array_equal(
        arrays["validation_files"].astype(str), validation_files
    ):
        raise RuntimeError("Validation view cache order mismatch")
    for key in ("train_view_counts", "validation_view_counts"):
        if np.any(arrays[key] < 1) or np.any(arrays[key] > max_windows):
            raise RuntimeError(f"Invalid {key}")
    for key in ("train_view_embeddings", "validation_view_embeddings"):
        if not np.all(np.isfinite(arrays[key])):
            raise RuntimeError(f"Non-finite {key}")


def build_or_load_view_cache(
    *,
    fold: int,
    pad_artifact: dict[str, np.ndarray],
    pad_metadata: dict,
    pad_oof: dict[str, np.ndarray],
    checkpoint_root: Path,
    cache_dir: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[dict[str, np.ndarray], dict]:
    profile = f"p0-campp-known446-ood-control-oof-f{fold}"
    checkpoint_path = checkpoint_root / profile / "campp_best_raw.pt"
    oof_path = checkpoint_root / profile / "campp_best_bundle" / "oof_predictions.npz"
    cache_path = cache_dir / f"fold{fold}_window_views.npz"
    metadata_path = cache_path.with_suffix(".json")
    train_files = pad_artifact["train_files"].astype(str)
    validation_files = pad_oof["files"].astype(str)
    expected = {
        "schema_version": VIEW_CACHE_SCHEMA,
        "fold": int(fold),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "pad_artifact_sha256": str(pad_metadata["artifact_sha256"]),
        "oof_sha256": sha256_file(oof_path),
        "train_file_sha256": digest_names(train_files),
        "validation_file_sha256": digest_names(validation_files),
    }
    if cache_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        reusable = (
            all(metadata.get(key) == value for key, value in expected.items())
            and metadata.get("artifact_sha256") == sha256_file(cache_path)
        )
        if reusable:
            with np.load(cache_path) as data:
                arrays = {key: data[key].copy() for key in data.files}
            validate_view_cache(arrays, pad_artifact, pad_oof)
            return arrays, {**metadata, "cache_status": "reused"}

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    config = deepcopy(checkpoint["config"])
    class_map = checkpoint["class_map"]
    model = create_model_from_config(
        config, num_known_speakers=len(class_map) - 1
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    train_labels = pad_artifact["competition_labels"].astype(np.int64)
    validation_labels = pad_oof["labels"].astype(np.int64)
    train_dataset = make_eval_dataset(train_files, train_labels, config, "pad")
    validation_dataset = make_eval_dataset(
        validation_files, validation_labels, config, "pad"
    )
    train_views, train_aggregate = extract_view_embeddings(
        model=model,
        dataset=train_dataset,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        description=f"Fold {fold} enrollment views",
    )
    validation_views, validation_aggregate = extract_view_embeddings(
        model=model,
        dataset=validation_dataset,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        description=f"Fold {fold} validation views",
    )
    train_diff = float(np.max(np.abs(
        train_aggregate - pad_artifact["train_embeddings"]
    )))
    validation_diff = float(np.max(np.abs(
        validation_aggregate - pad_oof["embeddings"]
    )))
    if max(train_diff, validation_diff) > LONG_EQUIVALENCE_ATOL:
        raise RuntimeError(
            f"Fold {fold} aggregate embedding reproduction failed: "
            f"train={train_diff}, validation={validation_diff}"
        )

    audio = config["audio"]
    audio_dir = resolve_repo_path(config["data"]["audio_dir"])
    sample_rate = int(audio["sample_rate"])
    target_length = int(sample_rate * float(audio["duration_seconds"]))
    hop_ratio = float(audio.get("eval_hop_ratio", 0.5))
    max_windows = int(audio.get("max_eval_windows", 8))
    train_counts = view_counts_for_files(
        train_files,
        audio_dir,
        sample_rate,
        target_length,
        hop_ratio,
        max_windows,
    )
    validation_counts = view_counts_for_files(
        validation_files,
        audio_dir,
        sample_rate,
        target_length,
        hop_ratio,
        max_windows,
    )
    arrays = {
        "train_files": train_files.astype(str),
        "train_view_embeddings": train_views,
        "train_view_counts": train_counts,
        "validation_files": validation_files.astype(str),
        "validation_view_embeddings": validation_views,
        "validation_view_counts": validation_counts,
    }
    validate_view_cache(arrays, pad_artifact, pad_oof)
    atomic_savez(cache_path, arrays)
    metadata = {
        **expected,
        "artifact_sha256": sha256_file(cache_path),
        "audio_dir": str(audio_dir),
        "sample_rate": sample_rate,
        "duration_seconds": float(audio["duration_seconds"]),
        "eval_hop_ratio": hop_ratio,
        "max_eval_windows": max_windows,
        "train_files": int(len(train_files)),
        "validation_files": int(len(validation_files)),
        "train_real_views": int(train_counts.sum()),
        "validation_real_views": int(validation_counts.sum()),
        "train_count_distribution": {
            str(count): int(np.sum(train_counts == count))
            for count in range(1, max_windows + 1)
        },
        "validation_count_distribution": {
            str(count): int(np.sum(validation_counts == count))
            for count in range(1, max_windows + 1)
        },
        "aggregate_reproduction": {
            "tolerance": LONG_EQUIVALENCE_ATOL,
            "train_max_abs_diff": train_diff,
            "validation_max_abs_diff": validation_diff,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return arrays, {**metadata, "cache_status": "built"}


def expand_enrollment_views(
    view_embeddings: np.ndarray,
    view_counts: np.ndarray,
    groups: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    chunks = []
    ids = []
    group_view_counts = np.zeros(len(groups), dtype=np.int64)
    for group_id, file_indices in enumerate(groups):
        group_chunks = [
            view_embeddings[index, : int(view_counts[index])]
            for index in file_indices
        ]
        group_views = np.concatenate(group_chunks, axis=0)
        chunks.append(group_views)
        ids.append(np.full(len(group_views), group_id, dtype=np.int64))
        group_view_counts[group_id] = len(group_views)
    if np.any(group_view_counts <= 0):
        raise RuntimeError("A speaker group has no enrollment views")
    return (
        np.concatenate(chunks, axis=0).astype(np.float32),
        np.concatenate(ids).astype(np.int64),
        group_view_counts,
    )


def multiview_logmeanexp_scores(
    *,
    query_views: np.ndarray,
    query_counts: np.ndarray,
    enrollment_views: np.ndarray,
    enrollment_group_ids: np.ndarray,
    group_view_counts: np.ndarray,
    beta: float,
    device: torch.device,
    file_batch_size: int,
) -> np.ndarray:
    """Normalised LME over every valid query/enrollment view pair."""
    query = torch.from_numpy(np.asarray(query_views, np.float32)).to(device)
    enroll = torch.from_numpy(np.asarray(enrollment_views, np.float32)).to(device)
    group_ids = torch.from_numpy(
        np.asarray(enrollment_group_ids, np.int64)
    ).to(device)
    group_counts = torch.from_numpy(
        np.asarray(group_view_counts, np.float32)
    ).to(device)
    num_groups = int(len(group_view_counts))
    max_query_views = int(query.shape[1])
    result = []
    with torch.inference_mode():
        for start in tqdm(
            range(0, len(query), int(file_batch_size)),
            desc="View-level LME scores",
        ):
            stop = min(len(query), start + int(file_batch_size))
            batch = query[start:stop]
            batch_files = int(len(batch))
            flat = batch.reshape(-1, batch.shape[-1])
            scaled = float(beta) * (flat @ enroll.T)
            index = group_ids.unsqueeze(0).expand(len(flat), -1)
            maxima = torch.full(
                (len(flat), num_groups),
                -torch.inf,
                dtype=scaled.dtype,
                device=device,
            )
            maxima.scatter_reduce_(
                1, index, scaled, reduce="amax", include_self=True
            )
            stable = torch.exp(scaled - maxima.gather(1, index))
            sums = torch.zeros_like(maxima)
            sums.scatter_add_(1, index, stable)
            enrollment_lme = (
                maxima + torch.log(sums) - torch.log(group_counts)[None, :]
            ) / float(beta)
            enrollment_lme = enrollment_lme.reshape(
                batch_files, max_query_views, num_groups
            )
            counts = torch.from_numpy(
                query_counts[start:stop].astype(np.int64)
            ).to(device)
            valid = (
                torch.arange(max_query_views, device=device)[None, :]
                < counts[:, None]
            )
            query_scaled = float(beta) * enrollment_lme
            query_scaled = query_scaled.masked_fill(~valid[:, :, None], -torch.inf)
            file_scores = (
                torch.logsumexp(query_scaled, dim=1)
                - torch.log(counts.float())[:, None]
            ) / float(beta)
            result.append(file_scores.cpu().numpy())
    scores = np.concatenate(result, axis=0).astype(np.float32)
    if not np.all(np.isfinite(scores)):
        raise RuntimeError("View-level LME produced non-finite scores")
    return scores


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-root", type=Path, default=ROOT / "checkpoints"
    )
    parser.add_argument(
        "--pad-cache-dir",
        type=Path,
        default=ROOT / "data" / "experiments" / "campp_control_centroid_crossfit",
    )
    parser.add_argument(
        "--view-cache-dir",
        type=Path,
        default=ROOT / "data" / "experiments" / "campp_multiview_lme20",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "generated" / "campp_multiview_lme20.json",
    )
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--score-batch-size", type=int, default=16)
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

    pad_oofs, pad_artifacts, pad_metadata = load_fold_inputs(
        args.checkpoint_root, args.pad_cache_dir
    )
    caches = []
    cache_metadata = []
    for fold in range(NUM_FOLDS):
        arrays, metadata = build_or_load_view_cache(
            fold=fold,
            pad_artifact=pad_artifacts[fold],
            pad_metadata=pad_metadata[fold],
            pad_oof=pad_oofs[fold],
            checkpoint_root=args.checkpoint_root,
            cache_dir=args.view_cache_dir,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        caches.append(arrays)
        cache_metadata.append(metadata)

    fold_rows = []
    all_files = []
    all_labels = []
    all_pad_predictions = []
    all_view_predictions = []
    for fold in range(NUM_FOLDS):
        artifact = pad_artifacts[fold]
        oof = pad_oofs[fold]
        cache = caches[fold]
        groups = group_indices(artifact)
        enrollment_views, enrollment_group_ids, group_view_counts = (
            expand_enrollment_views(
                cache["train_view_embeddings"],
                cache["train_view_counts"],
                groups,
            )
        )
        scores = multiview_logmeanexp_scores(
            query_views=cache["validation_view_embeddings"],
            query_counts=cache["validation_view_counts"],
            enrollment_views=enrollment_views,
            enrollment_group_ids=enrollment_group_ids,
            group_view_counts=group_view_counts,
            beta=LOCKED_LME_BETA,
            device=device,
            file_batch_size=args.score_batch_size,
        )
        pad_predictions = locked_lme_predictions(
            artifact, oof["embeddings"], oof["competition_probs"]
        )
        view_predictions = decision_predictions(
            head=oof["competition_probs"],
            scores=scores,
            probability_kappa=LOCKED_RAW_KAPPA,
            raw_max_scores=scores.max(axis=1),
        )
        labels = oof["labels"].astype(np.int64)
        baseline = metric_bundle(labels, pad_predictions)
        candidate = metric_bundle(labels, view_predictions)
        baseline_correct = pad_predictions == labels
        candidate_correct = view_predictions == labels
        fold_rows.append({
            "fold": fold,
            "baseline": baseline,
            "candidate": candidate,
            "delta": metric_delta(candidate, baseline),
            "rescued_errors": int(np.sum(~baseline_correct & candidate_correct)),
            "introduced_errors": int(np.sum(baseline_correct & ~candidate_correct)),
            "changed_predictions": int(np.sum(
                pad_predictions != view_predictions
            )),
            "enrollment_real_views": int(len(enrollment_views)),
            "query_real_views": int(cache["validation_view_counts"].sum()),
            "group_view_count_min": int(group_view_counts.min()),
            "group_view_count_median": float(np.median(group_view_counts)),
            "group_view_count_max": int(group_view_counts.max()),
        })
        all_files.append(oof["files"].astype(str))
        all_labels.append(labels)
        all_pad_predictions.append(pad_predictions)
        all_view_predictions.append(view_predictions)

    files = np.concatenate(all_files)
    if len(set(files.tolist())) != len(files):
        raise RuntimeError("OOF validation files overlap across folds")
    labels = np.concatenate(all_labels)
    pad_predictions = np.concatenate(all_pad_predictions)
    view_predictions = np.concatenate(all_view_predictions)
    baseline = metric_bundle(labels, pad_predictions)
    candidate = metric_bundle(labels, view_predictions)
    aggregate = {
        "baseline": baseline,
        "candidate": candidate,
        "delta": metric_delta(candidate, baseline),
        "rescued_errors": int(np.sum(
            (pad_predictions != labels) & (view_predictions == labels)
        )),
        "introduced_errors": int(np.sum(
            (pad_predictions == labels) & (view_predictions != labels)
        )),
        "changed_predictions": int(np.sum(
            pad_predictions != view_predictions
        )),
    }
    if abs(baseline["macro_f1"] - LOCKED_BASELINE_MACRO_F1) > 1e-10:
        raise RuntimeError("Locked pad baseline reproduction failed")
    gate = acceptance_gate(fold_rows, aggregate)
    report = {
        "contract": {
            "hypothesis": (
                "retain real temporal views and score all query/enrollment pairs"
            ),
            "single_changed_variable": (
                "file-mean embedding LME20 -> view-pair LME20"
            ),
            "duplicate_padding_views_excluded": True,
            "view_pair_normalisation": "divide by query_views * enrollment_views",
            "within_group_weighting": (
                "equal weight per real temporal view; longer files may add more views"
            ),
            "weights": "fixed selected Raw CAM++ Control Fold0/1/2 checkpoints",
            "folds": "fixed kfold/folds3/seed42 OOF",
            "lme_beta": LOCKED_LME_BETA,
            "alpha": LOCKED_ALPHA,
            "kappa": LOCKED_RAW_KAPPA,
            "tau": LOCKED_TAU,
            "lambda_unknown": LOCKED_UNKNOWN_WEIGHT,
            "parameter_selection": "none",
            "leaderboard_tuning": False,
        },
        "provenance": {
            "pad_cache_metadata": pad_metadata,
            "view_cache_metadata": cache_metadata,
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
