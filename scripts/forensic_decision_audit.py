"""Forensic audit of historical open-set decision artifacts.

This script is intentionally read-only.  It inventories the cached validation
artifacts and evaluates the exact legacy centroid/head decision formula against
every compatible unknown-centroid file.  The output is used to identify the
real historical control before any new decision rule is tuned.

Run with::

    uv run --no-sync python scripts/forensic_decision_audit.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PROCESSED = ROOT / "data" / "processed"
REGISTRY = ROOT / "checkpoints" / "modelrigestry"
NUM_CLASSES = 447
HISTORICAL_COMMIT = "4a47c98"
HISTORICAL_CHECKPOINT = REGISTRY / "campp_best (5).pt"
HISTORICAL_DIR = PROCESSED / "forensics" / "historical_4a47c98"


def sha12(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


def artifact_metadata(path: Path) -> dict[str, Any]:
    stat = path.stat()
    meta: dict[str, Any] = {
        "path": str(path.relative_to(ROOT)),
        "bytes": stat.st_size,
        "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "sha256_12": sha12(path),
    }
    try:
        if path.suffix == ".npy":
            meta["shape"] = list(np.load(path, mmap_mode="r").shape)
        elif path.suffix == ".npz":
            with np.load(path) as data:
                meta["arrays"] = {key: list(data[key].shape) for key in data.files}
        elif path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            meta["json_type"] = type(payload).__name__
            if isinstance(payload, dict):
                meta["keys"] = list(payload)[:20]
    except Exception as exc:  # pragma: no cover - diagnostic only
        meta["inspection_error"] = repr(exc)
    return meta


def inventory() -> list[dict[str, Any]]:
    patterns = (
        "val_probs_campp*.npy",
        "val_emb_campp*.npy",
        "val_labels*.npy",
        "centroids_campp*.npz",
        "centroids_unknown_campp*.npz",
        "decision_config.json",
    )
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(PROCESSED.glob(pattern))
    paths.extend(REGISTRY.glob("campp_best*.pt"))
    return [artifact_metadata(path) for path in sorted(set(paths))]


def zip_inventory() -> list[dict[str, Any]]:
    """List decision/model artifacts embedded in project-level submissions."""
    reports: list[dict[str, Any]] = []
    interesting = (
        "decision_config.json",
        "ensemble_fusion_weights.json",
        ".pt",
        ".npz",
        "unknown_clusters.json",
    )
    for path in sorted(ROOT.glob("*.zip")):
        row: dict[str, Any] = {
            **artifact_metadata(path),
            "members": [],
        }
        try:
            with zipfile.ZipFile(path) as archive:
                for info in archive.infolist():
                    if any(token in info.filename for token in interesting):
                        row["members"].append(
                            {
                                "name": info.filename,
                                "bytes": info.file_size,
                                "crc32": f"{info.CRC:08x}",
                                "date_time": list(info.date_time),
                            }
                        )
        except zipfile.BadZipFile as exc:
            row["zip_error"] = repr(exc)
        reports.append(row)
    return reports


def softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = values - values.max(axis=axis, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / (exponent.sum(axis=axis, keepdims=True) + 1e-12)


def normalize_rows(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def load_centroid_npz(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    with np.load(path) as data:
        centroids = data["centroids"].astype(np.float64)
        speaker_ids = (
            data["speaker_ids"].astype(np.int64)
            if "speaker_ids" in data.files
            else None
        )
    return normalize_rows(centroids), speaker_ids


def centroids_from_assignments(
    embeddings: np.ndarray,
    assignments: np.ndarray,
    expected_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    sizes: list[int] = []
    for identity in expected_ids:
        selected = embeddings[assignments == identity]
        if not len(selected):
            raise RuntimeError(f"No embeddings found for centroid id {identity}")
        rows.append(selected.mean(axis=0))
        sizes.append(len(selected))
    return normalize_rows(np.stack(rows)).astype(np.float32), np.asarray(sizes)


def historical_map_from_git() -> dict[str, int]:
    result = subprocess.run(
        ["git", "show", f"{HISTORICAL_COMMIT}:submission/unknown_clusters.json"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    )
    return {str(key): int(value) for key, value in json.loads(result.stdout).items()}


def historical_split_from_git(checkpoint: dict[str, Any]):
    """Execute the data-pipeline implementation stored in the LB commit."""
    import importlib.util

    HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)
    source = subprocess.run(
        ["git", "show", f"{HISTORICAL_COMMIT}:src/data_pipeline.py"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout
    source_path = HISTORICAL_DIR / "data_pipeline_4a47c98.py"
    source_path.write_bytes(source)
    spec = importlib.util.spec_from_file_location("historical_data_pipeline", source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the historical data pipeline")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    config = checkpoint["config"]
    audio = config["audio"]
    data = config["data"]
    split = data.get("split", {})
    return module.prepare_clean_split(
        labels_path=data["labels_path"],
        audio_dir=data["audio_dir"],
        processed_labels=str(HISTORICAL_DIR / "cleaned_labels.csv"),
        val_per_known=1,
        unknown_val_ratio=0.2,
        min_valid_duration=audio.get("min_valid_duration", 1.0),
        random_seed=int(split.get("seed", 42)),
        split_report_path=str(HISTORICAL_DIR / "split_report.json"),
        split_scheme=str(split.get("scheme", "single")),
        fold=int(split.get("fold", 0)),
        folds=int(split.get("folds", 3)),
    )


def extract_missing_embeddings(
    checkpoint: dict[str, Any],
    rows,
    device,
) -> np.ndarray:
    """Embed historical files omitted by the modern full-data cleaner."""
    import torch
    from torch.utils.data import DataLoader

    from src.data_pipeline import SpeakerDataset
    from src.model_factory import create_model_from_config

    if not len(rows):
        return np.empty((0, 192), dtype=np.float32)
    config = checkpoint["config"]
    class_map = checkpoint["class_map"]
    rows = rows.copy()
    rows["label"] = rows["speaker_id"].map(class_map).astype(int)
    audio = config["audio"]
    data = config["data"]
    dataset = SpeakerDataset(
        rows,
        data["audio_dir"],
        sample_rate=audio["sample_rate"],
        duration_seconds=audio["duration_seconds"],
        augment=False,
        num_train_windows=audio.get("num_train_windows", 1),
        eval_hop_ratio=audio.get("eval_hop_ratio", 0.5),
        max_eval_windows=audio.get("max_eval_windows", 8),
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    model = create_model_from_config(config, num_known_speakers=len(class_map) - 1)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for windows, _ in loader:
            chunks.append(model.embed(windows.to(device)).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def rebuild_historical_artifacts(force_embeddings: bool = False) -> dict[str, Any]:
    """Rebuild the Aug-19 centroids in the exact checkpoint embedding space.

    The historical unknown assignment is read directly from Git.  No modern
    KMeans rerun is allowed here because that would silently change the control.
    """
    import torch

    from src.unknown_clustering import _extract_train_embs

    if not HISTORICAL_CHECKPOINT.exists():
        raise FileNotFoundError(HISTORICAL_CHECKPOINT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(HISTORICAL_CHECKPOINT, map_location="cpu", weights_only=False)
    full_embeddings, _, full_files = _extract_train_embs(
        str(HISTORICAL_CHECKPOINT),
        device,
        force=force_embeddings,
        split_scheme="full",
        scope="full",
    )
    train_df, val_df, _ = historical_split_from_git(checkpoint)
    train_df = train_df.copy()
    train_df["label"] = train_df["speaker_id"].map(checkpoint["class_map"]).astype(int)
    train_manifest_path = HISTORICAL_DIR / "train_manifest.csv"
    train_df.to_csv(train_manifest_path, index=False)
    files = train_df["audio_file"].astype(str).tolist()
    labels = train_df["label"].to_numpy(dtype=np.int64)

    full_lookup = {
        str(name): full_embeddings[index]
        for index, name in enumerate(full_files)
    }
    missing_files = [name for name in files if name not in full_lookup]
    missing_df = train_df[train_df["audio_file"].isin(missing_files)].copy()
    missing_embeddings = extract_missing_embeddings(checkpoint, missing_df, device)
    missing_lookup = {
        str(name): missing_embeddings[index]
        for index, name in enumerate(missing_df["audio_file"].astype(str))
    }
    all_lookup = {**full_lookup, **missing_lookup}
    still_missing = [name for name in files if name not in all_lookup]
    if still_missing:
        raise RuntimeError(f"Could not embed {len(still_missing)} historical train files")
    embeddings = np.stack([all_lookup[name] for name in files])
    embeddings = normalize_rows(embeddings.astype(np.float64)).astype(np.float32)

    val_df = val_df.copy()
    val_df["label"] = val_df["speaker_id"].map(checkpoint["class_map"]).astype(int)
    val_manifest_path = HISTORICAL_DIR / "val_manifest.csv"
    val_df.to_csv(val_manifest_path, index=False)
    full_manifest_path = HISTORICAL_DIR / "full_manifest.csv"
    full_manifest = __import__("pandas").concat(
        [train_df.assign(historical_split="train"), val_df.assign(historical_split="val")],
        ignore_index=True,
    )
    full_manifest.to_csv(full_manifest_path, index=False)
    cached_val_labels_path = PROCESSED / "val_labels.npy"
    cached_val_labels = np.load(cached_val_labels_path).astype(np.int64)
    historical_val_labels = val_df["label"].to_numpy(dtype=np.int64)
    val_cache_order_matches = bool(
        len(cached_val_labels) == len(historical_val_labels)
        and np.array_equal(cached_val_labels, historical_val_labels)
    )

    known_ids = np.arange(1, NUM_CLASSES, dtype=np.int64)
    known_mask = labels > 0
    known_centroids, known_sizes = centroids_from_assignments(
        embeddings[known_mask], labels[known_mask], known_ids,
    )

    historical_map = historical_map_from_git()
    unknown_positions = {name: index for index, name in enumerate(files) if labels[index] == 0}
    missing_map_files = sorted(set(historical_map) - set(unknown_positions))
    unmapped_unknown_files = sorted(set(unknown_positions) - set(historical_map))
    usable_files = [name for name in historical_map if name in unknown_positions]
    if missing_map_files or unmapped_unknown_files:
        raise RuntimeError(
            "Historical map/train split mismatch: "
            f"missing_map_files={len(missing_map_files)}, "
            f"unmapped_unknown_files={len(unmapped_unknown_files)}"
        )
    unknown_embeddings = np.stack(
        [embeddings[unknown_positions[name]] for name in usable_files]
    )
    cluster_ids = np.asarray([historical_map[name] for name in usable_files], dtype=np.int64)
    expected_clusters = np.arange(554, dtype=np.int64)
    unknown_centroids, unknown_sizes = centroids_from_assignments(
        unknown_embeddings, cluster_ids, expected_clusters,
    )

    HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)
    known_path = HISTORICAL_DIR / "centroids_campp.npz"
    unknown_path = HISTORICAL_DIR / "centroids_unknown_campp.npz"
    map_path = HISTORICAL_DIR / "unknown_clusters.json"
    np.savez_compressed(
        known_path,
        centroids=known_centroids,
        speaker_ids=known_ids,
        cluster_sizes=known_sizes,
    )
    np.savez_compressed(
        unknown_path,
        centroids=unknown_centroids,
        cluster_ids=expected_clusters,
        cluster_sizes=unknown_sizes,
    )
    map_path.write_text(
        json.dumps(historical_map, ensure_ascii=False, indent=0),
        encoding="utf-8",
    )
    return {
        "checkpoint": artifact_metadata(HISTORICAL_CHECKPOINT),
        "device": str(device),
        "n_train": len(files),
        "n_val": len(val_df),
        "n_known_train": int(known_mask.sum()),
        "n_unknown_train": int((~known_mask).sum()),
        "n_embeddings_reused_from_modern_full_cache": len(files) - len(missing_files),
        "n_embeddings_reextracted": len(missing_files),
        "val_manifest": str(val_manifest_path.relative_to(ROOT)),
        "train_manifest": str(train_manifest_path.relative_to(ROOT)),
        "full_manifest": str(full_manifest_path.relative_to(ROOT)),
        "val_cache_label_order_matches_historical_split": val_cache_order_matches,
        "historical_map_entries": len(historical_map),
        "historical_map_sha256_12": sha12(map_path),
        "known_centroids": artifact_metadata(known_path),
        "unknown_centroids": artifact_metadata(unknown_path),
        "known_cluster_size": {
            "min": int(known_sizes.min()),
            "mean": float(known_sizes.mean()),
            "max": int(known_sizes.max()),
        },
        "unknown_cluster_size": {
            "min": int(unknown_sizes.min()),
            "mean": float(unknown_sizes.mean()),
            "max": int(unknown_sizes.max()),
        },
    }


def centroid_probabilities(
    embeddings: np.ndarray,
    known_centroids: np.ndarray,
    known_ids: np.ndarray,
    unknown_centroids: np.ndarray,
    kappa: float,
) -> tuple[np.ndarray, np.ndarray]:
    embeddings = normalize_rows(embeddings.astype(np.float64))
    centroids = np.vstack([known_centroids, unknown_centroids])
    unknown_ids = np.arange(
        int(known_ids.max()) + 1,
        int(known_ids.max()) + 1 + len(unknown_centroids),
        dtype=np.int64,
    )
    speaker_ids = np.concatenate([known_ids, unknown_ids])

    cosine = embeddings @ centroids.T
    max_cosine = cosine.max(axis=1)
    mass = softmax(kappa * cosine, axis=1)
    p_unknown_distance = np.clip(1.0 - max_cosine, 0.0, 1.0)
    wide = np.zeros((len(embeddings), int(speaker_ids.max()) + 1), dtype=np.float64)
    wide[:, 0] = p_unknown_distance
    wide[:, speaker_ids] = (1.0 - p_unknown_distance[:, None]) * mass
    wide /= wide.sum(axis=1, keepdims=True) + 1e-12

    collapsed = np.zeros((len(embeddings), NUM_CLASSES), dtype=np.float64)
    collapsed[:, 0] = wide[:, 0] + wide[:, NUM_CLASSES:].sum(axis=1)
    collapsed[:, 1:] = wide[:, 1:NUM_CLASSES]
    collapsed /= collapsed.sum(axis=1, keepdims=True) + 1e-12
    return collapsed, max_cosine


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    known = y_true != 0
    unknown = ~known
    return {
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=np.arange(NUM_CLASSES),
                average="macro",
                zero_division=0,
            )
        ),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "known_accuracy": float(accuracy_score(y_true[known], y_pred[known])),
        "unknown_recall": float(np.mean(y_pred[unknown] == 0)),
        "known_to_unknown": int(np.sum(known & (y_pred == 0))),
        "known_to_wrong_known": int(np.sum(known & (y_pred != 0) & (y_pred != y_true))),
        "unknown_to_known": int(np.sum(unknown & (y_pred != 0))),
    }


def evaluate_legacy_cache() -> list[dict[str, Any]]:
    head_path = PROCESSED / "val_probs_campp_best (5).pt.npy"
    emb_path = PROCESSED / "val_emb_campp_best (5).pt.npy"
    known_path = PROCESSED / "centroids_campp_best (5).pt.npz"
    label_candidates = [
        PROCESSED / "val_labels_old.npy",
        PROCESSED / "val_labels.npy",
    ]
    required = [head_path, emb_path, known_path]
    if not all(path.exists() for path in required):
        return [{"error": "legacy cached artifacts are incomplete"}]

    head = np.load(head_path).astype(np.float64)
    embeddings = np.load(emb_path).astype(np.float64)
    known_centroids, known_ids = load_centroid_npz(known_path)
    if known_ids is None:
        known_ids = np.arange(1, len(known_centroids) + 1, dtype=np.int64)

    unknown_candidates = sorted(PROCESSED.glob("centroids_unknown_campp*.npz"))
    historical_known = HISTORICAL_DIR / "centroids_campp.npz"
    historical_unknown = HISTORICAL_DIR / "centroids_unknown_campp.npz"
    centroid_pairs: list[tuple[Path, Path]] = [
        (known_path, path) for path in unknown_candidates
    ]
    if historical_known.exists() and historical_unknown.exists():
        centroid_pairs.append((historical_known, historical_unknown))
    results: list[dict[str, Any]] = []
    for labels_path in label_candidates:
        if not labels_path.exists():
            continue
        labels = np.load(labels_path).astype(np.int64)
        if len(labels) != len(head):
            continue
        for candidate_known_path, unknown_path in centroid_pairs:
            known_centroids, known_ids = load_centroid_npz(candidate_known_path)
            if known_ids is None:
                known_ids = np.arange(1, len(known_centroids) + 1, dtype=np.int64)
            unknown_centroids, _ = load_centroid_npz(unknown_path)
            centroid_probs, max_cosine = centroid_probabilities(
                embeddings,
                known_centroids,
                known_ids,
                unknown_centroids,
                kappa=24.0,
            )
            fused = 0.35 * head + 0.65 * centroid_probs
            fused[:, 0] *= 0.5
            fused /= fused.sum(axis=1, keepdims=True) + 1e-12
            prediction = fused.argmax(axis=1).astype(np.int64)
            prediction[max_cosine < 0.0] = 0
            results.append(
                {
                    "labels": str(labels_path.relative_to(ROOT)),
                    "known_centroids": str(candidate_known_path.relative_to(ROOT)),
                    "unknown_centroids": str(unknown_path.relative_to(ROOT)),
                    "unknown_k": len(unknown_centroids),
                    "legacy_params": {
                        "alpha": 0.35,
                        "kappa": 24.0,
                        "tau": 0.0,
                        "lambda_unknown": 0.5,
                    },
                    **metrics(labels, prediction),
                }
            )
    return sorted(results, key=lambda row: float(row.get("macro_f1", -1)), reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "generated" / "forensic_decision_audit.json",
    )
    parser.add_argument(
        "--rebuild-historical",
        action="store_true",
        help="Rebuild the exact Aug-19 centroid artifacts before evaluation.",
    )
    parser.add_argument(
        "--force-embeddings",
        action="store_true",
        help="Force the historical checkpoint embedding pass even if cached.",
    )
    args = parser.parse_args()

    rebuilt = None
    if args.rebuild_historical:
        rebuilt = rebuild_historical_artifacts(force_embeddings=args.force_embeddings)
    report = {
        "generated_at": datetime.now().isoformat(),
        "historical_rebuild": rebuilt,
        "inventory": inventory(),
        "zip_inventory": zip_inventory(),
        "legacy_cache_evaluations": evaluate_legacy_cache(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
