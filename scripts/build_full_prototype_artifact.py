"""Build the full-enrollment CAM++ artifact for the locked LME-20 policy.

This is an inference artifact, not model training.  It embeds every usable,
duplicate-cleaned competition enrollment file with the fixed Fold-0 CAM++
checkpoint, keeps all known-speaker embeddings, and partitions the unlabeled
enrollment set into the fixed 554 KMeans groups established by the OOF audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_pipeline import SpeakerDataset, prepare_clean_split  # noqa: E402
from src.model_factory import create_model_from_config  # noqa: E402
from src.unknown_clustering import cluster_kmeans  # noqa: E402


NUM_KNOWN = 446
NUM_UNKNOWN_GROUPS = 554
NUM_GROUPS = NUM_KNOWN + NUM_UNKNOWN_GROUPS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_names(names: list[str]) -> str:
    payload = "\n".join(sorted(names)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def l2norm_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / (np.linalg.norm(values, axis=1, keepdims=True) + 1e-12)


def validate_artifact(path: Path) -> dict[str, int]:
    with np.load(path, allow_pickle=False) as data:
        embeddings = data["embeddings"]
        speaker_ids = data["speaker_ids"]
        files = data["files"]
        group_sizes = data["group_sizes"]
    expected = np.arange(1, NUM_GROUPS + 1, dtype=np.int64)
    if embeddings.ndim != 2 or len(embeddings) != len(speaker_ids):
        raise RuntimeError("Prototype artifact has invalid embedding shape")
    if len(files) != len(embeddings):
        raise RuntimeError("Prototype artifact file/embedding count mismatch")
    if not np.array_equal(np.unique(speaker_ids), expected):
        raise RuntimeError("Prototype artifact does not contain dense 1..1000 groups")
    if not np.array_equal(
        group_sizes, np.bincount(speaker_ids, minlength=NUM_GROUPS + 1)[1:]
    ):
        raise RuntimeError("Prototype artifact group sizes are inconsistent")
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.isfinite(embeddings).all() or not np.allclose(norms, 1.0, atol=2e-4):
        raise RuntimeError("Prototype embeddings are not finite and unit-normalised")
    return {
        "files": int(len(files)),
        "embedding_dim": int(embeddings.shape[1]),
        "groups": int(len(np.unique(speaker_ids))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints" / "p0-campp-known446-ood-control-oof-f0"
        / "campp_best_raw.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "processed" / "campp_lme20_full_prototypes.npz",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=ROOT / "data" / "experiments" / "campp_lme20_full_prototypes",
    )
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-eval-windows", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output.with_suffix(".json")
    cluster_map_path = args.output.with_name(
        args.output.stem + "_unknown_clusters.json"
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    checkpoint_class_map = checkpoint["class_map"]
    # The validated Control checkpoint is internally 1001-way (446 known +
    # 554 pseudo-unknown + class zero) and collapses to 447 competition
    # columns at inference.  A legacy 447-way checkpoint is also admissible;
    # embeddings are independent of the speaker-head width.
    if len(checkpoint_class_map) not in {NUM_KNOWN + 1, NUM_GROUPS + 1}:
        raise RuntimeError(
            "Expected a 447- or 1001-entry checkpoint class map, got "
            f"{len(checkpoint_class_map)}"
        )
    data_cfg = config["data"]
    audio_cfg = config["audio"]

    frame, _, rebuilt_class_map = prepare_clean_split(
        labels_path=data_cfg["labels_path"],
        audio_dir=data_cfg["audio_dir"],
        processed_labels=str(args.work_dir / "full_clean_labels.csv"),
        val_per_known=1,
        unknown_val_ratio=0.2,
        min_valid_duration=audio_cfg.get("min_valid_duration", 1.0),
        random_seed=args.seed,
        split_report_path=str(args.work_dir / "full_split_report.json"),
        split_scheme="full",
        fold=0,
        folds=3,
        clean_duplicates=True,
    )
    frame = frame.reset_index(drop=True)
    known_checkpoint_map = {
        label: int(index)
        for label, index in checkpoint_class_map.items()
        if 1 <= int(index) <= NUM_KNOWN
    }
    known_rebuilt_map = {
        label: int(index)
        for label, index in rebuilt_class_map.items()
        if label != "unknown"
    }
    if known_checkpoint_map != known_rebuilt_map:
        raise RuntimeError("Full-data known class map differs from checkpoint class map")
    file_names = frame["audio_file"].astype(str).tolist()
    split_digest = digest_names(file_names)
    checkpoint_sha = sha256_file(checkpoint_path)

    if args.output.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        reusable = (
            metadata.get("checkpoint_sha256") == checkpoint_sha
            and metadata.get("full_file_sha256") == split_digest
            and metadata.get("artifact_sha256") == sha256_file(args.output)
            and metadata.get("seed") == args.seed
            and metadata.get("unknown_groups") == NUM_UNKNOWN_GROUPS
            and metadata.get("max_eval_windows") == args.max_eval_windows
        )
        if reusable:
            print(json.dumps({
                "status": "reused",
                "output": str(args.output),
                "validation": validate_artifact(args.output),
                "metadata": str(metadata_path),
            }, indent=2))
            return 0

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model = create_model_from_config(
        config, num_known_speakers=len(checkpoint_class_map) - 1
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    frame_for_dataset = frame.copy()
    frame_for_dataset["label"] = (
        frame_for_dataset["speaker_id"].map(checkpoint_class_map).astype(int)
    )
    dataset = SpeakerDataset(
        frame_for_dataset,
        data_cfg["audio_dir"],
        sample_rate=audio_cfg["sample_rate"],
        duration_seconds=audio_cfg["duration_seconds"],
        augment=False,
        min_valid_duration=audio_cfg.get("min_valid_duration", 1.0),
        num_train_windows=audio_cfg.get("num_train_windows", 1),
        eval_hop_ratio=audio_cfg.get("eval_hop_ratio", 0.5),
        max_eval_windows=args.max_eval_windows,
        eval_speech_aware=audio_cfg.get("eval_speech_aware", False),
        speech_relative_db=audio_cfg.get("speech_relative_db", 35.0),
        short_audio_mode=audio_cfg.get("short_audio_mode", "pad"),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    chunks = []
    with torch.inference_mode():
        for windows, _ in tqdm(loader, desc="Full enrollment embeddings"):
            chunks.append(model.embed(windows.to(device)).cpu().numpy())
    embeddings = l2norm_rows(np.concatenate(chunks, axis=0))

    labels = frame_for_dataset["label"].to_numpy(np.int64)
    unknown_mask = frame_for_dataset["speaker_id"].eq("unknown").to_numpy()
    if np.any((~unknown_mask) & ((labels < 1) | (labels > NUM_KNOWN))):
        raise RuntimeError("Known full-data labels fall outside 1..446")
    unknown_cluster_ids = cluster_kmeans(
        embeddings[unknown_mask], NUM_UNKNOWN_GROUPS, seed=args.seed
    ).astype(np.int64)
    if set(unknown_cluster_ids.tolist()) != set(range(NUM_UNKNOWN_GROUPS)):
        raise RuntimeError("KMeans did not produce 554 non-empty unknown groups")

    speaker_ids = labels.copy()
    speaker_ids[unknown_mask] = NUM_KNOWN + 1 + unknown_cluster_ids
    full_cluster_ids = np.full(len(frame), -1, dtype=np.int64)
    full_cluster_ids[unknown_mask] = unknown_cluster_ids
    group_sizes = np.bincount(speaker_ids, minlength=NUM_GROUPS + 1)[1:]
    if len(group_sizes) != NUM_GROUPS or np.any(group_sizes == 0):
        raise RuntimeError("Full prototype artifact has an empty internal group")

    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            embeddings=embeddings.astype(np.float32),
            speaker_ids=speaker_ids.astype(np.int64),
            files=np.asarray(file_names, dtype=str),
            competition_labels=np.where(unknown_mask, 0, labels).astype(np.int64),
            unknown_cluster_ids=full_cluster_ids,
            group_sizes=group_sizes.astype(np.int64),
        )
    os.replace(temporary, args.output)

    cluster_map = {
        name: int(cluster_id)
        for name, cluster_id in zip(
            frame_for_dataset.loc[unknown_mask, "audio_file"].astype(str),
            unknown_cluster_ids,
        )
    }
    cluster_map_path.write_text(
        json.dumps(cluster_map, sort_keys=True), encoding="utf-8"
    )
    validation = validate_artifact(args.output)
    metadata = {
        "artifact": str(args.output),
        "artifact_sha256": sha256_file(args.output),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "full_file_sha256": split_digest,
        "files": int(len(frame)),
        "known_files": int((~unknown_mask).sum()),
        "unknown_files": int(unknown_mask.sum()),
        "known_groups": NUM_KNOWN,
        "unknown_groups": NUM_UNKNOWN_GROUPS,
        "total_groups": NUM_GROUPS,
        "group_size_min": int(group_sizes.min()),
        "group_size_mean": float(group_sizes.mean()),
        "group_size_max": int(group_sizes.max()),
        "seed": args.seed,
        "max_eval_windows": args.max_eval_windows,
        "unknown_cluster_map": str(cluster_map_path),
        "unknown_cluster_map_sha256": sha256_file(cluster_map_path),
        "validation": validation,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": "built",
        "output": str(args.output),
        "metadata": str(metadata_path),
        "validation": validation,
        "artifact_sha256": metadata["artifact_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
