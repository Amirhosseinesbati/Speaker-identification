"""Cache raw open-set evidence on one common full-data manifest.

The cache keeps the competition probabilities as well as the evidence that is
normally destroyed by the 447-way collapse: OOD-head probability, the complete
speaker-head distribution (including pseudo-unknown clusters), speaker
embeddings, and window agreement.  All models are evaluated on the same files
and in the same order so decision rules and ensembles can be audited without
rerunning the encoders.

Example::

    uv run --no-sync python scripts/dump_decision_evidence.py \
      --model historical="checkpoints/modelrigestry/campp_best (5).pt"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cli_utils import setup_utf8_stdio  # noqa: E402
from src.data_pipeline import SpeakerDataset  # noqa: E402
from src.model_factory import create_model_from_config  # noqa: E402


setup_utf8_stdio()

DEFAULT_MANIFEST = (
    ROOT
    / "data"
    / "processed"
    / "forensics"
    / "historical_4a47c98"
    / "full_manifest.csv"
)
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "decision_evidence"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_model(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--model must be TAG=CHECKPOINT")
    tag, raw_path = value.split("=", 1)
    tag = tag.strip()
    path = Path(raw_path.strip())
    if not path.is_absolute():
        path = ROOT / path
    if not tag or not path.exists():
        raise argparse.ArgumentTypeError(f"Invalid model specification: {value}")
    return tag, path


def l2_normalize(values: torch.Tensor) -> torch.Tensor:
    return functional.normalize(values.float(), p=2, dim=-1)


@torch.inference_mode()
def dump_model(
    tag: str,
    checkpoint_path: Path,
    manifest: pd.DataFrame,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
) -> dict:
    checkpoint_hash = sha256(checkpoint_path)
    cache_path = output_dir / f"{tag}.npz"
    metadata_path = output_dir / f"{tag}.json"
    if cache_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("checkpoint_sha256") == checkpoint_hash
            and metadata.get("n_files") == len(manifest)
            and metadata.get("manifest_audio_sha256")
            == hashlib.sha256(
                "\n".join(manifest["audio_file"].astype(str)).encode("utf-8")
            ).hexdigest()
        ):
            print(f"  ✓ [{tag}] evidence cache is current — skipping")
            return metadata

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    class_map = checkpoint["class_map"]
    model = create_model_from_config(config, num_known_speakers=len(class_map) - 1)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    frame = manifest.copy()
    # Evaluation always uses the fixed competition labels.  Unknown pseudo
    # identities from a cluster checkpoint must not leak into ground truth.
    frame["label"] = np.where(
        frame["speaker_id"].astype(str) == "unknown",
        0,
        frame["speaker_id"].map(class_map).fillna(0),
    ).astype(int)
    audio = config["audio"]
    data = config["data"]
    dataset = SpeakerDataset(
        frame,
        data["audio_dir"],
        sample_rate=audio["sample_rate"],
        duration_seconds=audio["duration_seconds"],
        augment=False,
        num_train_windows=audio.get("num_train_windows", 1),
        eval_hop_ratio=audio.get("eval_hop_ratio", 0.5),
        max_eval_windows=audio.get("max_eval_windows", 8),
        eval_speech_aware=audio.get("eval_speech_aware", False),
        speech_relative_db=audio.get("speech_relative_db", 35.0),
        short_audio_mode=audio.get("short_audio_mode", "pad"),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_head: list[np.ndarray] = []
    all_speaker: list[np.ndarray] = []
    all_ood: list[np.ndarray] = []
    all_embeddings: list[np.ndarray] = []
    all_agreement: list[np.ndarray] = []
    num_known = int(model.num_output_classes) - 1
    num_tail = int(model.num_unknown_clusters)

    for windows, _ in tqdm(loader, desc=f"  [{tag}] decision evidence"):
        # (B, W, 1, T) -> one encoder pass over B*W windows.
        batch, window_count = int(windows.shape[0]), int(windows.shape[1])
        flat = windows.flatten(0, 1).to(device)
        with torch.autocast(
            device_type="cuda",
            enabled=device.type == "cuda",
        ):
            hidden, _ = model.encoder(flat)
            pooled = model.pooling(hidden)
            ood_logits = model.head_ood(pooled) if model.head_ood is not None else None
            speaker_logits = model.head_speaker(pooled)
            if hasattr(model.head_speaker, "embedding_proj"):
                raw_embedding = model.head_speaker.embedding_proj(pooled)
            else:
                raw_embedding = pooled

        speaker_window = torch.softmax(speaker_logits.float(), dim=1)
        speaker_file = speaker_window.view(batch, window_count, -1).mean(dim=1)
        if ood_logits is None:
            ood_window = torch.zeros(
                len(flat), 1, dtype=torch.float32, device=device,
            )
        else:
            ood_window = torch.sigmoid(ood_logits.float())
        ood_file = ood_window.view(batch, window_count, 1).mean(dim=1)

        scaled = (1.0 - ood_window) * speaker_window
        known_scaled = scaled[:, :num_known]
        if num_tail:
            unknown = ood_window + scaled[:, num_known:].sum(dim=1, keepdim=True)
        else:
            unknown = ood_window
        competition_window = torch.cat([unknown, known_scaled], dim=1)
        competition_file = competition_window.view(batch, window_count, -1).mean(dim=1)
        competition_file /= competition_file.sum(dim=1, keepdim=True).clamp_min(1e-12)

        embedding_file = raw_embedding.float().view(batch, window_count, -1).mean(dim=1)
        embedding_file = l2_normalize(embedding_file)
        window_top = speaker_window[:, :num_known].argmax(dim=1).view(batch, window_count)
        aggregate_top = speaker_file[:, :num_known].argmax(dim=1, keepdim=True)
        agreement = (window_top == aggregate_top).float().mean(dim=1)

        all_head.append(competition_file.cpu().numpy().astype(np.float32))
        all_speaker.append(speaker_file.cpu().numpy().astype(np.float32))
        all_ood.append(ood_file.squeeze(1).cpu().numpy().astype(np.float32))
        all_embeddings.append(embedding_file.cpu().numpy().astype(np.float32))
        all_agreement.append(agreement.cpu().numpy().astype(np.float32))

    arrays = {
        "audio_file": manifest["audio_file"].astype(str).to_numpy(),
        "speaker_id": manifest["speaker_id"].astype(str).to_numpy(),
        "label": frame["label"].to_numpy(dtype=np.int64),
        "historical_split": manifest["historical_split"].astype(str).to_numpy(),
        "head_probs": np.concatenate(all_head),
        "speaker_probs": np.concatenate(all_speaker),
        "ood_prob": np.concatenate(all_ood),
        "embedding": np.concatenate(all_embeddings),
        "window_agreement": np.concatenate(all_agreement),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **arrays)
    metadata = {
        "tag": tag,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": checkpoint_hash,
        "created_at": datetime.now().isoformat(),
        "n_files": len(manifest),
        "num_known": num_known,
        "num_unknown_clusters": num_tail,
        "speaker_width": int(arrays["speaker_probs"].shape[1]),
        "embedding_dim": int(arrays["embedding"].shape[1]),
        "manifest_audio_sha256": hashlib.sha256(
            "\n".join(manifest["audio_file"].astype(str)).encode("utf-8")
        ).hexdigest(),
        "cache": str(cache_path.relative_to(ROOT)),
        "cache_bytes": cache_path.stat().st_size,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"  ✓ [{tag}] {cache_path.relative_to(ROOT)} ({cache_path.stat().st_size / 2**20:.1f} MiB)")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", type=parse_model, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    manifest = pd.read_csv(manifest_path)
    required = {"audio_file", "speaker_id", "historical_split"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"Manifest lacks columns: {sorted(required - set(manifest.columns))}")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if str(device) == "auto":
        device = torch.device("cpu")
    print(f"Decision evidence: {len(manifest):,} common files | device={device}")
    summaries = [
        dump_model(tag, path, manifest, output_dir, device, args.batch_size)
        for tag, path in args.model
    ]
    (output_dir / "manifest.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
