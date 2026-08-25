"""Self-describing checkpoint and MLflow artifact bundle utilities."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch
import yaml
import numpy as np


FORMAT_VERSION = "speaker-id-checkpoint/v2"


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _versions() -> dict[str, str]:
    names = ("torch", "torchaudio", "numpy", "pandas", "scikit-learn",
             "speechbrain", "modelscope", "mlflow")
    result = {"python": platform.python_version()}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result


def checkpoint_metadata(
    config: dict,
    class_map: dict,
    metrics: Optional[dict] = None,
    history: Optional[list] = None,
) -> dict:
    model_cfg = config.get("model", {}) or {}
    competition_known = int(model_cfg.get("competition_num_known", 446))
    pseudo_count = max(0, len(class_map) - 1 - competition_known)
    return {
        "format_version": FORMAT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": _git_revision(),
        "architecture": {
            "encoder": model_cfg.get("encoder_type"),
            "pooling": model_cfg.get("pooling_type"),
            "speaker_head": model_cfg.get("speaker_head_type"),
            "ood_head": bool(model_cfg.get("ood_head", True)),
            "competition_known_classes": competition_known,
            "pseudo_ood_classes": pseudo_count,
            "metric_classes": len(class_map) - 1,
            "competition_output_classes": competition_known + 1,
        },
        "target_schema": {
            "metric_label": "0=unmapped unknown; 1..K=metric identity",
            "is_ood": f"metric_label==0 or metric_label>{competition_known}",
            "competition_label": f"0=unknown; 1..{competition_known}=known speaker",
        },
        "audio_policy": config.get("audio", {}),
        "split": (config.get("data", {}) or {}).get("split", {}),
        "metrics": metrics or {},
        "history_epochs": len(history or []),
        "versions": _versions(),
    }


def enrich_checkpoint(
    checkpoint: dict,
    config: dict,
    class_map: dict,
    metrics: Optional[dict] = None,
    history: Optional[list] = None,
) -> dict:
    """Return a checkpoint that carries enough context for offline analysis."""
    out = dict(checkpoint)
    out["config"] = config
    out["class_map"] = class_map
    out["metadata"] = checkpoint_metadata(config, class_map, metrics, history)
    if metrics is not None:
        out["final_metrics"] = metrics
    if history is not None:
        out["training_history"] = history
    return out


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_training_bundle(
    checkpoint_path: str | Path,
    config: dict,
    class_map: dict,
    history: list,
    metrics: dict,
) -> Path:
    """Create a compact, human-readable sidecar bundle next to a checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    bundle = checkpoint_path.with_suffix("")
    bundle = bundle.parent / f"{bundle.name}_bundle"
    bundle.mkdir(parents=True, exist_ok=True)

    (bundle / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    (bundle / "class_map.json").write_text(
        json.dumps(class_map, indent=2, ensure_ascii=False), encoding="utf-8")
    (bundle / "training_history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    metadata = checkpoint_metadata(config, class_map, metrics, history)
    (bundle / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    cluster_path = Path(str((config.get("model", {}) or {}).get(
        "unknown_cluster_path", "")))
    if cluster_path.is_file():
        (bundle / "unknown_clusters.json").write_bytes(cluster_path.read_bytes())

    card = f"""# Speaker identification model

- Checkpoint: `{checkpoint_path.name}`
- Format: `{FORMAT_VERSION}`
- Encoder: `{metadata['architecture']['encoder']}`
- Strategy: metric head with {metadata['architecture']['pseudo_ood_classes']} pseudo-OOD identities; OOD head={metadata['architecture']['ood_head']}
- Competition output: {metadata['architecture']['competition_output_classes']} classes
- Best Macro-F1: {metrics.get('best_val_macro_f1', metrics.get('macro_f1', 'n/a'))}
- Git revision: `{metadata['git_revision']}`

The checkpoint embeds the resolved config, class map, target schema, metrics,
training history and package versions.  This directory adds readable copies
for MLflow/DagsHub inspection without loading PyTorch.
"""
    (bundle / "MODEL_CARD.md").write_text(card, encoding="utf-8")

    manifest = {
        "format_version": FORMAT_VERSION,
        "checkpoint": str(checkpoint_path),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "files": sorted(p.name for p in bundle.iterdir() if p.is_file()),
    }
    (bundle / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return bundle


def save_oof_predictions(
    bundle_dir: str | Path,
    files: list[str],
    labels: torch.Tensor,
    speaker_logits: torch.Tensor,
    ood_logits: Optional[torch.Tensor],
    num_unknown_clusters: int,
    split: dict,
) -> Path:
    """Persist fold predictions needed for unbiased concatenated OOF scoring."""
    bundle_dir = Path(bundle_dir)
    path = bundle_dir / "oof_predictions.npz"
    np.savez_compressed(
        path,
        files=np.asarray(files, dtype=str),
        labels=labels.detach().cpu().numpy().astype(np.int64),
        speaker_logits=speaker_logits.detach().cpu().numpy().astype(np.float32),
        ood_logits=(ood_logits.detach().cpu().numpy().astype(np.float32)
                    if ood_logits is not None else np.empty((len(files), 0), dtype=np.float32)),
        num_unknown_clusters=np.asarray([int(num_unknown_clusters)], dtype=np.int64),
        split_scheme=np.asarray([str(split.get("scheme", "single"))]),
        split_fold=np.asarray([int(split.get("fold", 0))], dtype=np.int64),
        split_folds=np.asarray([int(split.get("folds", 1))], dtype=np.int64),
        split_seed=np.asarray([int(split.get("seed", 42))], dtype=np.int64),
    )
    manifest_path = bundle_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files"] = sorted(
            p.name for p in bundle_dir.iterdir() if p.is_file())
        manifest["oof_predictions_sha256"] = _sha256(path)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
