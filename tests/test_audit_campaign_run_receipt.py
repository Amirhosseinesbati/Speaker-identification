import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.audit_campaign_run_receipt import audit_campaign_run


PROFILE = "paired-control"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    config_path = tmp_path / "configs" / "experiments" / f"{PROFILE}.yaml"
    checkpoint_dir = tmp_path / "checkpoints" / PROFILE
    bundle = checkpoint_dir / "campp_best_bundle" / "oof_predictions.npz"
    config_path.parent.mkdir(parents=True)
    bundle.parent.mkdir(parents=True)
    config_path.write_text("training:\n  epochs: 2\n", encoding="utf-8")
    config = {
        "data": {"split": {"scheme": "kfold", "folds": 3, "fold": 0, "seed": 42}},
        "model": {"competition_num_known": 2, "num_unknown_clusters": 1},
        "logging": {"checkpoint_dir": f"checkpoints/{PROFILE}"},
        "training": {"epochs": 2, "selection_variant": "raw"},
    }
    checkpoint = {
        "config": config,
        "epoch": 2,
        "weight_variant": "raw",
        "class_map": {"a": 0, "b": 1, "u": 2, "unknown": 3},
        "training_history": [{"epoch": 1}, {"epoch": 2}],
        "model_state_dict": {},
    }
    best = checkpoint_dir / "campp_best.pt"
    raw = checkpoint_dir / "campp_best_raw.pt"
    torch.save(checkpoint, best)
    torch.save(checkpoint, raw)
    probs = np.asarray([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]], np.float32)
    np.savez_compressed(
        bundle,
        files=np.asarray(["a.wav", "b.wav"]),
        labels=np.asarray([0, 2]),
        speaker_logits=np.zeros((2, 2), np.float32),
        ood_logits=np.zeros((2, 1), np.float32),
        competition_probs=probs,
        embeddings=np.zeros((2, 4), np.float32),
        split_scheme=np.asarray(["kfold"]),
        split_folds=np.asarray([3]),
        split_fold=np.asarray([0]),
        split_seed=np.asarray([42]),
    )
    artifacts = []
    for path in (config_path, best, raw, bundle):
        artifacts.append({
            "path": path.relative_to(tmp_path).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha(path),
        })
    state = {
        "completed_runs": [{
            "profile": PROFILE,
            "status": "complete",
            "exit_code": 0,
            "git_commit": "abc123",
            "config_sha256": _sha(config_path),
            "started_at_utc": "2026-01-01T00:00:00Z",
            "finished_at_utc": "2026-01-01T01:00:00Z",
            "artifacts": artifacts,
        }]
    }
    state_path = tmp_path / "campaign_state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    return state_path, bundle


def test_audit_campaign_run_verifies_receipts_checkpoint_and_oof(tmp_path):
    state_path, _ = _fixture(tmp_path)
    result = audit_campaign_run(tmp_path, state_path, PROFILE)

    assert result["passed"] is True
    assert result["readable_checkpoints"] == 2
    assert result["canonical_checkpoint"]["history_points"] == 2
    assert result["canonical_checkpoint"]["class_map_size"] == 4
    assert result["oof"]["rows"] == 2
    assert result["oof"]["unique_files"] == 2


def test_audit_campaign_run_rejects_corrupted_receipt(tmp_path):
    state_path, bundle = _fixture(tmp_path)
    with bundle.open("ab") as handle:
        handle.write(b"corruption")

    with pytest.raises(RuntimeError, match="size mismatch"):
        audit_campaign_run(tmp_path, state_path, PROFILE)


def test_audit_campaign_run_rejects_non_contiguous_history(tmp_path):
    state_path, _ = _fixture(tmp_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    best = tmp_path / "checkpoints" / PROFILE / "campp_best.pt"
    checkpoint = torch.load(best, map_location="cpu", weights_only=False)
    checkpoint["training_history"][-1]["epoch"] = 3
    torch.save(checkpoint, best)
    receipt = next(
        item for item in state["completed_runs"][0]["artifacts"]
        if item["path"].endswith("campp_best.pt")
    )
    receipt["size_bytes"] = best.stat().st_size
    receipt["sha256"] = _sha(best)
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(RuntimeError, match="not contiguous"):
        audit_campaign_run(tmp_path, state_path, PROFILE)
