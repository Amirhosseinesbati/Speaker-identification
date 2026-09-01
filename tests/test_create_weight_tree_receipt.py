from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.create_weight_tree_receipt import create_weight_tree_receipt
from scripts.download_all_weights import _has_hf_model_weights


def _snapshot(tmp_path: Path, weight_name: str = "pytorch_model.bin") -> Path:
    root = tmp_path / "wavlm"
    root.mkdir()
    (root / "config.json").write_text(json.dumps({
        "model_type": "wavlm",
        "hidden_size": 768,
        "num_hidden_layers": 12,
    }), encoding="utf-8")
    (root / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    (root / weight_name).write_bytes(b"weights")
    return root


@pytest.mark.parametrize("weight_name", ["pytorch_model.bin", "model.safetensors"])
def test_receipt_accepts_either_huggingface_weight_format(
    tmp_path: Path, weight_name: str,
) -> None:
    root = _snapshot(tmp_path, weight_name)
    receipt = create_weight_tree_receipt(
        root, "microsoft/wavlm-base-plus", "abc123", 768, 12,
    )

    assert _has_hf_model_weights(root) is True
    assert receipt["runtime_format"] == weight_name
    assert receipt["config"]["weighted_sum_state_count"] == 13
    assert receipt["total_size_bytes"] == sum(
        item["size_bytes"] for item in receipt["files"]
    )
    assert len(receipt["tree_sha256"]) == 64


def test_receipt_rejects_ambiguous_weight_payload(tmp_path: Path) -> None:
    root = _snapshot(tmp_path)
    (root / "model.safetensors").write_bytes(b"other")

    with pytest.raises(RuntimeError, match="ambiguous"):
        create_weight_tree_receipt(
            root, "microsoft/wavlm-base-plus", "abc123", 768, 12,
        )


def test_receipt_rejects_config_mismatch(tmp_path: Path) -> None:
    root = _snapshot(tmp_path)

    with pytest.raises(RuntimeError, match="hidden_size mismatch"):
        create_weight_tree_receipt(
            root, "microsoft/wavlm-base-plus", "abc123", 1024, 12,
        )
