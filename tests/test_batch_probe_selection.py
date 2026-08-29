import pytest
import torch

from scripts.probe_training_batch import _consistency_weight, _training_view
from src.batch_probe import select_recommended_batch


def test_selects_fastest_batch_with_required_headroom():
    rows = [
        {"batch_size": 16, "status": "ok", "peak_vram_gib": 10.0,
         "files_per_second": 10.0},
        {"batch_size": 32, "status": "ok", "peak_vram_gib": 18.0,
         "files_per_second": 17.0},
        {"batch_size": 48, "status": "ok", "peak_vram_gib": 22.0,
         "files_per_second": 18.0},
        {"batch_size": 64, "status": "oom"},
    ]
    assert select_recommended_batch(rows, 24.0, 0.10) == 32


def test_returns_none_when_every_candidate_is_oom_or_over_limit():
    rows = [
        {"batch_size": 16, "status": "oom"},
        {"batch_size": 24, "status": "ok", "peak_vram_gib": 23.0,
         "files_per_second": 9.0},
    ]
    assert select_recommended_batch(rows, 24.0, 0.10) is None


def test_probe_extracts_supervised_view_from_paired_batch():
    augmented = torch.zeros(4, 8, 1, 160)
    clean = torch.ones_like(augmented)
    batch = ({"augmented": augmented, "clean": clean}, torch.arange(4))

    assert _training_view(batch) is augmented

    with pytest.raises(ValueError, match="exactly"):
        _training_view(({"augmented": augmented}, torch.arange(4)))


def test_probe_passes_only_enabled_consistency_weight():
    config = {
        "training": {
            "loss": {
                "consistency": {"enabled": True, "weight": 0.1},
            }
        }
    }
    assert _consistency_weight(config) == pytest.approx(0.1)
    config["training"]["loss"]["consistency"]["enabled"] = False
    assert _consistency_weight(config) == 0.0
