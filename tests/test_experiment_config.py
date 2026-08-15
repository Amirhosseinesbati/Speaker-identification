"""
Tests for named experiment profiles (`src/experiment_config.py`).

Covers the pure merge/diff primitives plus a save→load round-trip against a
temporary base config (no torch / zenml / mlflow imports, so these run fast and
without a GPU).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import experiment_config as ec


def test_deep_merge_nested_dicts_and_list_replace():
    base = {"a": 1, "nested": {"x": 1, "y": 2}, "lst": [1, 2]}
    over = {"nested": {"y": 3, "z": 4}, "lst": [9]}
    merged = ec.deep_merge(base, over)
    assert merged["a"] == 1
    assert merged["nested"] == {"x": 1, "y": 3, "z": 4}
    assert merged["lst"] == [9]
    # base must not be mutated
    assert base["nested"] == {"x": 1, "y": 2}


def test_diff_captures_only_changed_keys():
    base = {"a": 1, "nested": {"x": 1, "y": 2}}
    over = {"a": 1, "nested": {"x": 1, "y": 3, "z": 4}, "new": 5}
    assert ec._diff(base, over) == {"nested": {"y": 3, "z": 4}, "new": 5}


def test_save_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(ec, "BASE_CONFIG", tmp_path / "base.yaml")
    monkeypatch.setattr(ec, "EXPERIMENTS_DIR", tmp_path / "experiments")

    base = {"model": {"encoder_type": "ecapa", "n": 1}, "training": {"lr": 1e-3}}
    ec.BASE_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    import yaml
    ec.BASE_CONFIG.write_text(yaml.safe_dump(base), encoding="utf-8")

    overridden = {
        "model": {"encoder_type": "campp", "n": 1},
        "training": {"lr": 3e-4},
    }
    ec.save_profile("myexp", overridden)
    assert ec.list_profiles() == ["myexp"]

    resolved = ec.load_profile("myexp")
    assert resolved["model"]["encoder_type"] == "campp"
    assert resolved["model"]["n"] == 1
    assert resolved["training"]["lr"] == 3e-4

    # The profile file stores only the diff, not the unchanged keys.
    raw = yaml.safe_load(ec._profile_path("myexp").read_text(encoding="utf-8"))
    assert "_meta" in raw
    assert raw["model"] == {"encoder_type": "campp"}
    assert raw["training"] == {"lr": 3e-4}


def test_resolve_config_arg_prefers_existing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(ec, "EXPERIMENTS_DIR", tmp_path / "experiments")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("a: 1", encoding="utf-8")
    assert ec.resolve_config_arg(str(cfg)) == cfg.resolve()


def test_resolve_config_arg_unknown_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(ec, "EXPERIMENTS_DIR", tmp_path / "experiments")
    with pytest.raises(FileNotFoundError):
        ec.resolve_config_arg("no_such_profile_xyz")
