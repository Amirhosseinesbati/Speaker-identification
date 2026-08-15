"""
Named experiment profiles (Audit §17.1).

The Streamlit Config tab used to overwrite ``configs/default_config.yaml`` on
every "Save", which silently destroys reproducibility (a second click eats the
previous config). This module introduces named profiles that live in
``configs/experiments/<name>.yaml`` and *inherit* from ``default_config.yaml``:
each profile stores only the keys that differ from the base, so experiments are
small, diffable, and mergeable when the base evolves.

Public API (pure — no torch / zenml / mlflow imports, safe to unit-test):
    load_base()              -> dict          base config (default_config.yaml)
    load_profile(name)       -> dict          base + overrides deep-merged
    save_profile(name, cfg)  -> Path          write only the diff vs base
    list_profiles()          -> list[str]     stem names (no _resolved/ internals)
    is_profile(name)         -> bool
    resolve_profile(name)    -> Path          materialise a full config to disk
    resolve_config_arg(val)  -> Path          --config/--experiment → file path
    deep_merge(base, over)   -> dict          recursive dict merge (lists replace)
"""

from __future__ import annotations

import copy
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"
BASE_CONFIG = CONFIG_DIR / "default_config.yaml"
EXPERIMENTS_DIR = CONFIG_DIR / "experiments"
RESOLVED_DIR = EXPERIMENTS_DIR / "_resolved"


# ────────────────────────────────────────────────────────────────
#  Merge / diff primitives
# ────────────────────────────────────────────────────────────────

def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` onto ``base``.

    Nested dicts are merged key-by-key; scalars and lists are replaced whole
    (a list in an override means "use this exact list", matching how YAML config
    blocks like ``audio.waveform.gaussian_noise.amp`` are replaced).
    """
    if not isinstance(base, dict) or not isinstance(override, dict):
        return copy.deepcopy(override)
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _diff(base: dict, override: dict) -> dict:
    """Return the minimal subtree of ``override`` that differs from ``base``."""
    if not isinstance(override, dict):
        return copy.deepcopy(override)
    diff: dict = {}
    for key, value in override.items():
        if key not in base:
            diff[key] = copy.deepcopy(value)
        elif isinstance(value, dict) and isinstance(base[key], dict):
            sub = _diff(base[key], value)
            if sub:
                diff[key] = sub
        elif base[key] != value:
            diff[key] = copy.deepcopy(value)
    return diff


def _git_revision() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# ────────────────────────────────────────────────────────────────
#  IO
# ────────────────────────────────────────────────────────────────

def load_base() -> dict:
    with open(BASE_CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _profile_path(name: str) -> Path:
    if not name.endswith(".yaml"):
        name = f"{name}.yaml"
    return EXPERIMENTS_DIR / name


def load_profile(name: str, base: Optional[dict] = None) -> dict:
    """Resolve a named profile: base config deep-merged with its overrides."""
    path = _profile_path(name)
    if not path.exists():
        raise FileNotFoundError(f"Experiment profile not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    doc.pop("_meta", None)
    return deep_merge(base if base is not None else load_base(), doc)


def save_profile(name: str, config: dict, base: Optional[dict] = None) -> Path:
    """Persist ``config`` as a named profile storing only the diff vs ``base``."""
    base = base if base is not None else load_base()
    overrides = _diff(base, config)

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    path = _profile_path(name)
    doc = {
        "_meta": {
            "name": name,
            "base": BASE_CONFIG.name,
            "created": datetime.now().isoformat(timespec="seconds"),
            "base_git": _git_revision(),
        }
    }
    doc.update(overrides)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(doc, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return path


def list_profiles() -> list:
    if not EXPERIMENTS_DIR.exists():
        return []
    return sorted(
        p.stem for p in EXPERIMENTS_DIR.glob("*.yaml")
        if not p.name.startswith("_")
    )


def is_profile(name: str) -> bool:
    return _profile_path(name).exists()


def resolve_profile(name: str, base: Optional[dict] = None) -> Path:
    """Materialise a named profile into a full config file and return its path.

    The pipeline steps all consume a YAML *path*, so a resolved profile is
    written under ``configs/experiments/_resolved/`` (gitignored) and that path
    is handed to ``run_pipeline.py``. The original profile stays the source of
    truth for the diff.
    """
    merged = load_profile(name, base=base)
    RESOLVED_DIR.mkdir(parents=True, exist_ok=True)
    out = RESOLVED_DIR / f"{name}.yaml"
    with open(out, "w", encoding="utf-8") as f:
        yaml.dump(merged, f, default_flow_style=False, sort_keys=False,
                  allow_unicode=True)
    return out


def resolve_config_arg(value: str) -> Path:
    """Resolve a ``--config``/``--experiment`` CLI value to a config file path.

    Accepts an existing file path, or a named profile (which is materialised).
    Raises ``FileNotFoundError`` with a helpful message otherwise.
    """
    path = Path(value)
    if path.is_file():
        return path.resolve()
    if is_profile(value):
        return resolve_profile(value)
    raise FileNotFoundError(
        f"Config file or experiment profile not found: {value!r}\n"
        f"Known profiles: {list_profiles()}"
    )


# ────────────────────────────────────────────────────────────────
#  Smoke test
# ────────────────────────────────────────────────────────────────

def _smoke_test() -> None:
    base = {"a": 1, "nested": {"x": 1, "y": 2}, "lst": [1, 2]}
    over = {"nested": {"y": 3, "z": 4}, "lst": [9]}
    merged = deep_merge(base, over)
    assert merged["a"] == 1
    assert merged["nested"] == {"x": 1, "y": 3, "z": 4}
    assert merged["lst"] == [9]
    assert _diff(base, merged) == {"nested": {"y": 3, "z": 4}, "lst": [9]}
    print("  experiment_config smoke test passed ✅")


if __name__ == "__main__":
    _smoke_test()
