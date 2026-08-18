"""
Experiment matrix (Audit §17.2) — encoders × recipes × seeds × folds.

Expands a Cartesian product into *named experiment profiles* (see
``src/experiment_config.py``), so a campaign of "ecapa/campp × frozen/full ×
3 seeds × 3 folds" becomes a set of diffable ``configs/experiments/*.yaml``
files that the queue runner (``src/experiment_queue.py``) executes one by one.

Pure module: only numpy-free dict manipulation + the experiment-config layer,
so it unit-tests fast and imports nothing heavy.
"""

from __future__ import annotations

import copy
import json
from itertools import product
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from src.experiment_config import deep_merge, load_base, save_profile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MATRIX_DIR = PROJECT_ROOT / "data" / "experiments"
MANIFEST_PATH = MATRIX_DIR / "matrix_manifest.json"

ENCODERS = ["ecapa", "campp", "eres2net", "titanet", "wavlm"]

# Freeze flag differs per encoder family (see deploy_app._encoder_save_config).
FREEZE_KEYS = {
    "ecapa": "freeze_encoder",
    "campp": "freeze_encoder",
    "eres2net": "freeze_encoder",
    "titanet": "freeze_encoder",
    "wavlm": "freeze_feature_extractor",
}
PARTIAL_KEYS = {
    "ecapa": "unfreeze_last_n_blocks",
    "campp": "unfreeze_last_n_blocks",
    "eres2net": "unfreeze_last_n_blocks",
}

# Recipe name → how to set the ACTIVE encoder's freeze mode. Full FT is the
# Phase-2 default; frozen/partial are kept for controlled ablations (A1).
#
# The ``*_ft`` recipes additionally enable the aggressive augmentation stack +
# fine-tuning training settings (see AGGRESSIVE_AUGMENTATION / FINE_TUNE_SETTINGS
# below) — the encoder fine-tuning strategy that targets the ~0.97 Macro-F1
# ceiling (frozen VoxCeleb encoders alone plateau at ~0.9496 on val).
RECIPES = {
    "frozen": {"freeze_mode": "frozen"},
    "full": {"freeze_mode": "full"},
    "partial": {"freeze_mode": "partial", "unfreeze_last_n_blocks": 2},
    # ``frozen_ft`` isolates the AUGMENTATION effect: frozen encoder + aggressive
    # augmentation + more crops + regularisation, WITHOUT any fine-tuning. Compare
    # it against ``frozen`` (gentle) and ``full_ft`` (aggressive + fine-tuned) to
    # decompose the fine-tuning gain into "augmentation" vs "trainable encoder".
    "frozen_ft": {"freeze_mode": "frozen", "fine_tune": True},
    "full_ft": {"freeze_mode": "full", "fine_tune": True},
    "partial_ft": {"freeze_mode": "partial", "unfreeze_last_n_blocks": 2,
                   "fine_tune": True},
    # Two-phase fine-tune: heads-only for `freeze_epochs` epochs (encoder frozen),
    # then full encoder fine-tune — the strategy that avoids propagating garbage
    # gradients from a random-initialised head into the encoder at epoch 0.
    "two_phase_ft": {"freeze_mode": "full", "fine_tune": True, "freeze_epochs": 20},
}

# Aggressive domain augmentation used by the ``*_ft`` recipes. Enables encoder
# fine-tuning on ~5 files/speaker without overfitting: speed perturbation
# (near-always-on), MUSAN noise/music, RIR reverb, mp3 codec roundtrip — the
# classic VoxCeleb fine-tuning stack. Merged onto the base augmentation block
# (every other effect keeps its base probability).
AGGRESSIVE_AUGMENTATION = {
    "waveform": {
        "time_stretch": {"p": 0.9, "rate": [0.9, 1.1]},
    },
    "domain": {
        "musan": {"noise_p": 0.8, "music_p": 0.5, "snr_db": [0.0, 15.0]},
        "rirs_reverb": {"p": 0.6},
        "mp3_codec_roundtrip": {"p": 0.5},
    },
}

# Training settings for the ``*_ft`` recipes: low encoder LR, heavier weight
# decay + label smoothing, and more random crops per file (implicit data
# multiplication — the few-shot overfitting guard).
FINE_TUNE_SETTINGS = {
    "training": {
        "encoder_lr": 1.0e-5,
        "weight_decay": 3.0e-4,
        "label_smoothing": 0.1,
        "loss": {"speaker": {"label_smoothing": 0.1}},
    },
    "audio": {"num_train_windows": 8},
}

DEFAULT_SEEDS = [42, 1337, 2026]


# ────────────────────────────────────────────────────────────────
#  Cell construction
# ────────────────────────────────────────────────────────────────

def apply_recipe(encoder: str, recipe: str) -> dict:
    """Return the ``encoder_config[<enc>]`` overrides for a freeze recipe."""
    if recipe not in RECIPES:
        raise KeyError(f"Unknown recipe {recipe!r}. Valid: {sorted(RECIPES)}")
    spec = RECIPES[recipe]
    key = FREEZE_KEYS[encoder]
    mode = spec["freeze_mode"]

    out: dict = {}
    if mode == "frozen":
        out[key] = True
    else:
        out[key] = False
    if encoder in PARTIAL_KEYS:
        out[PARTIAL_KEYS[encoder]] = (
            int(spec.get("unfreeze_last_n_blocks", 0)) if mode == "partial" else 0
        )
    return out


def build_cell_config(
    base: dict,
    encoder: str,
    recipe: str,
    seed: int,
    fold: Optional[int] = None,
    scheme: str = "single",
) -> dict:
    """Deep-copy the base and apply one matrix cell (encoder/recipe/seed/fold)."""
    cfg = copy.deepcopy(base)

    mc = cfg.setdefault("model", {})
    mc["encoder_type"] = encoder
    mc.setdefault("encoder_config", {}).setdefault(encoder, {})
    mc["encoder_config"][encoder].update(apply_recipe(encoder, recipe))

    # Fine-tune recipes also enable the aggressive augmentation stack + the
    # fine-tuning training settings (low encoder LR, heavier regularisation,
    # more crops). deep_merge preserves every base key that isn't overridden.
    if recipe in RECIPES and RECIPES[recipe].get("fine_tune"):
        cfg["augmentation"] = deep_merge(
            cfg.get("augmentation", {}), AGGRESSIVE_AUGMENTATION)
        cfg = deep_merge(cfg, FINE_TUNE_SETTINGS)
        # Two-phase recipes keep the encoder frozen for the first N epochs.
        fe = RECIPES[recipe].get("freeze_epochs")
        if fe:
            cfg.setdefault("training", {})["freeze_epochs"] = int(fe)

    split = cfg.setdefault("data", {}).setdefault("split", {})
    split["scheme"] = scheme
    split["seed"] = int(seed)
    if fold is not None:
        split["fold"] = int(fold)
        if int(fold) >= int(split.get("folds", 3)):
            split["folds"] = int(fold) + 1
    return cfg


def expand_matrix(
    encoders: Sequence[str],
    recipes: Sequence[str],
    seeds: Sequence[int],
    folds: Optional[Sequence[int]] = None,
    scheme: str = "single",
    base: Optional[dict] = None,
) -> List[dict]:
    """Expand a matrix spec into a list of ``{name, encoder, recipe, seed,
    fold, config}`` dicts (one per cell)."""
    base = base if base is not None else load_base()
    fold_values: List[Optional[int]] = list(folds) if folds else [None]
    cells: List[dict] = []
    for enc, recipe, seed, fold in product(encoders, recipes, seeds, fold_values):
        name = f"{enc}-{recipe}-s{seed}"
        if fold is not None:
            name += f"-f{fold}"
        cells.append({
            "name": name,
            "encoder": enc,
            "recipe": recipe,
            "seed": int(seed),
            "fold": int(fold) if fold is not None else None,
            "scheme": scheme,
            "config": build_cell_config(base, enc, recipe, seed, fold, scheme),
        })
    return cells


def write_matrix_profiles(cells: Sequence[dict], base: Optional[dict] = None) -> List[str]:
    """Persist each matrix cell as a named profile; returns the profile names."""
    base = base if base is not None else load_base()
    names = []
    for cell in cells:
        save_profile(cell["name"], cell["config"], base=base)
        names.append(cell["name"])

    MATRIX_DIR.mkdir(parents=True, exist_ok=True)
    manifest = [
        {k: v for k, v in cell.items() if k != "config"}
        for cell in cells
    ]
    MANIFEST_PATH.write_text(
        json.dumps({"n": len(cells), "cells": manifest}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return names


# ────────────────────────────────────────────────────────────────
#  Smoke test
# ────────────────────────────────────────────────────────────────

def _smoke_test() -> None:
    base = load_base()
    cells = expand_matrix(["ecapa", "campp"], ["frozen", "full"], [42],
                          scheme="single", base=base)
    assert len(cells) == 4
    ecapa_full = next(c for c in cells if c["encoder"] == "ecapa" and c["recipe"] == "full")
    assert ecapa_full["config"]["model"]["encoder_config"]["ecapa"]["freeze_encoder"] is False
    assert ecapa_full["config"]["model"]["encoder_config"]["ecapa"]["unfreeze_last_n_blocks"] == 0
    campp_frozen = next(c for c in cells if c["encoder"] == "campp" and c["recipe"] == "frozen")
    assert campp_frozen["config"]["model"]["encoder_config"]["campp"]["freeze_encoder"] is True
    assert campp_frozen["config"]["data"]["split"]["seed"] == 42
    # kfold cells get a fold index and keep the base folds value
    kfold = expand_matrix(["ecapa"], ["full"], [7], folds=[0, 1], scheme="kfold", base=base)
    assert len(kfold) == 2
    assert kfold[0]["config"]["data"]["split"]["scheme"] == "kfold"
    assert kfold[1]["config"]["data"]["split"]["fold"] == 1
    print("  experiment_matrix smoke test passed ✅")


if __name__ == "__main__":
    _smoke_test()
