"""
Tests for the experiment matrix (``src/experiment_matrix.py``).

Pure expansion logic only (no GPU, no torch) — verifies that encoders × recipes
× seeds × folds expand correctly and that per-encoder freeze keys are applied.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import experiment_matrix as em


@pytest.fixture
def base():
    return {
        "model": {
            "encoder_type": "ecapa",
            "encoder_config": {
                "ecapa": {"freeze_encoder": True, "unfreeze_last_n_blocks": 0,
                          "source": "speechbrain/spkrec-ecapa-voxceleb"},
                "campp": {"freeze_encoder": True},
                "wavlm": {"freeze_feature_extractor": True},
            },
        },
        "data": {"split": {"scheme": "single", "folds": 3, "fold": 0, "seed": 42}},
    }


def test_apply_recipe_freeze_keys(base):
    # ECAPA: full FT clears the partial flag; partial sets it.
    assert em.apply_recipe("ecapa", "frozen") == {
        "freeze_encoder": True, "unfreeze_last_n_blocks": 0}
    assert em.apply_recipe("ecapa", "full") == {
        "freeze_encoder": False, "unfreeze_last_n_blocks": 0}
    assert em.apply_recipe("ecapa", "partial") == {
        "freeze_encoder": False, "unfreeze_last_n_blocks": 2}
    # WavLM uses a different flag and has no partial support.
    assert em.apply_recipe("wavlm", "full") == {"freeze_feature_extractor": False}


def test_expand_matrix_cartesian_product(base):
    cells = em.expand_matrix(
        ["ecapa", "campp"], ["frozen", "full"], [42], scheme="single", base=base,
    )
    assert len(cells) == 4
    names = {c["name"] for c in cells}
    assert names == {
        "ecapa-frozen-s42", "ecapa-full-s42",
        "campp-frozen-s42", "campp-full-s42",
    }


def test_expand_matrix_kfold_fold_index(base):
    cells = em.expand_matrix(
        ["ecapa"], ["full"], [7], folds=[0, 1], scheme="kfold", base=base,
    )
    assert len(cells) == 2
    split0 = cells[0]["config"]["data"]["split"]
    split1 = cells[1]["config"]["data"]["split"]
    assert split0["scheme"] == "kfold" and split0["seed"] == 7 and split0["fold"] == 0
    assert split1["fold"] == 1


def test_build_cell_config_does_not_mutate_base(base):
    import copy
    snapshot = copy.deepcopy(base)
    em.build_cell_config(base, "campp", "full", 42)
    assert base == snapshot


def test_full_ft_recipe_applies_aggressive_augmentation_and_ft_settings(base):
    cfg = em.build_cell_config(base, "campp", "full_ft", 42, scheme="single")
    # aggressive augmentation stack
    assert cfg["augmentation"]["waveform"]["time_stretch"]["p"] == 0.9
    assert cfg["augmentation"]["domain"]["musan"]["noise_p"] == 0.8
    assert cfg["augmentation"]["domain"]["musan"]["music_p"] == 0.5
    assert cfg["augmentation"]["domain"]["rirs_reverb"]["p"] == 0.6
    assert cfg["augmentation"]["domain"]["mp3_codec_roundtrip"]["p"] == 0.5
    # fine-tune training settings
    assert cfg["training"]["encoder_lr"] == 1e-5
    assert cfg["training"]["weight_decay"] == 3e-4
    assert cfg["training"]["label_smoothing"] == 0.1
    assert cfg["audio"]["num_train_windows"] == 8


def test_frozen_ft_keeps_encoder_frozen_but_applies_augmentation(base):
    cfg = em.build_cell_config(base, "campp", "frozen_ft", 42)
    assert cfg["model"]["encoder_config"]["campp"]["freeze_encoder"] is True
    assert cfg["augmentation"]["waveform"]["time_stretch"]["p"] == 0.9


def test_partial_ft_sets_unfreeze_blocks_for_campp_and_eres2net(base):
    campp = em.build_cell_config(base, "campp", "partial_ft", 42)
    assert campp["model"]["encoder_config"]["campp"]["freeze_encoder"] is False
    assert campp["model"]["encoder_config"]["campp"]["unfreeze_last_n_blocks"] == 2
    eres2net = em.build_cell_config(base, "eres2net", "partial_ft", 42)
    assert eres2net["model"]["encoder_config"]["eres2net"]["unfreeze_last_n_blocks"] == 2


def test_two_phase_ft_sets_freeze_epochs(base):
    cfg = em.build_cell_config(base, "campp", "two_phase_ft", 42)
    assert cfg["training"]["freeze_epochs"] == 20
    # single-phase recipes leave freeze_epochs unset
    assert em.build_cell_config(base, "campp", "full_ft", 42)["training"].get("freeze_epochs", 0) == 0
