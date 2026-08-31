from __future__ import annotations

import pickle

import numpy as np
import pytest
import torch

from src.checkpoint_io import load_project_checkpoint_safe


class UnsupportedPayload:
    pass


def test_safe_loader_accepts_project_tensor_and_numpy_rng_metadata(tmp_path) -> None:
    path = tmp_path / "checkpoint.pt"
    payload = {
        "model_state_dict": {"weight": torch.arange(4, dtype=torch.float32)},
        "config": {"training": {"seed": 42}},
        "class_map": {"unknown": 0, "speaker": 1},
        "rng_state": {"numpy": np.random.RandomState(42).get_state()},
    }
    torch.save(payload, path)

    loaded = load_project_checkpoint_safe(path)

    assert torch.equal(
        loaded["model_state_dict"]["weight"],
        payload["model_state_dict"]["weight"],
    )
    assert loaded["config"] == payload["config"]
    assert loaded["class_map"] == payload["class_map"]
    np.testing.assert_array_equal(
        loaded["rng_state"]["numpy"][1], payload["rng_state"]["numpy"][1]
    )


def test_safe_loader_rejects_custom_pickle_global(tmp_path) -> None:
    path = tmp_path / "unsupported.pt"
    torch.save({"custom": UnsupportedPayload()}, path)

    with pytest.raises(pickle.UnpicklingError):
        load_project_checkpoint_safe(path)


def test_safe_loader_requires_mapping_top_level(tmp_path) -> None:
    path = tmp_path / "tensor.pt"
    torch.save(torch.ones(2), path)

    with pytest.raises(TypeError, match="top-level mapping"):
        load_project_checkpoint_safe(path)
