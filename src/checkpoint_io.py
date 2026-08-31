"""Constrained checkpoint loading for project-generated training bundles.

PyTorch checkpoints in this repository include ordinary tensors plus NumPy RNG
state, configuration dictionaries, and class maps.  Loading them with generic
pickle (``weights_only=False``) gives those metadata fields unnecessary code
execution authority.  This module keeps PyTorch's restricted weights unpickler
enabled and allowlists only the NumPy ndarray reconstruction primitives needed
by the RNG array stored by our own trainer.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:  # NumPy 2.x public module layout
    from numpy._core.multiarray import _reconstruct as _numpy_reconstruct
except ImportError:  # pragma: no cover - compatibility with NumPy 1.x
    from numpy.core.multiarray import _reconstruct as _numpy_reconstruct


_DTYPES = (
    np.bool_,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.float16,
    np.float32,
    np.float64,
    np.complex64,
    np.complex128,
)
_NUMPY_SAFE_GLOBALS = [
    _numpy_reconstruct,
    np.ndarray,
    np.dtype,
    *sorted({type(np.dtype(dtype)) for dtype in _DTYPES}, key=repr),
]


def load_project_checkpoint_safe(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> Mapping[str, Any]:
    """Load a tensor/metadata checkpoint without enabling generic pickle.

    Custom application classes and arbitrary reducers remain forbidden.  The
    returned top-level object must be a mapping, matching every checkpoint
    produced by this project's training pipeline.
    """

    with torch.serialization.safe_globals(_NUMPY_SAFE_GLOBALS):
        payload = torch.load(
            Path(path), map_location=map_location, weights_only=True
        )
    if not isinstance(payload, Mapping):
        raise TypeError(
            "Project checkpoint must contain a top-level mapping, got "
            f"{type(payload).__name__}"
        )
    return payload
