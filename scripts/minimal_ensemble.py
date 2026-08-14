"""
Stateless ensemble fusion functions for submission inference.

Operate on numpy probability matrices (M models × N files × C classes) and are
used by ``submission/inference.py`` at eval time. All functions return a fused
(N, C) probability matrix whose rows sum to 1.

This minimal version ships in the submission package only (the repo's
src/ensemble.py keeps the full training/calibration tooling). build_submission.py
copies this file over src/ensemble.py.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy.special import softmax


def weighted_average_fusion(
    probs_list: Sequence[np.ndarray],
    weights: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Weighted average of per-model probability vectors.

    Args:
        probs_list: list of (N, C) probability arrays.
        weights:    per-model weights (len = len(probs_list)); uniform if None.

    Returns:
        fused_probs: (N, C) — rows sum to 1.
    """
    stacked = np.stack(probs_list, axis=0)  # (M, N, C)
    if weights is None:
        weights = np.ones(len(probs_list)) / len(probs_list)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        weights = weights / weights.sum()
    fused = np.tensordot(weights, stacked, axes=(0, 0))  # (N, C)
    row_sums = fused.sum(axis=1, keepdims=True)
    return fused / (row_sums + 1e-12)


def geometric_mean_fusion(
    probs_list: Sequence[np.ndarray],
    eps: float = 1e-9,
) -> np.ndarray:
    """Geometric mean fusion — less affected by a single near-zero model.

    Computes exp(mean(log(p + ε))) then renormalises.
    """
    log_probs = [np.log(np.maximum(p, eps)) for p in probs_list]
    geo = np.exp(np.mean(log_probs, axis=0))  # (N, C)
    row_sums = geo.sum(axis=1, keepdims=True)
    return geo / (row_sums + 1e-12)


def rank_average_fusion(
    probs_list: Sequence[np.ndarray],
    temperature: float = 1.0,
) -> np.ndarray:
    """Rank-based fusion — independent of per-model probability scale.

    1. For each model, compute per-row ranks (higher prob → higher rank).
    2. Average ranks across models.
    3. Convert average ranks back to a probability distribution via softmax.
    """
    all_ranks = []
    for probs in probs_list:
        order = np.argsort(probs, axis=1)
        ranks = np.argsort(order, axis=1).astype(np.float64)
        all_ranks.append(ranks)

    avg_ranks = np.mean(all_ranks, axis=0)  # (N, C)
    return softmax(avg_ranks / max(temperature, 1e-6), axis=1)


def max_prob_fusion(
    probs_list: Sequence[np.ndarray],
) -> np.ndarray:
    """Element-wise maximum over model probabilities.

    Take the highest probability for each class across models, then renormalise.
    """
    stacked = np.stack(probs_list, axis=0)  # (M, N, C)
    fused = stacked.max(axis=0)  # (N, C)
    row_sums = fused.sum(axis=1, keepdims=True)
    return fused / (row_sums + 1e-12)
