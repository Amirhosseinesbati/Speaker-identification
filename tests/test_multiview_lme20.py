import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from scipy.special import logsumexp

from scripts.audit_multiview_lme20 import (
    extract_view_embeddings,
    multiview_logmeanexp_scores,
    unique_eval_window_count,
)


def test_unique_eval_window_count_matches_repeat_policy():
    assert unique_eval_window_count(100, 100, 0.5, 8) == 1
    assert unique_eval_window_count(101, 100, 0.5, 8) == 2
    assert unique_eval_window_count(250, 100, 0.5, 8) == 4
    assert unique_eval_window_count(1000, 100, 0.5, 8) == 8


def test_multiview_logmeanexp_matches_bruteforce_pairs():
    query = np.array(
        [
            [[1.0, 0.0], [0.0, 1.0], [9.0, 9.0]],
            [[1.0, 0.0], [9.0, 9.0], [9.0, 9.0]],
        ],
        dtype=np.float32,
    )
    query_counts = np.array([2, 1])
    enrollment = np.array(
        [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]], dtype=np.float32
    )
    group_ids = np.array([0, 0, 1])
    group_counts = np.array([2, 1])
    beta = 5.0

    actual = multiview_logmeanexp_scores(
        query_views=query,
        query_counts=query_counts,
        enrollment_views=enrollment,
        enrollment_group_ids=group_ids,
        group_view_counts=group_counts,
        beta=beta,
        device=torch.device("cpu"),
        file_batch_size=2,
    )

    expected = np.empty((2, 2), dtype=np.float64)
    for file_index, query_count in enumerate(query_counts):
        valid_queries = query[file_index, :query_count]
        for group in range(2):
            valid_enrollment = enrollment[group_ids == group]
            pair_scores = valid_queries @ valid_enrollment.T
            expected[file_index, group] = (
                logsumexp(beta * pair_scores)
                - np.log(pair_scores.size)
            ) / beta
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


class _BatchSensitiveToyModel(torch.nn.Module):
    """Expose batching-order mistakes that ordinary pointwise layers hide."""

    def _embed_single(self, waveforms):
        base = waveforms[:, 0, :2]
        batch_context = waveforms[:, 0].mean(dim=0)[:2]
        return base + batch_context

    def embed(self, waveforms):
        raw = torch.stack(
            [self._embed_single(waveforms[:, view]) for view in range(waveforms.shape[1])],
            dim=1,
        )
        return F.normalize(raw.mean(dim=1), p=2, dim=1)


def test_extract_view_embeddings_matches_both_locked_batching_paths():
    windows = torch.tensor(
        [
            [[[1.0, 2.0, 0.0]], [[3.0, 4.0, 0.0]]],
            [[[5.0, 6.0, 0.0]], [[7.0, 8.0, 0.0]]],
        ]
    )
    labels = torch.zeros(2, dtype=torch.long)
    model = _BatchSensitiveToyModel().eval()
    views, aggregate = extract_view_embeddings(
        model=model,
        dataset=TensorDataset(windows, labels),
        device=torch.device("cpu"),
        batch_size=2,
        num_workers=0,
        description="test",
        batching="window_major",
    )

    with torch.inference_mode():
        expected_raw = torch.stack(
            [model._embed_single(windows[:, view]) for view in range(2)], dim=1
        )
        expected_views = F.normalize(expected_raw, p=2, dim=2).numpy()
        expected_aggregate = model.embed(windows).numpy()
    np.testing.assert_allclose(views, expected_views, rtol=0, atol=0)
    np.testing.assert_allclose(aggregate, expected_aggregate, rtol=0, atol=0)

    file_views, file_aggregate = extract_view_embeddings(
        model=model,
        dataset=TensorDataset(windows, labels),
        device=torch.device("cpu"),
        batch_size=2,
        num_workers=0,
        description="test-file-major",
        batching="file_major",
    )
    with torch.inference_mode():
        expected_file_raw = torch.stack(
            [model._embed_single(file_windows) for file_windows in windows], dim=0
        )
        expected_file_views = F.normalize(expected_file_raw, p=2, dim=2).numpy()
        expected_file_aggregate = F.normalize(
            expected_file_raw.mean(dim=1), p=2, dim=1
        ).numpy()
    np.testing.assert_allclose(file_views, expected_file_views, rtol=0, atol=0)
    np.testing.assert_allclose(
        file_aggregate, expected_file_aggregate, rtol=0, atol=0
    )
    assert not np.allclose(file_aggregate, aggregate)
