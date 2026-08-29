import numpy as np
import torch
from scipy.special import logsumexp

from scripts.audit_multiview_lme20 import (
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
