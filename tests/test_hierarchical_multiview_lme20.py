import numpy as np
import torch
from scipy.special import logsumexp

from scripts.audit_hierarchical_multiview_lme20 import (
    flatten_valid_enrollment_views,
    hierarchical_logmeanexp_scores,
)


def test_flatten_valid_enrollment_views_excludes_repeated_padding_views():
    views = np.arange(3 * 4 * 2, dtype=np.float32).reshape(3, 4, 2)
    counts = np.array([1, 3, 2], dtype=np.int64)
    flat, file_ids = flatten_valid_enrollment_views(views, counts)
    expected = np.concatenate([views[0, :1], views[1, :3], views[2, :2]])
    np.testing.assert_array_equal(flat, expected)
    np.testing.assert_array_equal(file_ids, np.array([0, 1, 1, 1, 2, 2]))


def test_hierarchical_lme_matches_bruteforce_three_levels():
    query = np.array(
        [
            [[1.0, 0.0], [0.0, 1.0], [9.0, 9.0]],
            [[1.0, 0.0], [9.0, 9.0], [9.0, 9.0]],
        ],
        dtype=np.float32,
    )
    query_counts = np.array([2, 1])
    enrollment = np.array(
        [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]],
        dtype=np.float32,
    )
    view_file_ids = np.array([0, 0, 1, 2])
    file_view_counts = np.array([2, 1, 1])
    file_group_ids = np.array([0, 0, 1])
    group_file_counts = np.array([2, 1])
    beta = 5.0

    actual = hierarchical_logmeanexp_scores(
        query_views=query,
        query_counts=query_counts,
        enrollment_views=enrollment,
        enrollment_view_file_ids=view_file_ids,
        enrollment_file_view_counts=file_view_counts,
        enrollment_file_group_ids=file_group_ids,
        group_file_counts=group_file_counts,
        beta=beta,
        device=torch.device("cpu"),
        file_batch_size=2,
    )

    expected = np.empty((2, 2), dtype=np.float64)
    for query_file, query_count in enumerate(query_counts):
        query_group_scores = []
        for query_view in query[query_file, :query_count]:
            per_file = []
            for file_id in range(3):
                similarities = enrollment[view_file_ids == file_id] @ query_view
                per_file.append(
                    (logsumexp(beta * similarities) - np.log(len(similarities)))
                    / beta
                )
            per_file = np.asarray(per_file)
            per_group = []
            for group_id in range(2):
                values = per_file[file_group_ids == group_id]
                per_group.append(
                    (logsumexp(beta * values) - np.log(len(values))) / beta
                )
            query_group_scores.append(per_group)
        query_group_scores = np.asarray(query_group_scores)
        expected[query_file] = (
            logsumexp(beta * query_group_scores, axis=0) - np.log(query_count)
        ) / beta
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
