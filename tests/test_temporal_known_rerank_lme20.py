import numpy as np

from scripts.audit_temporal_known_rerank_lme20 import (
    known_only_rerank_predictions,
)


def test_known_only_rerank_preserves_binary_boundary_and_reranks_known():
    head = np.full((3, 447), 1e-6, dtype=np.float64)
    head[:, 0] = 0.2
    head[0, 10] = 0.8
    head[1, 1] = 0.8
    head[2, 2] = 0.8
    head /= head.sum(axis=1, keepdims=True)
    scores = np.full((3, 1000), -1.0, dtype=np.float64)
    scores[0, 9] = 1.0
    scores[1, 2] = 1.0
    scores[2, 3] = 1.0
    baseline = np.array([0, 1, 2], dtype=np.int64)

    candidate = known_only_rerank_predictions(
        head=head,
        temporal_scores=scores,
        baseline_predictions=baseline,
    )

    np.testing.assert_array_equal(candidate == 0, baseline == 0)
    np.testing.assert_array_equal(candidate, np.array([0, 3, 4]))
