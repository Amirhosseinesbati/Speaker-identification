import numpy as np

from scripts.analyze_lme20_lda_crossfit import shrinkage_lda_transform


def test_shrinkage_lda_separates_simple_groups() -> None:
    embeddings = np.array([
        [1.0, 0.1, 0.0],
        [0.9, -0.1, 0.0],
        [-1.0, 0.1, 0.0],
        [-0.9, -0.1, 0.0],
        [0.0, 1.0, 0.1],
        [0.0, 0.9, -0.1],
    ])
    groups = [np.array([0, 1]), np.array([2, 3]), np.array([4, 5])]
    mean, transform, diagnostics = shrinkage_lda_transform(
        embeddings, groups, projection_dims=2, within_shrinkage=0.1
    )
    assert mean.shape == (3,)
    assert transform.shape == (3, 2)
    assert np.all(np.isfinite(transform))
    assert diagnostics["known_groups"] == 3


def test_shrinkage_lda_rejects_overlapping_groups() -> None:
    embeddings = np.eye(4)
    groups = [np.array([0, 1]), np.array([1, 2]), np.array([3])]
    try:
        shrinkage_lda_transform(
            embeddings, groups, projection_dims=2, within_shrinkage=0.1
        )
    except RuntimeError as error:
        assert "overlap" in str(error)
    else:
        raise AssertionError("Expected overlapping groups to be rejected")
