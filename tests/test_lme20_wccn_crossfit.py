import numpy as np

from scripts.analyze_lme20_wccn_crossfit import (
    shrinkage_wccn_transform,
    within_group_covariance,
)


def test_within_group_covariance_uses_only_residual_variation() -> None:
    embeddings = np.array([
        [1.0, 0.0],
        [0.8, 0.2],
        [0.0, 1.0],
        [0.2, 0.8],
        [0.7, 0.7],
    ])
    groups = [np.array([0, 1]), np.array([2, 3]), np.array([4])]
    covariance, degrees = within_group_covariance(embeddings, groups)
    assert degrees == 2
    np.testing.assert_allclose(covariance, covariance.T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(covariance) >= -1e-12)


def test_zero_strength_returns_identity() -> None:
    covariance = np.array([[2.0, 0.2], [0.2, 0.5]])
    transform, diagnostics = shrinkage_wccn_transform(covariance, 0.0)
    np.testing.assert_allclose(transform, np.eye(2))
    assert diagnostics["condition_number"] == 1.0


def test_full_strength_whitens_covariance_up_to_trace_scale() -> None:
    covariance = np.array([[2.0, 0.2], [0.2, 0.5]])
    transform, _ = shrinkage_wccn_transform(covariance, 1.0)
    whitened = transform.T @ covariance @ transform
    expected_scale = np.trace(covariance) / covariance.shape[0]
    np.testing.assert_allclose(
        whitened, expected_scale * np.eye(2), atol=1e-10
    )
