import numpy as np

from scripts.analyze_lme20_nap_crossfit import nuisance_projection


def test_nuisance_projection_is_symmetric_and_idempotent() -> None:
    covariance = np.diag([1.0, 2.0, 3.0, 4.0])
    projection, diagnostics = nuisance_projection(covariance, removed_dims=2)
    np.testing.assert_allclose(projection, projection.T, atol=1e-12)
    np.testing.assert_allclose(projection @ projection, projection, atol=1e-12)
    np.testing.assert_allclose(projection, np.diag([1.0, 1.0, 0.0, 0.0]))
    assert diagnostics["removed_within_variance_fraction"] == 0.7


def test_nuisance_projection_annihilates_top_eigenvectors() -> None:
    covariance = np.array([
        [2.0, 0.4, 0.0],
        [0.4, 1.0, 0.0],
        [0.0, 0.0, 0.5],
    ])
    _, eigenvectors = np.linalg.eigh(covariance)
    top = eigenvectors[:, -1]
    projection, _ = nuisance_projection(covariance, removed_dims=1)
    np.testing.assert_allclose(projection @ top, 0.0, atol=1e-12)
