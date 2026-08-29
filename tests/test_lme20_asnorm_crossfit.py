import numpy as np

from scripts.analyze_lme20_asnorm_crossfit import (
    adaptive_symmetric_normalise,
    cohort_z_statistics,
)


def test_cohort_z_statistics_excludes_same_group_rows() -> None:
    # The very high diagonal-like scores must not enter each group's Z cohort.
    scores = np.array([
        [100.0, 1.0, 2.0],
        [90.0, 3.0, 4.0],
        [5.0, 100.0, 6.0],
        [7.0, 8.0, 100.0],
    ])
    group_ids = np.array([0, 0, 1, 2])
    means, stds = cohort_z_statistics(scores, group_ids, top_n=2)
    np.testing.assert_allclose(means, [6.0, 5.5, 5.0])
    np.testing.assert_allclose(stds, [1.0, 2.5, 1.0])


def test_adaptive_symmetric_normalise_is_finite_and_matches_formula() -> None:
    scores = np.array([[0.1, 0.3, 0.2], [0.6, 0.2, 0.4]])
    z_means = np.array([0.2, 0.1, 0.3])
    z_stds = np.array([0.2, 0.1, 0.4])
    actual, diagnostics = adaptive_symmetric_normalise(
        scores, z_means, z_stds, top_n=2
    )
    top = np.array([[0.3, 0.2], [0.6, 0.4]])
    t_mean = top.mean(axis=1)
    t_std = top.std(axis=1)
    expected = 0.5 * (
        (scores - t_mean[:, None]) / t_std[:, None]
        + (scores - z_means[None, :]) / z_stds[None, :]
    )
    np.testing.assert_allclose(actual, expected)
    assert np.all(np.isfinite(actual))
    assert diagnostics["top_n"] == 2
