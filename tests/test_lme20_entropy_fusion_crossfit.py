import numpy as np

from scripts.analyze_lme20_entropy_fusion_crossfit import (
    LOCKED_ALPHA,
    adaptive_head_weights,
    entropy_reliability,
)


def test_entropy_reliability_orders_confident_before_uniform() -> None:
    probabilities = np.array([
        [0.98, 0.01, 0.01],
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
    ])
    reliability = entropy_reliability(probabilities)
    assert reliability[0] > reliability[1]
    np.testing.assert_allclose(reliability[1], 0.0, atol=1e-12)


def test_adaptive_weight_moves_toward_more_reliable_head() -> None:
    head = np.array([
        [0.98, 0.01, 0.01],
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
    ])
    prototype = head[::-1].copy()
    weights, diagnostics = adaptive_head_weights(head, prototype, gamma=2.0)
    assert weights[0] > LOCKED_ALPHA
    assert weights[1] < LOCKED_ALPHA
    assert diagnostics["head_weight_min"] < LOCKED_ALPHA
    assert diagnostics["head_weight_max"] > LOCKED_ALPHA


def test_zero_gamma_recovers_locked_weight() -> None:
    head = np.array([[0.7, 0.2, 0.1]])
    prototype = np.array([[0.2, 0.3, 0.5]])
    weights, _ = adaptive_head_weights(head, prototype, gamma=0.0)
    np.testing.assert_allclose(weights, LOCKED_ALPHA)
