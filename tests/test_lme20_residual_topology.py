import numpy as np

from scripts.analyze_lme20_residual_topology import (
    topology_name,
    unknown_mass_features,
)


def test_unknown_mass_features_measure_concentration() -> None:
    probabilities = np.array([
        [0.7, 0.1, 0.1, 0.1],
        [0.25, 0.25, 0.25, 0.25],
    ])
    features = unknown_mass_features(probabilities)
    np.testing.assert_allclose(features["unknown_mass"], 1.0)
    assert features["unknown_top1_fraction"][0] > features[
        "unknown_top1_fraction"
    ][1]
    assert features["unknown_effective_clusters"][0] < features[
        "unknown_effective_clusters"
    ][1]


def test_topology_name_covers_open_set_errors() -> None:
    assert topology_name(4, 4) == "correct"
    assert topology_name(4, 0) == "known_to_unknown"
    assert topology_name(4, 5) == "known_to_wrong_known"
    assert topology_name(0, 5) == "unknown_to_known"
