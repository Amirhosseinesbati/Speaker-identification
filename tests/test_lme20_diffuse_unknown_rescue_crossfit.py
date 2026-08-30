from scripts.analyze_lme20_diffuse_unknown_rescue_crossfit import (
    LOCKED_TAU,
    PARAMETER_GRID,
    diffuse_unknown_rescue,
)


def row(**overrides):
    base = {
        "prediction": 0,
        "head_prediction": 7,
        "winner_known_id": 7,
        "raw_max_score": LOCKED_TAU + 0.1,
        "unknown_effective_clusters": 12.0,
        "head_margin": 0.25,
    }
    base.update(overrides)
    return base


def test_rescue_requires_head_and_known_prototype_agreement() -> None:
    rows = [row(), row(winner_known_id=8), row(head_prediction=0)]
    predictions, changed = diffuse_unknown_rescue(
        rows,
        minimum_effective_clusters=8.0,
        minimum_head_margin=0.2,
    )
    assert predictions.tolist() == [7, 0, 0]
    assert changed.tolist() == [True, False, False]


def test_rescue_preserves_locked_raw_distance_gate() -> None:
    rows = [
        row(raw_max_score=LOCKED_TAU - 1e-6),
        row(raw_max_score=LOCKED_TAU),
    ]
    predictions, changed = diffuse_unknown_rescue(
        rows,
        minimum_effective_clusters=8.0,
        minimum_head_margin=0.2,
    )
    assert predictions.tolist() == [0, 7]
    assert changed.tolist() == [False, True]


def test_rescue_requires_diffuse_unknown_and_head_margin() -> None:
    rows = [
        row(unknown_effective_clusters=7.99),
        row(head_margin=0.199),
        row(unknown_effective_clusters=8.0, head_margin=0.2),
    ]
    predictions, changed = diffuse_unknown_rescue(
        rows,
        minimum_effective_clusters=8.0,
        minimum_head_margin=0.2,
    )
    assert predictions.tolist() == [0, 0, 7]
    assert changed.tolist() == [False, False, True]


def test_parameter_grid_is_small_and_preregistered() -> None:
    assert len(PARAMETER_GRID) == 9
    assert len(set(PARAMETER_GRID)) == len(PARAMETER_GRID)
