import pytest
import torch

from scripts.audit_raw_latest_weight_soup_lme20 import (
    SOUP_VARIANT,
    SOUP_WEIGHT,
    uniform_soup_state_dict,
)


def test_uniform_soup_contract_is_fixed() -> None:
    assert SOUP_VARIANT == "uniform_raw_latest_weight_soup"
    assert SOUP_WEIGHT == 0.5


def test_uniform_soup_averages_float_and_preserves_raw_integer() -> None:
    raw = {
        "weight": torch.tensor([1.0, 3.0]),
        "num_batches_tracked": torch.tensor(4, dtype=torch.int64),
    }
    latest = {
        "weight": torch.tensor([3.0, 5.0]),
        "num_batches_tracked": torch.tensor(12, dtype=torch.int64),
    }
    soup, summary = uniform_soup_state_dict(raw, latest)
    assert torch.equal(soup["weight"], torch.tensor([2.0, 4.0]))
    assert soup["num_batches_tracked"].item() == 4
    assert summary == {
        "averaged_floating_tensors": 1,
        "preserved_raw_nonfloating_tensors": 1,
    }


def test_uniform_soup_rejects_mismatched_keys() -> None:
    with pytest.raises(RuntimeError, match="keys differ"):
        uniform_soup_state_dict(
            {"a": torch.tensor([1.0])},
            {"b": torch.tensor([1.0])},
        )
