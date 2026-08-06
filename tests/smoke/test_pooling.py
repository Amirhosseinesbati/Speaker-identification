"""Smoke test for pooling module — CPU only, no downloads."""

import torch
import pytest
from src.pooling import (
    StatisticalPooling,
    AttentiveStatisticalPooling,
    create_pooling,
)


class TestStatisticalPooling:
    def test_output_shape(self):
        pool = StatisticalPooling()
        x = torch.randn(4, 50, 256)
        out = pool(x)
        assert out.shape == (4, 512)

    def test_output_shape_with_mask(self):
        pool = StatisticalPooling()
        x = torch.randn(2, 80, 128)
        mask = torch.ones(2, 80, dtype=torch.bool)
        mask[:, -30:] = False
        out = pool(x, mask)
        assert out.shape == (2, 256)
        assert not torch.isnan(out).any()

    def test_output_not_all_zero(self):
        pool = StatisticalPooling()
        x = torch.randn(3, 60, 512)
        out = pool(x)
        assert out.abs().sum() > 0

    def test_output_multiplier(self):
        pool = StatisticalPooling()
        assert pool.output_multiplier == 2


class TestAttentiveStatisticalPooling:
    def test_output_shape(self):
        pool = AttentiveStatisticalPooling(input_dim=256)
        x = torch.randn(4, 50, 256)
        out = pool(x)
        assert out.shape == (4, 512)

    def test_output_shape_with_mask(self):
        pool = AttentiveStatisticalPooling(input_dim=128)
        x = torch.randn(2, 80, 128)
        mask = torch.ones(2, 80, dtype=torch.bool)
        mask[:, -30:] = False
        out = pool(x, mask)
        assert out.shape == (2, 256)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_output_not_all_zero(self):
        pool = AttentiveStatisticalPooling(input_dim=512)
        x = torch.randn(3, 60, 512)
        out = pool(x)
        assert out.abs().sum() > 0

    def test_different_inputs_give_different_outputs(self):
        pool = AttentiveStatisticalPooling(input_dim=64)
        x1 = torch.randn(2, 30, 64)
        x2 = torch.randn(2, 30, 64) + 10.0  # shifted
        out1 = pool(x1)
        out2 = pool(x2)
        assert not torch.allclose(out1, out2)

    def test_output_multiplier(self):
        pool = AttentiveStatisticalPooling(input_dim=128)
        assert pool.output_multiplier == 2

    def test_custom_attention_dim(self):
        pool = AttentiveStatisticalPooling(input_dim=256, attention_dim=64)
        x = torch.randn(2, 40, 256)
        out = pool(x)
        assert out.shape == (2, 512)


class TestCreatePooling:
    def test_statistical(self):
        pool = create_pooling("statistical", 256)
        assert isinstance(pool, StatisticalPooling)

    def test_attentive(self):
        pool = create_pooling("attentive", 256)
        assert isinstance(pool, AttentiveStatisticalPooling)

    def test_invalid_type(self):
        with pytest.raises(ValueError):
            create_pooling("invalid_type", 256)
