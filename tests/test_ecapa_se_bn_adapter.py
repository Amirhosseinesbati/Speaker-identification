import torch
from torch import nn

from src.encoders import (
    _enable_se_bn_adapter_parameters,
    _set_se_bn_adapter_mode,
)


class SEBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // 2)
        self.fc2 = nn.Linear(channels // 2, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ToyEcapaEmbedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv1d(4, 4, 1)
        self.norm = nn.BatchNorm1d(4)
        self.se_block = SEBlock(4)
        self.tail = nn.Linear(4, 2)


def test_se_bn_adapter_selects_only_se_and_bn_affine_parameters() -> None:
    model = ToyEcapaEmbedding()

    receipt = _enable_se_bn_adapter_parameters(model)

    assert receipt["mode"] == "se_bn"
    assert receipt["se_module_names"] == ("se_block",)
    assert receipt["bn_module_names"] == ("norm",)
    assert receipt["adapter_parameters"] > 0
    assert 0.0 < receipt["adapter_fraction"] < 1.0
    assert all(parameter.requires_grad for parameter in model.se_block.parameters())
    assert model.norm.weight.requires_grad
    assert model.norm.bias.requires_grad
    assert not any(parameter.requires_grad for parameter in model.stem.parameters())
    assert not any(parameter.requires_grad for parameter in model.tail.parameters())


def test_se_bn_adapter_training_mode_updates_only_adapter_modules() -> None:
    model = ToyEcapaEmbedding()
    _enable_se_bn_adapter_parameters(model)

    _set_se_bn_adapter_mode(model, training=True)
    assert model.training is False
    assert model.stem.training is False
    assert model.tail.training is False
    assert model.norm.training is True
    assert model.se_block.training is True

    _set_se_bn_adapter_mode(model, training=False)
    assert all(module.training is False for module in model.modules())
