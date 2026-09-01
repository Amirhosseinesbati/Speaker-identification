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


def test_zero_lr_warm_started_head_passes_gradient_without_updating() -> None:
    """P9 can keep its trained heads fixed while adapting SE/BN.

    A zero-LR AdamW group must neither apply the gradient nor weight decay to
    the warm-started heads, while the classifier operation still propagates a
    useful gradient into the adapter parameters.
    """
    torch.manual_seed(7)
    encoder = ToyEcapaEmbedding()
    head = nn.Linear(2, 3)
    _enable_se_bn_adapter_parameters(encoder)
    _set_se_bn_adapter_mode(encoder, training=True)

    adapter_parameters = [
        parameter for parameter in encoder.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": adapter_parameters, "lr": 1e-3},
            {"params": head.parameters(), "lr": 0.0},
        ],
        weight_decay=1e-4,
    )

    head_before = {
        name: parameter.detach().clone()
        for name, parameter in head.named_parameters()
    }
    adapter_before = [parameter.detach().clone() for parameter in adapter_parameters]
    stem_before = encoder.stem.weight.detach().clone()

    features = encoder.norm(encoder.stem(torch.randn(6, 4, 8))).mean(dim=-1)
    gates = torch.sigmoid(
        encoder.se_block.fc2(torch.relu(encoder.se_block.fc1(features)))
    )
    logits = head(encoder.tail(features * gates))
    loss = logits.square().mean()
    loss.backward()

    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for parameter in adapter_parameters
    )
    assert all(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for parameter in head.parameters()
    )

    optimizer.step()

    assert any(
        not torch.equal(before, after.detach())
        for before, after in zip(adapter_before, adapter_parameters)
    )
    assert all(
        torch.equal(head_before[name], parameter.detach())
        for name, parameter in head.named_parameters()
    )
    assert torch.equal(stem_before, encoder.stem.weight.detach())
