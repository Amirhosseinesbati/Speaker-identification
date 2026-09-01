from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

import src.encoders as encoders
from src.training_utils import encoder_will_train


class _DummyTransformerLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = nn.Linear(2, 2)
        self.layer_norm = nn.LayerNorm(2)
        self.final_layer_norm = nn.LayerNorm(2)


class _DummyWavLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature_extractor = nn.Linear(1, 1)
        self.encoder_block = nn.Linear(2, 2)
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList([
            _DummyTransformerLayer(),
            _DummyTransformerLayer(),
        ])
        self.encoder.layer_norm = nn.LayerNorm(2)
        self.encoder.layerdrop = 0.1
        self.config = SimpleNamespace(
            num_hidden_layers=2,
            hidden_size=2,
            layer_norm_eps=1.0e-5,
            layerdrop=0.1,
        )

    def forward(self, input_values, output_hidden_states=False):
        base = torch.stack((input_values, input_values + 10.0), dim=-1)
        hidden = (base, base + 1.0, base + 2.0)
        return SimpleNamespace(
            last_hidden_state=hidden[-1],
            hidden_states=hidden if output_hidden_states else None,
        )


def _build(monkeypatch, **kwargs):
    monkeypatch.setattr(
        encoders.WavLMModel,
        "from_pretrained",
        lambda *args, **config: _DummyWavLM(),
    )
    return encoders.WavLMEncoder(
        local_path=".",
        allow_hub_download=False,
        **kwargs,
    )


def test_weighted_sum_and_full_freeze_are_independent(monkeypatch) -> None:
    model = _build(
        monkeypatch,
        freeze_encoder=True,
        freeze_feature_extractor=True,
        layer_aggregation="weighted_sum",
    )
    output, lengths = model(torch.tensor([[[1.0, 2.0]]]))

    torch.testing.assert_close(
        output,
        torch.tensor([[[2.0, 12.0], [3.0, 13.0]]]),
    )
    assert lengths is None
    assert model.layer_weights.requires_grad is True
    assert all(not parameter.requires_grad for parameter in model.wavlm.parameters())


def test_frozen_weighted_sum_keeps_only_backbone_in_eval(monkeypatch) -> None:
    model = _build(
        monkeypatch,
        freeze_encoder=True,
        layer_aggregation="weighted_sum",
    )

    model.train()

    assert model.training is True
    assert model.wavlm.training is False
    assert model.layer_weights.requires_grad is True


def test_frozen_backbone_eval_can_be_disabled_explicitly(monkeypatch) -> None:
    model = _build(
        monkeypatch,
        freeze_encoder=True,
        frozen_backbone_eval=False,
        layer_aggregation="weighted_sum",
    )

    model.train()

    assert model.wavlm.training is True


def test_unfrozen_wavlm_follows_training_mode(monkeypatch) -> None:
    model = _build(
        monkeypatch,
        freeze_encoder=False,
        freeze_feature_extractor=True,
        layer_aggregation="last_hidden",
    )

    model.train()

    assert model.wavlm.training is True


def test_legacy_feature_extractor_freeze_keeps_transformer_trainable(monkeypatch) -> None:
    model = _build(
        monkeypatch,
        freeze_encoder=False,
        freeze_feature_extractor=True,
        layer_aggregation="last_hidden",
    )

    assert all(
        not parameter.requires_grad
        for parameter in model.wavlm.feature_extractor.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in model.wavlm.encoder_block.parameters()
    )


def test_layer_adapter_matches_reference_layer_selection_and_softmax(
    monkeypatch,
) -> None:
    model = _build(
        monkeypatch,
        freeze_encoder=True,
        layer_aggregation="layer_adapter",
        layer_adapter_dim=2,
        layer_adapter_layer_norm=False,
    )
    with torch.no_grad():
        for adapter in model.layer_adapters:
            adapter.projection.weight.copy_(torch.eye(2))
            adapter.projection.bias.zero_()

    output, lengths = model(torch.tensor([[[1.0, 2.0]]]))

    # Only transformer outputs base+1 and base+2 are averaged.  Including the
    # initial convolutional state would produce base+1 instead of base+1.5.
    torch.testing.assert_close(
        output,
        torch.tensor([[[2.5, 12.5], [3.5, 13.5]]]),
    )
    assert lengths is None
    assert model.output_dim == 2
    assert len(model.layer_adapters) == 2
    assert len(model.layer_weights) == 2


def test_layer_adapter_freeze_keeps_only_reference_parameters_trainable(
    monkeypatch,
) -> None:
    model = _build(
        monkeypatch,
        freeze_encoder=True,
        layer_aggregation="layer_adapter",
        layer_adapter_dim=2,
        layer_adapter_tune_backbone_layer_norms=True,
    )

    backbone_trainable = {
        name
        for name, parameter in model.wavlm.named_parameters()
        if parameter.requires_grad
    }
    assert backbone_trainable
    assert all(
        name.startswith("encoder.layers.") and "layer_norm" in name
        for name in backbone_trainable
    )
    assert all(
        parameter.requires_grad
        for parameter in model.layer_adapters.parameters()
    )
    assert model.layer_weights.requires_grad is True
    assert all(
        not parameter.requires_grad
        for parameter in model.wavlm.encoder_block.parameters()
    )
    assert all(
        not parameter.requires_grad
        for parameter in model.wavlm.encoder.layer_norm.parameters()
    )
    assert model.wavlm.config.layerdrop == 0.0
    assert model.wavlm.encoder.layerdrop == 0.0


def test_layer_adapter_training_mode_depends_on_backbone_layer_norms(
    monkeypatch,
) -> None:
    reference_model = _build(
        monkeypatch,
        freeze_encoder=True,
        layer_aggregation="layer_adapter",
        layer_adapter_dim=2,
        layer_adapter_tune_backbone_layer_norms=True,
    )
    fully_frozen_model = _build(
        monkeypatch,
        freeze_encoder=True,
        layer_aggregation="layer_adapter",
        layer_adapter_dim=2,
        layer_adapter_tune_backbone_layer_norms=False,
    )

    reference_model.train()
    fully_frozen_model.train()

    assert reference_model.wavlm.training is True
    assert fully_frozen_model.wavlm.training is False
    assert all(adapter.training for adapter in fully_frozen_model.layer_adapters)


def test_layer_adapter_and_softmax_weights_receive_gradients(monkeypatch) -> None:
    model = _build(
        monkeypatch,
        freeze_encoder=True,
        layer_aggregation="layer_adapter",
        layer_adapter_dim=2,
        layer_adapter_layer_norm=False,
    )
    with torch.no_grad():
        for adapter in model.layer_adapters:
            adapter.projection.weight.copy_(torch.eye(2))
            adapter.projection.bias.zero_()

    output, _ = model(torch.tensor([[[1.0, 2.0]]]))
    output.square().mean().backward()

    assert model.layer_weights.grad is not None
    assert torch.count_nonzero(model.layer_weights.grad).item() > 0
    assert all(
        adapter.projection.weight.grad is not None
        and torch.count_nonzero(adapter.projection.weight.grad).item() > 0
        for adapter in model.layer_adapters
    )
    assert all(
        parameter.grad is None
        for parameter in model.wavlm.feature_extractor.parameters()
    )


def test_reference_l_adapter_parameter_count() -> None:
    adapters = nn.ModuleList([
        encoders.WavLMLayerAdapter(768, 512)
        for _ in range(12)
    ])

    assert sum(
        parameter.numel() for parameter in adapters.parameters()
    ) == 4_737_024


def test_create_encoder_propagates_layer_adapter_config(monkeypatch) -> None:
    monkeypatch.setattr(
        encoders.WavLMModel,
        "from_pretrained",
        lambda *args, **config: _DummyWavLM(),
    )
    config = {
        "model": {
            "encoder_type": "wavlm",
            "encoder_config": {
                "wavlm": {
                    "local_path": ".",
                    "freeze_encoder": True,
                    "frozen_backbone_eval": True,
                    "layer_aggregation": "layer_adapter",
                    "layer_adapter_dim": 2,
                    "layer_adapter_activation": "gelu",
                    "layer_adapter_layer_norm": False,
                    "layer_adapter_init_std": 2.0e-3,
                    "layer_adapter_tune_backbone_layer_norms": False,
                }
            },
        }
    }

    model = encoders.create_encoder(config)

    assert isinstance(model, encoders.WavLMEncoder)
    assert model.layer_aggregation == "layer_adapter"
    assert model.frozen_backbone_eval is True
    assert model.output_dim == 2
    assert isinstance(model.layer_adapters[0].activation, nn.GELU)
    assert isinstance(model.layer_adapters[0].layer_norm, nn.Identity)
    assert all(not parameter.requires_grad for parameter in model.wavlm.parameters())


def test_encoder_will_train_uses_full_wavlm_freeze_flag() -> None:
    frozen = {
        "model": {
            "encoder_type": "wavlm",
            "encoder_config": {
                "wavlm": {
                    "freeze_encoder": True,
                    "freeze_feature_extractor": True,
                }
            },
        }
    }
    legacy_full_ft = {
        "model": {
            "encoder_type": "wavlm",
            "encoder_config": {
                "wavlm": {"freeze_feature_extractor": True}
            },
        }
    }
    frozen_weighted_sum = {
        "model": {
            "encoder_type": "wavlm",
            "encoder_config": {
                "wavlm": {
                    "freeze_encoder": True,
                    "layer_aggregation": "weighted_sum",
                }
            },
        }
    }
    frozen_layer_adapter = {
        "model": {
            "encoder_type": "wavlm",
            "encoder_config": {
                "wavlm": {
                    "freeze_encoder": True,
                    "layer_aggregation": "layer_adapter",
                }
            },
        }
    }

    assert encoder_will_train(frozen) is False
    assert encoder_will_train(legacy_full_ft) is True
    assert encoder_will_train(frozen_weighted_sum) is True
    assert encoder_will_train(frozen_layer_adapter) is True
