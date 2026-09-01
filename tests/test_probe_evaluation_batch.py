from __future__ import annotations

import torch
from torch import nn

from scripts.probe_evaluation_batch import (
    _model_invariants,
    _validate_model_invariants,
)


class _Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.wavlm = nn.Linear(3, 4)
        self.wavlm.config = type("Config", (), {"num_hidden_layers": 12})()
        for parameter in self.wavlm.parameters():
            parameter.requires_grad = False
        self.layer_weights = nn.Parameter(torch.zeros(13))
        self.layer_adapters = nn.ModuleList()
        self.layer_aggregation = "weighted_sum"


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _Encoder()


def test_model_invariants_separate_frozen_backbone_from_layer_weights() -> None:
    invariants = _model_invariants(_Model())

    assert invariants["wavlm_parameters"] == 16
    assert invariants["wavlm_trainable_parameters"] == 0
    assert invariants["layer_weight_count"] == 13
    assert invariants["layer_weight_trainable"] is True
    assert invariants["wavlm_other_trainable_parameters"] == 0
    assert invariants["layer_adapter_parameters"] == 0
    _validate_model_invariants(_Model(), invariants)


class _TransformerLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = nn.Linear(2, 2)
        self.layer_norm = nn.LayerNorm(2)
        self.final_layer_norm = nn.LayerNorm(2)


class _AdapterEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.wavlm = nn.Module()
        self.wavlm.encoder = nn.Module()
        self.wavlm.encoder.layers = nn.ModuleList([
            _TransformerLayer(),
            _TransformerLayer(),
        ])
        self.wavlm.feature_projection = nn.Linear(2, 2)
        self.wavlm.config = type("Config", (), {"num_hidden_layers": 2})()
        for parameter in self.wavlm.parameters():
            parameter.requires_grad = False
        for name, parameter in self.wavlm.named_parameters():
            if name.startswith("encoder.layers.") and "layer_norm" in name:
                parameter.requires_grad = True
        self.layer_weights = nn.Parameter(torch.zeros(2))
        self.layer_adapters = nn.ModuleList([
            nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.LayerNorm(3)),
            nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.LayerNorm(3)),
        ])
        self.layer_aggregation = "layer_adapter"
        self.layer_adapter_tune_backbone_layer_norms = True


class _AdapterModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _AdapterEncoder()


def test_model_invariants_cover_l_adapter_and_only_transformer_norms() -> None:
    model = _AdapterModel()
    invariants = _model_invariants(model)

    assert invariants["layer_aggregation"] == "layer_adapter"
    assert invariants["layer_weight_count"] == 2
    assert invariants["layer_adapter_count"] == 2
    assert invariants["layer_adapter_parameters"] == 30
    assert invariants["layer_adapter_trainable_parameters"] == 30
    assert invariants["transformer_layer_norm_parameters"] == 16
    assert invariants["transformer_layer_norm_trainable_parameters"] == 16
    assert invariants["wavlm_other_trainable_parameters"] == 0
    _validate_model_invariants(model, invariants)


def test_l_adapter_invariants_reject_non_norm_backbone_parameter() -> None:
    model = _AdapterModel()
    next(model.encoder.wavlm.feature_projection.parameters()).requires_grad = True
    invariants = _model_invariants(model)

    try:
        _validate_model_invariants(model, invariants)
    except RuntimeError as exc:
        assert "outside transformer LayerNorm" in str(exc)
    else:
        raise AssertionError("trainable non-LayerNorm backbone escaped validation")
