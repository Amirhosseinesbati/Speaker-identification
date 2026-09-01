from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

import src.encoders as encoders
from src.training_utils import encoder_will_train


class _DummyWavLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature_extractor = nn.Linear(1, 1)
        self.encoder_block = nn.Linear(1, 1)
        self.config = SimpleNamespace(num_hidden_layers=2, hidden_size=1)

    def forward(self, input_values, output_hidden_states=False):
        base = input_values.unsqueeze(-1)
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

    torch.testing.assert_close(output, torch.tensor([[[2.0], [3.0]]]))
    assert lengths is None
    assert model.layer_weights.requires_grad is True
    assert all(not parameter.requires_grad for parameter in model.wavlm.parameters())


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

    assert encoder_will_train(frozen) is False
    assert encoder_will_train(legacy_full_ft) is True
