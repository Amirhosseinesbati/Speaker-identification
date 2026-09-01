from __future__ import annotations

import torch
from torch import nn

from scripts.probe_evaluation_batch import _model_invariants


class _Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.wavlm = nn.Linear(3, 4)
        for parameter in self.wavlm.parameters():
            parameter.requires_grad = False
        self.layer_weights = nn.Parameter(torch.zeros(13))


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
