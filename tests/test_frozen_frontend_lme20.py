from __future__ import annotations

import numpy as np
import torch

from scripts.audit_frozen_frontend_lme20 import (
    PRIMARY_VARIANT,
    encode_multiwindow,
    evidence_variants,
    evaluate_gate,
)


class _DummyEncoder(torch.nn.Module):
    def forward(self, waveforms: torch.Tensor):
        value = waveforms.mean(dim=(1, 2), keepdim=False)
        hidden = torch.stack([value, torch.ones_like(value)], dim=1).unsqueeze(1)
        return hidden, None


def test_multiwindow_frontend_averages_before_normalisation():
    windows = torch.tensor([[[[1.0, 1.0]], [[3.0, 3.0]]]])
    actual = encode_multiwindow(_DummyEncoder(), windows)
    expected = torch.tensor([[2.0, 1.0]])
    expected = torch.nn.functional.normalize(expected, p=2, dim=1)
    torch.testing.assert_close(actual, expected)


def test_primary_evidence_is_fixed_equal_prototype_average():
    head = np.full((2, 447), 1.0 / 447.0)
    campp = np.full((2, 447), 0.1)
    eres = np.full((2, 447), 0.3)
    campp_max = np.asarray([0.4, 0.6])
    eres_max = np.asarray([0.8, 0.2])
    variants = evidence_variants(
        campp_head=head,
        campp_prototype=campp,
        campp_max_score=campp_max,
        frontend_prototype=eres,
        frontend_max_score=eres_max,
    )
    primary = variants[PRIMARY_VARIANT]
    np.testing.assert_array_equal(primary[0], head)
    np.testing.assert_allclose(primary[1], 0.2)
    np.testing.assert_allclose(primary[2], [0.6, 0.4])


def test_gate_requires_effect_guardrails_fold_stability_and_rescue():
    gate = {
        "minimum_aggregate_macro_gain": 0.001,
        "maximum_aggregate_known_accuracy_drop": 0.001,
        "maximum_aggregate_ood_f1_drop": 0.001,
        "minimum_positive_folds": 2,
        "minimum_fold_macro_delta": -0.001,
        "minimum_baseline_error_rescue_rate": 0.15,
        "require_more_rescued_than_introduced": True,
    }
    passed = evaluate_gate(
        aggregate_delta={
            "macro_f1": 0.0015,
            "accuracy": 0.001,
            "known_accuracy": -0.0005,
            "ood_f1": 0.001,
        },
        fold_macro_deltas=[0.002, 0.001, -0.0005],
        rescued_errors=25,
        introduced_errors=10,
        baseline_errors=100,
        gate=gate,
    )
    assert passed["passed"] is True

    failed = evaluate_gate(
        aggregate_delta={
            "macro_f1": 0.0015,
            "accuracy": 0.001,
            "known_accuracy": -0.0011,
            "ood_f1": 0.001,
        },
        fold_macro_deltas=[0.002, 0.001, -0.0005],
        rescued_errors=25,
        introduced_errors=10,
        baseline_errors=100,
        gate=gate,
    )
    assert failed["passed"] is False
    assert failed["checks"]["aggregate_known_guardrail"] is False
