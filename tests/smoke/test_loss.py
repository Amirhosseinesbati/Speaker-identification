"""Smoke test for loss functions: FocalLoss and TwoPartLoss."""

import torch
import torch.nn as nn
import pytest

from src.train import FocalLoss, TwoPartLoss


class TestFocalLoss:
    def test_output_is_scalar(self):
        loss_fn = FocalLoss(gamma=2.0)
        logits = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_gamma_zero_equals_cross_entropy(self):
        """With gamma=0, focal loss ≈ cross entropy."""
        ce_loss = nn.CrossEntropyLoss()
        focal_loss = FocalLoss(gamma=0.0)

        logits = torch.randn(8, 5)
        targets = torch.randint(0, 5, (8,))

        ce_val = ce_loss(logits, targets)
        focal_val = focal_loss(logits, targets)

        assert torch.allclose(ce_val, focal_val, atol=1e-5)

    def test_ignore_index(self):
        loss_fn = FocalLoss(gamma=2.0, ignore_index=-100)
        logits = torch.randn(4, 5)
        targets = torch.tensor([0, 1, -100, 2])
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_all_ignored_returns_zero(self):
        loss_fn = FocalLoss(gamma=2.0, ignore_index=-100)
        logits = torch.randn(4, 5)
        targets = torch.full((4,), -100)
        loss = loss_fn(logits, targets)
        assert loss.item() == 0.0

    def test_hard_examples_weighted_more(self):
        """Hard examples (low confidence) should contribute more to loss."""
        loss_fn = FocalLoss(gamma=2.0, reduction="none")

        # Easy example: very high logit for correct class
        logits_easy = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0]])
        targets = torch.tensor([0])

        # Hard example: nearly uniform logits
        logits_hard = torch.tensor([[0.1, 0.1, 0.1, 0.1, 0.1]])
        targets_hard = torch.tensor([0])

        loss_easy = loss_fn(logits_easy, targets).item()
        loss_hard = loss_fn(logits_hard, targets_hard).item()

        # Hard example should have HIGHER unweighted CE but lower focal weight
        # Focal loss should still give hard example more total weight overall
        # Actually focal loss reduces the contribution of easy examples
        assert loss_easy < loss_hard, \
            f"Easy loss ({loss_easy:.6f}) should be < hard loss ({loss_hard:.6f})"


class TestTwoPartLoss:
    def test_output_structure(self):
        criterion = TwoPartLoss(use_focal=True)
        ood_logits = torch.randn(4, 1)
        speaker_logits = torch.randn(4, 446)
        labels = torch.tensor([0, 1, 2, 3])  # mix of known and unknown

        total, components = criterion(ood_logits, speaker_logits, labels)
        assert total.ndim == 0
        assert total.item() > 0
        assert "loss_ood" in components
        assert "loss_speaker" in components
        assert "loss_total" in components

    def test_all_unknown(self):
        criterion = TwoPartLoss(use_focal=True)
        ood_logits = torch.randn(8, 1)
        speaker_logits = torch.randn(8, 446)
        labels = torch.zeros(8, dtype=torch.long)  # all unknown

        total, components = criterion(ood_logits, speaker_logits, labels)
        assert total.ndim == 0
        assert components["loss_ood"] > 0

    def test_all_known(self):
        criterion = TwoPartLoss(use_focal=False)
        ood_logits = torch.randn(8, 1)
        speaker_logits = torch.randn(8, 446)
        labels = torch.arange(1, 9, dtype=torch.long)  # all known

        total, components = criterion(ood_logits, speaker_logits, labels)
        assert total.ndim == 0
        assert components["loss_speaker"] > 0

    def test_focal_vs_ce(self):
        """Both modes should produce valid losses."""
        for use_focal in [True, False]:
            criterion = TwoPartLoss(use_focal=use_focal)
            ood = torch.randn(4, 1)
            spk = torch.randn(4, 446)
            labels = torch.tensor([0, 1, 5, 0])

            total, comp = criterion(ood, spk, labels)
            assert not torch.isnan(total)
            assert not torch.isinf(total)
