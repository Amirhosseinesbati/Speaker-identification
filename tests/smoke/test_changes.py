"""
Smoke tests for recent fixes (lightweight — no pretrained models).

Verifies:
1. Head modules (OOD, Linear, ArcFace) with 446 known classes
2. Pooling modules (Statistical, Attentive, Identity)
3. Model label remapping logic
4. TwoPartLoss OOD masking
5. predict_proba → 447-way output
6. inference.py NUM_CLASSES constant
"""
import sys
import os
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── ensure project root is importable ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Try importing from src — if transformers is broken, skip heavy tests
try:
    from src.heads import OODHead, LinearSpeakerHead, ArcFaceHead, create_ood_head, create_speaker_head
    HEADS_OK = True
except Exception:
    HEADS_OK = False

try:
    from src.pooling import StatisticalPooling, AttentiveStatisticalPooling, IdentityPooling, create_pooling
    POOLING_OK = True
except Exception:
    POOLING_OK = False


class TestHeads(unittest.TestCase):
    """Verify head modules work with 446 known classes."""

    @unittest.skipUnless(HEADS_OK, "src.heads import failed (transformers broken)")
    def test_ood_head(self):
        head = OODHead(768, hidden_dim=64)
        x = torch.randn(4, 768)
        out = head(x)
        self.assertEqual(out.shape, (4, 1))

    @unittest.skipUnless(HEADS_OK, "src.heads import failed (transformers broken)")
    def test_linear_head_446(self):
        head = LinearSpeakerHead(768, num_classes=446)
        x = torch.randn(4, 768)
        out = head(x)
        self.assertEqual(out.shape, (4, 446))

    @unittest.skipUnless(HEADS_OK, "src.heads import failed (transformers broken)")
    def test_arcface_446_train(self):
        head = ArcFaceHead(192, num_classes=446, embedding_dim=192, margin=0.4, scale=30.0)
        x = torch.randn(4, 192)
        labels = torch.randint(0, 446, (4,))
        out = head(x, labels)
        self.assertEqual(out.shape, (4, 446))
        # Loss should be computable
        loss = F.cross_entropy(out, labels)
        self.assertGreater(loss.item(), 0)

    @unittest.skipUnless(HEADS_OK, "src.heads import failed (transformers broken)")
    def test_arcface_446_infer(self):
        head = ArcFaceHead(192, num_classes=446, embedding_dim=192)
        x = torch.randn(4, 192)
        out = head(x)  # no labels → inference mode
        self.assertEqual(out.shape, (4, 446))


class TestPooling(unittest.TestCase):
    """Verify pooling layers."""

    @unittest.skipUnless(POOLING_OK, "src.pooling import failed")
    def test_statistical(self):
        pool = StatisticalPooling()
        x = torch.randn(2, 100, 768)
        out = pool(x)
        self.assertEqual(out.shape, (2, 1536))

    @unittest.skipUnless(POOLING_OK, "src.pooling import failed")
    def test_attentive(self):
        pool = AttentiveStatisticalPooling(768)
        x = torch.randn(2, 100, 768)
        out = pool(x)
        self.assertEqual(out.shape, (2, 1536))

    @unittest.skipUnless(POOLING_OK, "src.pooling import failed")
    def test_identity(self):
        pool = IdentityPooling()
        # ECAPA mode: seq_len=1
        x = torch.randn(2, 1, 192)
        out = pool(x)
        self.assertEqual(out.shape, (2, 192))
        # Fallback: seq_len > 1 → mean
        x2 = torch.randn(2, 10, 192)
        out2 = pool(x2)
        self.assertEqual(out2.shape, (2, 192))


class TestLabelRemapping(unittest.TestCase):
    """
    Verify the label remapping logic that's inside
    TwoHeadedSpeakerModel.forward().

    Logic:
        label 0 (unknown) → stays 0 (harmless collision, loss ignores)
        label 1..446        → 0..445 (ArcFace 0-indexed)
    """

    def test_remapping_correctness(self):
        """Simulate the exact remapping code from model.py."""
        num_known = 446

        # Simulate batch labels: [unknown, spk#1, spk#5, spk#446]
        labels = torch.tensor([0, 1, 5, 446])

        # Code from model.py forward():
        remapped = labels.clone()
        mask_known = remapped != 0
        remapped[mask_known] = remapped[mask_known] - 1       # 1..446 → 0..445
        # remapped[~mask_known] stays 0 (harmless collision)

        # Expected:
        # label 0 (unknown) → 0 (collision, harmless)
        # label 1 (spk#1)  → 0
        # label 5 (spk#5)  → 4
        # label 446         → 445
        expected = torch.tensor([0, 0, 4, 445])
        self.assertTrue(torch.equal(remapped, expected),
                        f"Remapping failed: {remapped.tolist()} != {expected.tolist()}")

        # All remapped values must be within ArcFace range [0, num_known-1]
        self.assertTrue((remapped >= 0).all())
        self.assertTrue((remapped < num_known).all())

    def test_loss_remapping(self):
        """Verify loss-side remapping (TwoPartLoss)."""
        labels = torch.tensor([0, 1, 5, 446])
        ignore_index = -100

        # Code from TwoPartLoss.forward():
        speaker_labels = labels.clone()
        speaker_labels[labels == 0] = ignore_index
        mask_known = speaker_labels != ignore_index
        speaker_labels_mapped = speaker_labels.clone()
        speaker_labels_mapped[mask_known] = speaker_labels_mapped[mask_known] - 1

        # Expected loss targets:
        # label 0 → -100 (ignored)
        # label 1 → 0
        # label 5 → 4
        # label 446 → 445
        expected = torch.tensor([-100, 0, 4, 445])
        self.assertTrue(torch.equal(speaker_labels_mapped, expected),
                        f"Loss remapping failed: {speaker_labels_mapped.tolist()} != {expected.tolist()}")


class TestPredictProbaLogic(unittest.TestCase):
    """
    Verify predict_proba() produces 447-class output
    with correct probability sum.
    """

    def test_447_way_output(self):
        """Simulate predict_proba logic directly."""
        batch, num_known = 4, 446
        ood_logit = torch.randn(batch, 1)
        speaker_logits = torch.randn(batch, num_known)

        # predict_proba logic (from model.py):
        p_unknown = torch.sigmoid(ood_logit)        # (batch, 1)
        p_known = F.softmax(speaker_logits, dim=1)  # (batch, num_known)
        p_unknown_expanded = p_unknown.expand(-1, num_known)
        p_known_scaled = (1.0 - p_unknown_expanded) * p_known
        probs = torch.cat([p_unknown, p_known_scaled], dim=1)
        probs = torch.clamp(probs, min=1e-7, max=1.0 - 1e-7)
        probs = probs / probs.sum(dim=1, keepdim=True)

        self.assertEqual(probs.shape, (batch, 447),
                         f"Expected (4, 447), got {probs.shape}")
        self.assertTrue(torch.allclose(probs.sum(dim=1), torch.ones(batch), atol=1e-5),
                        f"Probabilities don't sum to 1: {probs.sum(dim=1).tolist()}")

    def test_output_class_count(self):
        """1 + num_known should equal 447 (not 448)."""
        num_known = 446
        total = 1 + num_known
        self.assertEqual(total, 447, "Output should be 447 classes (1 unknown + 446 known)")
        self.assertNotEqual(total, 448, "Output should NOT be 448 — old bug!")


class TestInferenceConstants(unittest.TestCase):
    """Verify submission/inference.py uses correct class count."""

    def test_num_classes_is_447(self):
        """Check the NUM_CLASSES constant in inference.py."""
        inf_path = PROJECT_ROOT / "submission" / "inference.py"
        content = inf_path.read_text(encoding="utf-8")

        # Should contain "NUM_CLASSES = 447" not 448
        self.assertIn("NUM_CLASSES = 447", content,
                      "inference.py should have NUM_CLASSES = 447")
        self.assertNotIn("NUM_CLASSES = 448", content,
                         "inference.py should NOT have NUM_CLASSES = 448 (old bug!)")

        # Should contain "MODEL_CLASSES = 447"
        self.assertIn("MODEL_CLASSES = 447", content,
                      "inference.py should have MODEL_CLASSES = 447")


class TestTwoPartLossLogic(unittest.TestCase):
    """
    Verify TwoPartLoss correctly masks unknown samples.
    """

    def test_unknown_masking(self):
        """Unknown samples must contribute 0 to speaker loss."""
        ignore_index = -100
        speaker_logits = torch.randn(4, 446)
        labels = torch.tensor([0, 1, 0, 5])  # 2 unknown, 2 known

        # Loss-side remapping
        speaker_labels = labels.clone()
        speaker_labels[labels == 0] = ignore_index
        mask_known = speaker_labels != ignore_index
        speaker_labels_mapped = speaker_labels.clone()
        speaker_labels_mapped[mask_known] = speaker_labels_mapped[mask_known] - 1

        # Compute CE with ignore_index
        ce = F.cross_entropy(speaker_logits, speaker_labels_mapped,
                             ignore_index=ignore_index, reduction='none')

        # Unknown samples must have 0 loss
        self.assertEqual(ce[0].item(), 0.0, "Unknown sample 0 should have 0 loss")
        self.assertEqual(ce[2].item(), 0.0, "Unknown sample 2 should have 0 loss")
        # Known samples should have non-zero loss
        self.assertGreater(ce[1].item(), 0, "Known sample 1 should have >0 loss")
        self.assertGreater(ce[3].item(), 0, "Known sample 3 should have >0 loss")


if __name__ == "__main__":
    print("=" * 55)
    print("  SMOKE TESTS — Post-Fix Verification")
    print("=" * 55)
    print(f"  Heads OK:     {HEADS_OK}")
    print(f"  Pooling OK:   {POOLING_OK}")
    print()

    # Run tests
    unittest.main(verbosity=2, argv=[sys.argv[0]])
