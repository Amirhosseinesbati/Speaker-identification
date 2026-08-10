"""
Ensemble Model for Multi-Encoder Speaker Identification.

Combines multiple TwoHeadedSpeakerModel instances via:
    - Average fusion: mean of all probability vectors
    - Learned MLP fusion: MLP on concatenated probability vectors

The ensemble produces a single (batch, 447) probability vector from
multiple base models with potentially different encoder architectures.

Usage:
    # Average fusion
    ensemble = EnsembleModel([model_wavlm, model_ecapa, model_campp, model_eres2net, model_titanet])

    # Learned fusion
    fusion = LearnedFusion(num_models=3, num_classes=447)
    ensemble = EnsembleModel(models, fusion=fusion)
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import TwoHeadedSpeakerModel


class LearnedFusion(nn.Module):
    """
    MLP-based learned fusion of ensemble probability vectors.

    Input: concatenated probability vectors from N models: (batch, N * num_classes)
    Output: fused probability vector: (batch, num_classes)

    Architecture:
        Linear(N*C, 512) → ReLU → Dropout(0.3) → Linear(512, C) → Softmax
    """

    def __init__(
        self,
        num_models: int,
        num_classes: int,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_models = num_models
        self.num_classes = num_classes

        self.mlp = nn.Sequential(
            nn.Linear(num_models * num_classes, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, probs_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            probs_list: List of (batch, num_classes) probability tensors

        Returns:
            fused_probs: (batch, num_classes) — sum(dim=1) ≈ 1.0
        """
        concat = torch.cat(probs_list, dim=1)  # (batch, N*C)
        logits = self.mlp(concat)
        return F.softmax(logits, dim=1)


class EnsembleModel(nn.Module):
    """
    Ensemble of multiple TwoHeadedSpeakerModel instances.

    Supports two fusion strategies:
        - "average":    Arithmetic mean of all model probability vectors
        - "learned_mlp": MLP-based fusion (requires training)

    Usage with average fusion:
        ensemble = EnsembleModel([model1, model2, model3], fusion_method="average")
        probs = ensemble.predict_proba(waveforms)

    Usage with learned fusion:
        fusion = LearnedFusion(num_models=3, num_classes=447)
        ensemble = EnsembleModel(models, fusion=fusion)
        # Train fusion MLP separately on validation set
        probs = ensemble.predict_proba(waveforms)
    """

    def __init__(
        self,
        models: List[TwoHeadedSpeakerModel],
        fusion_method: str = "average",
        fusion: Optional[LearnedFusion] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            models:        List of trained TwoHeadedSpeakerModel instances
            fusion_method: "average" or "learned_mlp" (if fusion=None)
            fusion:        Pre-built LearnedFusion module (optional, overrides fusion_method)
            device:        Device to run models on
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
        self.num_classes = self.models[0].num_known_speakers + 1  # + unknown

        if fusion is not None:
            self.fusion = fusion
            self.fusion_method = "learned_mlp"
        elif fusion_method == "learned_mlp":
            self.fusion = LearnedFusion(self.num_models, self.num_classes)
            self.fusion_method = "learned_mlp"
        else:
            self.fusion = None
            self.fusion_method = "average"

        print(f"  🎯 Ensemble: {self.num_models} models, "
              f"fusion={self.fusion_method}, classes={self.num_classes}")

    def forward(
        self,
        waveforms: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Return list of probability vectors from each model."""
        all_probs = []
        for model in self.models:
            probs = model.predict_proba(waveforms)
            all_probs.append(probs)
        return all_probs

    def predict_proba(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Compute fused probability vector.

        Args:
            waveforms: (batch, 1, T)

        Returns:
            probs: (batch, num_classes) — sum(dim=1) ≈ 1.0
        """
        all_probs = self.forward(waveforms)

        if self.fusion_method == "average":
            # Element-wise mean
            stacked = torch.stack(all_probs, dim=0)  # (N, batch, C)
            return stacked.mean(dim=0)  # (batch, C)

        elif self.fusion_method == "learned_mlp":
            return self.fusion(all_probs)

        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

    def get_trainable_params(self) -> int:
        """Count trainable parameters (only fusion MLP if learned)."""
        if self.fusion is not None:
            return sum(p.numel() for p in self.fusion.parameters() if p.requires_grad)
        return 0

    def eval(self):
        """Set all models and fusion to eval mode."""
        super().eval()
        for model in self.models:
            model.eval()

    def train(self, mode: bool = True):
        """Set fusion MLP to train mode (models stay frozen)."""
        super().train(mode)
        # Keep base models in eval mode — only train fusion
        for model in self.models:
            model.eval()
        if self.fusion is not None:
            self.fusion.train(mode)


# ═══════════════════════════════════════════════════════════
#  Smoke Test
# ═══════════════════════════════════════════════════════════

def _smoke_test():
    print("=" * 50)
    print("  Ensemble Smoke Test")
    print("=" * 50)

    # Create dummy models (simple linear for fast testing)
    class DummyModel(nn.Module):
        def __init__(self, num_known=10):
            super().__init__()
            self.num_known_speakers = num_known
            self.linear = nn.Linear(80000, num_known + 1)

        def predict_proba(self, x):
            x = x.squeeze(1)
            logits = self.linear(x)
            return F.softmax(logits, dim=1)

    num_known = 10
    models = [DummyModel(num_known) for _ in range(3)]

    # ── Average fusion ──
    ensemble_avg = EnsembleModel(models, fusion_method="average")
    waveforms = torch.randn(4, 1, 80000)
    probs_avg = ensemble_avg.predict_proba(waveforms)
    assert probs_avg.shape == (4, num_known + 1)
    assert torch.allclose(probs_avg.sum(dim=1), torch.ones(4), atol=1e-5)
    print(f"  Average fusion: {probs_avg.shape} ✅")

    # ── Learned MLP fusion ──
    ensemble_mlp = EnsembleModel(models, fusion_method="learned_mlp")
    probs_mlp = ensemble_mlp.predict_proba(waveforms)
    assert probs_mlp.shape == (4, num_known + 1)
    assert torch.allclose(probs_mlp.sum(dim=1), torch.ones(4), atol=1e-5)
    print(f"  Learned MLP:    {probs_mlp.shape} ✅")

    # ── Training mode ──
    ensemble_mlp.train()
    assert ensemble_mlp.fusion.training  # fusion IS training
    # Models should remain in eval mode
    for m in ensemble_mlp.models:
        assert not m.training
    print(f"  Train/eval mode: ✅")

    # ── Gradients flow through fusion ──
    opt = torch.optim.Adam(ensemble_mlp.fusion.parameters(), lr=1e-3)
    ensemble_mlp.train()
    probs = ensemble_mlp.predict_proba(waveforms)
    target = torch.randint(0, num_known + 1, (4,))
    loss = F.cross_entropy(torch.log(probs + 1e-7), target)
    loss.backward()
    opt.step()
    assert loss.item() > 0
    print(f"  Fusion training: loss={loss.item():.4f} ✅")

    print()
    print("  ALL ENSEMBLE TESTS PASSED ✅")


if __name__ == "__main__":
    _smoke_test()
