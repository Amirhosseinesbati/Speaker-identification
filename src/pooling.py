"""
Pooling layers for aggregating frame-level speaker features into
utterance-level embeddings.

Classes:
    StatisticalPooling           — mean + std (original)
    AttentiveStatisticalPooling  — attention-weighted mean + std (improved)
"""

from typing import Optional

import torch
import torch.nn as nn


class StatisticalPooling(nn.Module):
    """
    Aggregates frame-level features into a fixed-size utterance-level embedding
    by concatenating mean and standard deviation across the time dimension.

    Input:  (batch, seq_len, feat_dim)
    Output: (batch, feat_dim * 2)
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:    (batch, seq_len, feat_dim) — frame-level features
            mask: Optional (batch, seq_len) boolean mask (True = valid frame)
        Returns:
            (batch, feat_dim * 2) — concatenated [mean; std]
        """
        if mask is not None:
            mask = mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
            x = x * mask
            denom = mask.sum(dim=1).clamp(min=1)
            mean = x.sum(dim=1) / denom
            # Unbiased variance with Bessel correction
            var = ((x - mean.unsqueeze(1)) ** 2 * mask).sum(dim=1) / (denom - 1).clamp(min=1)
            std = torch.sqrt(var.clamp(min=1e-8))
        else:
            mean = x.mean(dim=1)  # (batch, feat_dim)
            std = x.std(dim=1, unbiased=False)  # (batch, feat_dim)

        return torch.cat([mean, std], dim=1)  # (batch, feat_dim * 2)

    @property
    def output_multiplier(self) -> int:
        """How many times the input dim the output is."""
        return 2


class AttentiveStatisticalPooling(nn.Module):
    """
    Attention-weighted statistical pooling.

    Instead of a simple mean, this computes a weighted average where each
    frame's importance is learned via a small attention network. This is
    the standard pooling used in modern speaker recognition systems
    (ECAPA-TDNN, x-vector variants).

    Architecture:
        attention = softmax(Linear(Tanh(Linear(x))))
        weighted_mean = sum(attention * x)
        weighted_std  = sqrt(sum(attention * (x - mean)^2))

    Input:  (batch, seq_len, feat_dim)
    Output: (batch, feat_dim * 2)
    """

    def __init__(self, input_dim: int, attention_dim: Optional[int] = None):
        """
        Args:
            input_dim:     Feature dimension (e.g., 768 for WavLM-base)
            attention_dim: Hidden dim of the attention MLP (default: input_dim // 2)
        """
        super().__init__()
        if attention_dim is None:
            attention_dim = input_dim // 2

        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )
        self.input_dim = input_dim

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:    (batch, seq_len, feat_dim) — frame-level features
            mask: Optional (batch, seq_len) bool (True = valid, False = pad)
        Returns:
            (batch, feat_dim * 2) — concatenated [weighted_mean; weighted_std]
        """
        # Compute attention weights: (batch, seq_len, 1)
        attn_scores = self.attention(x)  # (batch, seq_len, 1)

        if mask is not None:
            # Large negative for masked positions → 0 after softmax
            attn_scores = attn_scores.masked_fill(~mask.unsqueeze(-1), -1e9)

        attn_weights = torch.softmax(attn_scores, dim=1)  # (batch, seq_len, 1)

        # Weighted mean: (batch, feat_dim)
        weighted_mean = torch.sum(attn_weights * x, dim=1)

        # Weighted std
        diff_sq = (x - weighted_mean.unsqueeze(1)) ** 2
        weighted_var = torch.sum(attn_weights * diff_sq, dim=1)
        weighted_std = torch.sqrt(weighted_var.clamp(min=1e-8))

        return torch.cat([weighted_mean, weighted_std], dim=1)  # (batch, feat_dim * 2)

    @property
    def output_multiplier(self) -> int:
        return 2


class IdentityPooling(nn.Module):
    """
    Identity pooling — passes input through unchanged.

    Used when the encoder already produces utterance-level embeddings
    (e.g., ECAPA-TDNN has internal attentive pooling).

    Input:  (batch, seq_len_or_1, feat_dim)
    Output: (batch, feat_dim)
    """

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.ndim == 3:
            # Squeeze the sequence dimension if it's 1
            if x.size(1) == 1:
                return x.squeeze(1)
            # Otherwise mean-pool (fallback)
            return x.mean(dim=1)
        return x

    @property
    def output_multiplier(self) -> int:
        return 1


# ═══════════════════════════════════════════════════════════
#  Pooling Factory
# ═══════════════════════════════════════════════════════════

def create_pooling(
    pooling_type: str,
    input_dim: int,
) -> nn.Module:
    """
    Create a pooling layer based on config string.

    Args:
        pooling_type: "statistical" | "attentive"
        input_dim:    Feature dimension of the encoder output

    Returns:
        nn.Module pooling layer
    """
    pooling_type = pooling_type.lower().strip()

    if pooling_type == "statistical":
        return StatisticalPooling()
    elif pooling_type == "attentive":
        return AttentiveStatisticalPooling(input_dim)
    elif pooling_type == "identity":
        return IdentityPooling()
    else:
        raise ValueError(
            f"Unknown pooling_type: '{pooling_type}'. "
            f"Expected 'statistical', 'attentive', or 'identity'."
        )


# ═══════════════════════════════════════════════════════════
#  Smoke Test
