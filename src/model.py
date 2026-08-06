"""
Phase 2: Two-Headed WavLM Architecture for Open-Set Speaker Identification.

Architecture:
- Base: microsoft/wavlm-base-plus (HuggingFace)
- Pooling: StatisticalPooling (mean + std) over sequence dimension
- Head 1 (OOD Detector): Linear → Sigmoid → P_unknown
- Head 2 (Speaker ID): Linear → Softmax → P_known_i (446 classes)
- Fusion: p_0 = P_unknown, p_i = (1 - P_unknown) * P_known_i
- Output: (batch, 447) with sum(dim=1) ≈ 1.0
"""

from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import yaml
from transformers import WavLMModel, WavLMConfig


# ─────────────────────────────────────────────────────────
#  Statistical Pooling
# ─────────────────────────────────────────────────────────

class StatisticalPooling(nn.Module):
    """
    Aggregates frame-level features into a fixed-size utterance-level embedding
    by concatenating mean and standard deviation across the time dimension.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, feat_dim) — WavLM hidden states
            mask: Optional (batch, seq_len) boolean mask (True = valid)
        Returns:
            (batch, feat_dim * 2) — concatenated mean and std
        """
        if mask is not None:
            mask = mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
            x = x * mask
            denom = mask.sum(dim=1).clamp(min=1)
            mean = x.sum(dim=1) / denom
            # Use unbiased std with Bessel correction
            var = ((x - mean.unsqueeze(1)) ** 2 * mask).sum(dim=1) / (denom - 1).clamp(min=1)
            std = torch.sqrt(var.clamp(min=1e-8))
        else:
            mean = x.mean(dim=1)  # (batch, feat_dim)
            std = x.std(dim=1, unbiased=False)  # (batch, feat_dim)

        return torch.cat([mean, std], dim=1)  # (batch, feat_dim * 2)


# ─────────────────────────────────────────────────────────
#  Two-Headed WavLM Model
# ─────────────────────────────────────────────────────────

class TwoHeadedWavLM(nn.Module):
    """
    Two-Headed Cascade Architecture for Open-Set Speaker Identification.

    - Head 1 (OOD): Detects whether the speaker is unknown (class 0) or known.
    - Head 2 (Speaker): Classifies among 446 known speakers.
    - Combined output: 447-dimensional probability vector.
    """

    def __init__(
        self,
        config: dict,
        num_known_speakers: int = 446,
    ):
        super().__init__()
        self.config = config
        model_cfg = config["model"]
        audio_cfg = config["audio"]

        # ── Resolve encoder config (backward-compat: old flat format → new nested) ──
        encoder_type = model_cfg.get("encoder_type", "wavlm")
        if "encoder_config" in model_cfg:
            enc_cfg = model_cfg["encoder_config"].get(encoder_type, {})
            base_model = enc_cfg.get("base_model", model_cfg.get("base_model", "microsoft/wavlm-base-plus"))
            freeze_fe = enc_cfg.get("freeze_feature_extractor",
                                    model_cfg.get("freeze_feature_extractor", False))
        else:
            # Old flat config format
            base_model = model_cfg.get("base_model", "microsoft/wavlm-base-plus")
            freeze_fe = model_cfg.get("freeze_feature_extractor", False)

        # ── Base WavLM Model ──
        self.wavlm = WavLMModel.from_pretrained(base_model)

        # Feature extractor freeze toggle
        if freeze_fe:
            self._freeze_feature_extractor()
            print("  🔒 Feature extractor: FROZEN (for local 6GB VRAM)")
        else:
            print("  🔓 Feature extractor: UNFROZEN (for Vast.ai training)")

        # ── Pooling ──
        self.pooling = StatisticalPooling()

        # WavLM-base-plus hidden size = 768 → pooled dim = 768 * 2 = 1536
        wavlm_hidden_size = self.wavlm.config.hidden_size
        pooled_dim = wavlm_hidden_size * 2
        print(f"  📐 WavLM hidden size: {wavlm_hidden_size} | Pooled dim: {pooled_dim}")

        # ── Head 1: OOD / Unknown Detector ──
        # Single neuron with Sigmoid → P(unknown)
        self.head_ood = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, 1),
        )

        # ── Head 2: Known Speaker Classifier ──
        # 446 known speakers (classes 1 to 446)
        self.head_speaker = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, num_known_speakers),
        )

        self.num_known_speakers = num_known_speakers

    def _freeze_feature_extractor(self):
        """Freeze WavLM's feature extractor (CNN layers) to save VRAM."""
        if hasattr(self.wavlm, "feature_extractor"):
            for param in self.wavlm.feature_extractor.parameters():
                param.requires_grad = False

    def unfreeze_feature_extractor(self):
        """Unfreeze for full fine-tuning (e.g., on Vast.ai)."""
        if hasattr(self.wavlm, "feature_extractor"):
            for param in self.wavlm.feature_extractor.parameters():
                param.requires_grad = True
        print("  🔓 Feature extractor unfrozen for full fine-tuning.")

    def forward(
        self,
        waveforms: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            waveforms: (batch, 1, T) — audio waveforms, resampled to 16kHz
            attention_mask: Optional (batch, T) — 1 for valid samples, 0 for padding

        Returns:
            logits: (batch, 447) — logits (before softmax for head 2),
                    sum(dim=1) not guaranteed to be 1.0 (use predict_proba for probabilities)
        """
        batch_size = waveforms.size(0)

        # WavLM expects (batch, T) — squeeze channel dim
        input_values = waveforms.squeeze(1)  # (batch, T)

        # Forward through WavLM
        wavlm_outputs = self.wavlm(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        # last_hidden_state: (batch, seq_len, hidden_size)
        hidden_states = wavlm_outputs.last_hidden_state

        # Pooling: (batch, hidden_size * 2)
        pooled = self.pooling(hidden_states)

        # Head 1: OOD logit (single value per sample)
        ood_logit = self.head_ood(pooled)  # (batch, 1)

        # Head 2: Known speaker logits (446-dim)
        speaker_logits = self.head_speaker(pooled)  # (batch, 446)

        return ood_logit, speaker_logits

    def predict_proba(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Get proper probability vectors summing to 1.0.

        Returns:
            probs: (batch, 447) — probabilities for classes 0..446
                   where probs[:, 0] = P(unknown), probs[:, 1:] = P(known_i)
        """
        ood_logit, speaker_logits = self.forward(waveforms)

        # P_unknown = sigmoid(ood_logit)
        p_unknown = torch.sigmoid(ood_logit)  # (batch, 1)

        # P_known_i = softmax(speaker_logits)  (446 classes)
        p_known = torch.softmax(speaker_logits, dim=1)  # (batch, 446)

        # Final fusion:
        # p_0 = P_unknown
        # p_i = (1 - P_unknown) * P_known_i
        p_unknown_expanded = p_unknown.expand(-1, self.num_known_speakers)
        p_known_scaled = (1.0 - p_unknown_expanded) * p_known

        # Concatenate: (batch, 1 + 446) = (batch, 447)
        probs = torch.cat([p_unknown, p_known_scaled], dim=1)

        # Clamp for numerical safety
        probs = torch.clamp(probs, min=1e-7, max=1.0 - 1e-7)
        # Renormalize to ensure sum == 1.0
        probs = probs / probs.sum(dim=1, keepdim=True)

        return probs

    def get_trainable_params(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def print_summary(self):
        """Print model architecture summary."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = self.get_trainable_params()
        frozen_params = total_params - trainable_params

        print(f"\n  📊 Model Summary:")
        print(f"     Total parameters:    {total_params:,}")
        print(f"     Trainable parameters: {trainable_params:,}")
        print(f"     Frozen parameters:    {frozen_params:,}")
        print(f"     Output dimension:     447 (0=unknown + 446 known)")


# ─────────────────────────────────────────────────────────
#  Model Factory
# ─────────────────────────────────────────────────────────

def create_model(
    config_path: str = "configs/default_config.yaml",
    num_known_speakers: int = 446,
) -> TwoHeadedWavLM:
    """Create model from config file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("=" * 50)
    print("  Building Two-Headed WavLM Model")
    print("=" * 50)

    model = TwoHeadedWavLM(config, num_known_speakers)
    model.print_summary()

    return model


# ─────────────────────────────────────────────────────────
#  Main Test Block
# ─────────────────────────────────────────────────────────

def main():
    """Instantiate model, test forward pass, verify output probabilities."""
    print("=" * 55)
    print("  Model Architecture Test")
    print("=" * 55)
    print()

    # Create model
    model = create_model()

    # Create dummy audio: batch=4, 3 seconds at 16kHz
    batch_size = 4
    num_samples = 48000  # 3s × 16kHz
    dummy_audio = torch.rand(batch_size, 1, num_samples) * 2 - 1  # [-1, 1]

    print(f"\n  Input shape: {dummy_audio.shape}")

    # Test forward (logits)
    ood_logit, speaker_logits = model(dummy_audio)
    print(f"\n  Forward pass (logits):")
    print(f"     OOD logit shape:      {ood_logit.shape}")
    print(f"     Speaker logits shape:  {speaker_logits.shape}")

    # Test predict_proba (probabilities)
    probs = model.predict_proba(dummy_audio)
    probs_sum = probs.sum(dim=1)

    print(f"\n  predict_proba output:")
    print(f"     Probabilities shape:   {probs.shape}")
    print(f"     Sum per row:           {probs_sum.tolist()}")

    # Assertions
    assert probs.shape == (batch_size, 447), f"Expected (4, 447), got {probs.shape}"
    assert torch.allclose(probs_sum, torch.ones(batch_size), atol=1e-5), \
        f"Probabilities don't sum to 1.0: {probs_sum}"

    print(f"\n  ✅ All assertions passed!")
    print(f"  ✅ Output is (batch, 447) with sum(dim=1) ≈ 1.0")
    print(f"\n  {'='*50}")
    print(f"  Model is ready for training!")
    print(f"  {'='*50}")


if __name__ == "__main__":
    main()
