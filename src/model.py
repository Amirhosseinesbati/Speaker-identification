"""
Modular Two-Headed Speaker Identification Model.

Architecture:
    Encoder (WavLM/ECAPA/CAM++/ERes2NetV2/TitaNet) → Pooling →
    ├── OOD Head (Linear → Sigmoid)      → P(unknown)
    └── Speaker Head (Linear/ArcFace)     → P(known_i)

Fusion:
    p[0] = P_unknown
    p[i] = (1 - P_unknown) * P_known_i

The model is constructed from composable components via create_model_from_config().
For backward compatibility, TwoHeadedWavLM is kept as an alias.
"""

from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.encoders import BaseEncoder
from src.pooling import StatisticalPooling


class TwoHeadedSpeakerModel(nn.Module):
    """
    Modular two-headed architecture for open-set speaker identification.

    Components are injected via constructor — use create_model_from_config()
    to build from a YAML config dict.

    Output:
        ood_logit:    (batch, 1)       — raw logit, sigmoid → P(unknown)
        speaker_logits: (batch, N)     — logits over known speakers
    """

    def __init__(
        self,
        encoder: BaseEncoder,
        pooling: nn.Module,
        speaker_head: nn.Module,
        ood_head: nn.Module,
        num_known_speakers: int,
        encoder_name: str = "unknown",
    ):
        super().__init__()
        self.encoder = encoder
        self.pooling = pooling
        self.head_speaker = speaker_head
        self.head_ood = ood_head
        self.num_known_speakers = num_known_speakers
        self.encoder_name = encoder_name

    def forward(
        self,
        waveforms: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            waveforms: (batch, 1, T) — raw audio, 16kHz
            labels:    Optional (batch,) — speaker labels for ArcFace training.
                       None for inference.

        Returns:
            ood_logit:      (batch, 1)  — raw logit (sigmoid → P(unknown))
            speaker_logits: (batch, N)  — logits over known speakers
        """
        # ── Encoder ──
        hidden_states, lengths = self.encoder(waveforms)
        # hidden_states: (batch, seq_len, hidden_dim)

        # ── Pooling ──
        pooled = self.pooling(hidden_states)  # (batch, pooled_dim)

        # ── OOD Head ──
        ood_logit = self.head_ood(pooled)  # (batch, 1)

        # ── Speaker Head ──
        # ArcFace needs labels during training, Linear ignores them
        if labels is not None and hasattr(self.head_speaker, 'forward'):
            import inspect
            sig = inspect.signature(self.head_speaker.forward)
            if 'labels' in sig.parameters:
                # Remap: original labels [0, 1..446] → ArcFace [0, 0..445]
                #   Known speakers 1..446 → ArcFace classes 0..445
                #   Unknown (label 0)     → ArcFace class 0 (same as speaker #1).
                #
                # This collision is HARMLESS because:
                # (a) TwoPartLoss ignores unknown samples (ignore_index=-100),
                #     so gradient contribution from unknown is always 0.
                # (b) ArcFace weight[0] is ONLY trained by speaker #1 samples.
                # (c) The ArcFace margin on unknown's output[0] never backprops.
                remapped = labels.clone()
                mask_known = remapped != 0
                remapped[mask_known] = remapped[mask_known] - 1       # 1..446 → 0..445
                # remapped[~mask_known] stays 0 (harmless collision)
                speaker_logits = self.head_speaker(pooled, labels=remapped)
            else:
                speaker_logits = self.head_speaker(pooled)
        else:
            speaker_logits = self.head_speaker(pooled)

        return ood_logit, speaker_logits

    def predict_proba(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Get proper probability vector over all classes (0..num_known).

        Formula:
            p[0] = sigmoid(ood_logit)
            p[i] = (1 - p[0]) * softmax(speaker_logits)[i]

        Args:
            waveforms: (batch, 1, T) or (batch, W, 1, T) — multi-window inputs
                       are run window-by-window and the logits averaged.

        Returns:
            probs: (batch, 1 + num_known) — sum(dim=1) ≈ 1.0
        """
        if waveforms.dim() == 4:
            # Loop windows (peak stays at (B, 1, T)) and average the LOGITS,
            # then fuse once — identical math to a flattened (B*W, ...) batch.
            B, W = waveforms.shape[0], waveforms.shape[1]
            ood_sum = spk_sum = None
            for w in range(W):
                o, s = self.forward(waveforms[:, w], labels=None)
                ood_sum = o if ood_sum is None else ood_sum + o
                spk_sum = s if spk_sum is None else spk_sum + s
            ood_logit = ood_sum / W
            speaker_logits = spk_sum / W
        else:
            ood_logit, speaker_logits = self.forward(waveforms, labels=None)

        # P(unknown) = sigmoid(ood_logit)
        p_unknown = torch.sigmoid(ood_logit)  # (batch, 1)

        # P(known_i) = softmax(speaker_logits)
        p_known = F.softmax(speaker_logits, dim=1)  # (batch, N)

        # Fusion: p_0 = P_unknown, p_i = (1 - P_unknown) * P_known_i
        p_unknown_expanded = p_unknown.expand(-1, self.num_known_speakers)
        p_known_scaled = (1.0 - p_unknown_expanded) * p_known

        # Concatenate: (batch, 1 + N) = (batch, 447)
        probs = torch.cat([p_unknown, p_known_scaled], dim=1)

        # Numerical safety
        probs = torch.clamp(probs, min=1e-7, max=1.0 - 1e-7)
        probs = probs / probs.sum(dim=1, keepdim=True)

        return probs

    def get_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def print_summary(self):
        """Print model architecture summary."""
        total = sum(p.numel() for p in self.parameters())
        trainable = self.get_trainable_params()
        frozen = total - trainable



# ═══════════════════════════════════════════════════════════
#  Backward Compatibility: TwoHeadedWavLM
# ═══════════════════════════════════════════════════════════

class TwoHeadedWavLM(TwoHeadedSpeakerModel):
    """
    Backward-compatible wrapper that builds the original WavLM architecture
    from a flat config dict (old-style or new-style).

    This class exists so existing checkpoint loading code and ZenML steps
    continue to work without modification.
    """

    def __init__(
        self,
        config: dict,
        num_known_speakers: int = 446,
    ):
        from src.encoders import WavLMEncoder
        from src.pooling import create_pooling
        from src.heads import OODHead, LinearSpeakerHead, create_speaker_head, create_ood_head

        model_cfg = config["model"]

        # ── Encoder ──
        encoder_type = model_cfg.get("encoder_type", "wavlm")
        if "encoder_config" in model_cfg:
            enc_cfg = model_cfg["encoder_config"].get(encoder_type, {})
            base_model = enc_cfg.get(
                "base_model",
                model_cfg.get("base_model", "microsoft/wavlm-base-plus"),
            )
            freeze_fe = enc_cfg.get(
                "freeze_feature_extractor",
                model_cfg.get("freeze_feature_extractor", False),
            )
        else:
            base_model = model_cfg.get("base_model", "microsoft/wavlm-base-plus")
            freeze_fe = model_cfg.get("freeze_feature_extractor", False)

        encoder = WavLMEncoder(
            base_model=base_model,
            freeze_feature_extractor=freeze_fe,
        )

        # ── Pooling ──
        pooling_type = model_cfg.get("pooling_type", "statistical")
        pooling = create_pooling(pooling_type, encoder.output_dim)
        pooled_dim = encoder.output_dim * pooling.output_multiplier

        # ── OOD Head ──
        ood_head = create_ood_head(config, pooled_dim)

        # ── Speaker Head ──
        speaker_head = create_speaker_head(config, pooled_dim, num_known_speakers)

        super().__init__(
            encoder=encoder,
            pooling=pooling,
            speaker_head=speaker_head,
            ood_head=ood_head,
            num_known_speakers=num_known_speakers,
            encoder_name=base_model,
        )

    # Keep old method signatures for backward compat
    def _freeze_feature_extractor(self):
        self.encoder.freeze()

    def unfreeze_feature_extractor(self):
        self.encoder.unfreeze()

    @property
    def wavlm(self):
        """Backward-compat: expose encoder.wavlm."""
        return self.encoder.wavlm


# ═══════════════════════════════════════════════════════════
#  Smoke Test
