"""
Speaker Encoder Backbones.

Each encoder accepts raw audio waveforms and produces frame-level hidden states.
All encoders share a common interface for seamless swapping.

Classes:
    BaseEncoder      — Abstract interface
    WavLMEncoder     — Microsoft WavLM (HuggingFace)
    ECAPAEncoder     — ECAPA-TDNN (SpeechBrain) — Phase 2.3
    HuBERTEncoder    — Facebook HuBERT (HuggingFace) — Phase 3.1
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import WavLMModel


class BaseEncoder(nn.Module, ABC):
    """Abstract encoder interface for speaker recognition backbones."""

    @abstractmethod
    def forward(
        self,
        waveforms: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            waveforms: (batch, 1, T) — raw audio, 16kHz mono

        Returns:
            hidden_states: (batch, seq_len, hidden_dim)
            lengths: Optional (batch,) — valid frame lengths (for masking)
        """
        ...

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Hidden dimension of the encoder output."""
        ...

    @abstractmethod
    def freeze(self) -> None:
        """Freeze encoder parameters (for feature extraction mode)."""
        ...

    @abstractmethod
    def unfreeze(self) -> None:
        """Unfreeze encoder parameters (for fine-tuning)."""
        ...


# ═══════════════════════════════════════════════════════════
#  WavLM Encoder
# ═══════════════════════════════════════════════════════════

class WavLMEncoder(BaseEncoder):
    """
    Microsoft WavLM encoder for speaker recognition.

    WavLM is a HuBERT-based model with gated relative position bias
    and utterance mixing, designed to preserve speaker identity.

    Supported models:
        microsoft/wavlm-base       (94M, 768-dim)
        microsoft/wavlm-base-plus (94M, 768-dim) ← default
        microsoft/wavlm-large      (317M, 1024-dim)
    """

    def __init__(
        self,
        base_model: str = "microsoft/wavlm-base-plus",
        freeze_feature_extractor: bool = True,
    ):
        super().__init__()
        self.base_model_name = base_model
        self.wavlm = WavLMModel.from_pretrained(base_model)

        if freeze_feature_extractor:
            self.freeze()
            print(f"  🔒 WavLM feature extractor: FROZEN")
        else:
            print(f"  🔓 WavLM feature extractor: UNFROZEN (full fine-tune)")

    def forward(
        self,
        waveforms: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            waveforms: (batch, 1, T)

        Returns:
            hidden_states: (batch, seq_len, hidden_dim)
            lengths: None (WavLM handles masking internally when attention_mask
                     is not provided)
        """
        # WavLM expects (batch, T) — squeeze channel dim
        input_values = waveforms.squeeze(1)  # (batch, T)

        outputs = self.wavlm(
            input_values=input_values,
            output_hidden_states=False,
        )
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)
        return hidden_states, None

    @property
    def output_dim(self) -> int:
        return self.wavlm.config.hidden_size

    def freeze(self) -> None:
        """Freeze only the CNN feature extractor (stem). Transformer stays trainable."""
        if hasattr(self.wavlm, "feature_extractor"):
            for param in self.wavlm.feature_extractor.parameters():
                param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze CNN feature extractor for full fine-tuning."""
        if hasattr(self.wavlm, "feature_extractor"):
            for param in self.wavlm.feature_extractor.parameters():
                param.requires_grad = True
        print("  🔓 WavLM feature extractor UNFROZEN.")


# ═══════════════════════════════════════════════════════════
#  HuBERT Encoder
# ═══════════════════════════════════════════════════════════

class HuBERTEncoder(BaseEncoder):
    """
    Facebook HuBERT encoder for speaker recognition.

    HuBERT (Hidden-Unit BERT) is a self-supervised speech model that
    learns discrete hidden units via clustering. While primarily designed
    for ASR, it produces strong speaker-discriminative features when
    fine-tuned, especially at intermediate layers.

    Supported models:
        facebook/hubert-base-ls960      (95M, 768-dim)
        facebook/hubert-large-ls960-ft  (317M, 1024-dim) ← default
        facebook/hubert-xlarge-ll60k    (1B, 1280-dim)
    """

    def __init__(
        self,
        base_model: str = "facebook/hubert-large-ls960-ft",
        freeze_feature_extractor: bool = True,
    ):
        super().__init__()
        self.base_model_name = base_model

        from transformers import HubertModel
        self.hubert = HubertModel.from_pretrained(base_model)

        if freeze_feature_extractor:
            self.freeze()
            print(f"  🔒 HuBERT feature extractor: FROZEN")
        else:
            print(f"  🔓 HuBERT feature extractor: UNFROZEN")

    def forward(
        self,
        waveforms: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_values = waveforms.squeeze(1)  # (batch, T)
        outputs = self.hubert(input_values=input_values, output_hidden_states=False)
        return outputs.last_hidden_state, None

    @property
    def output_dim(self) -> int:
        return self.hubert.config.hidden_size

    def freeze(self) -> None:
        if hasattr(self.hubert, "feature_extractor"):
            for param in self.hubert.feature_extractor.parameters():
                param.requires_grad = False

    def unfreeze(self) -> None:
        if hasattr(self.hubert, "feature_extractor"):
            for param in self.hubert.feature_extractor.parameters():
                param.requires_grad = True
        print("  🔓 HuBERT feature extractor UNFROZEN.")


# ═══════════════════════════════════════════════════════════
#  ECAPA-TDNN Encoder (SpeechBrain)
# ═══════════════════════════════════════════════════════════

class ECAPAEncoder(BaseEncoder):
    """
    ECAPA-TDNN encoder from SpeechBrain, pretrained on VoxCeleb.

    ECAPA-TDNN is a state-of-the-art speaker embedding model that uses:
    - Squeeze-Excitation Res2Blocks
    - Multi-layer feature aggregation
    - Attentive Statistical Pooling (built-in)

    The built-in pooling produces utterance-level 192-dim embeddings.
    When using this encoder, set pooling_type="identity" in config.

    Reference:
        Desplanques et al., "ECAPA-TDNN: Emphasized Channel Attention,
        Propagation and Aggregation in TDNN Based Speaker Verification"
        (INTERSPEECH 2020)

    Supported sources:
        speechbrain/spkrec-ecapa-voxceleb  (192-dim, VoxCeleb1+2, 0.80% EER)
    """

    def __init__(
        self,
        source: str = "speechbrain/spkrec-ecapa-voxceleb",
        freeze_encoder: bool = True,
        device: str = "cpu",
    ):
        super().__init__()
        self.source = source
        self._output_dim = 192  # ECAPA-TDNN embedding dimension
        self._frozen = freeze_encoder

        import os
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

        from speechbrain.inference.speaker import EncoderClassifier

        print(f"  Loading ECAPA-TDNN from {source}...")
        # Do NOT pass savedir on Windows — it causes symlink permission errors.
        # The model will be cached in HF Hub cache (~/.cache/huggingface).
        self.classifier = EncoderClassifier.from_hparams(
            source=source,
            run_opts={"device": device},
        )

        if freeze_encoder:
            self.freeze()
            print(f"  🔒 ECAPA-TDNN encoder: FROZEN")
        else:
            print(f"  🔓 ECAPA-TDNN encoder: UNFROZEN (full fine-tune)")

    def forward(
        self,
        waveforms: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            waveforms: (batch, 1, T)

        Returns:
            hidden_states: (batch, 1, 192) — seq_len=1 because ECAPA has
                           internal pooling that produces utterance-level embeddings.
                           We add a dummy seq dim for compatibility.
            lengths: None
        """
        # SpeechBrain expects (batch, T) — squeeze channel dim
        wav = waveforms.squeeze(1)  # (batch, T)

        # Disable AMP autocast for frozen ECAPA encoder.
        # ECAPA-TDNN's pretrained weights are float32 and SpeechBrain's
        # Conv1d layers are not autocast-compatible, causing:
        #   "Input type (float) and bias type (c10::Half) should be the same"
        with torch.amp.autocast(device_type="cuda", enabled=False):
            # encode_batch returns (batch, 192) utterance-level embeddings
            embeddings = self.classifier.encode_batch(wav)  # (batch, 192)
        # If output has extra dim, squeeze it
        if embeddings.ndim == 3:
            embeddings = embeddings.squeeze(1)  # (batch, 192)

        # Add dummy sequence dimension for pooling compatibility
        # IdentityPooling will squeeze this back
        hidden_states = embeddings.unsqueeze(1)  # (batch, 1, 192)

        return hidden_states, None

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def freeze(self) -> None:
        """Freeze all ECAPA-TDNN parameters."""
        for param in self.classifier.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze for fine-tuning."""
        for param in self.classifier.parameters():
            param.requires_grad = True
        print("  🔓 ECAPA-TDNN encoder UNFROZEN.")


# ═══════════════════════════════════════════════════════════
#  Encoder Factory
# ═══════════════════════════════════════════════════════════

def create_encoder(config: dict) -> BaseEncoder:
    """
    Build an encoder from config.

    Reads:
        model.encoder_type  → "wavlm" | "ecapa" | "hubert"
        model.encoder_config.<type> → encoder-specific kwargs

    Args:
        config: Full project config dict

    Returns:
        BaseEncoder instance
    """
    model_cfg = config["model"]
    encoder_type = model_cfg.get("encoder_type", "wavlm").lower().strip()

    # ── Resolve encoder config (backward-compat) ──
    if "encoder_config" in model_cfg:
        enc_cfg = model_cfg["encoder_config"].get(encoder_type, {})
        # Merge old flat keys as fallback
        if "base_model" not in enc_cfg and "base_model" in model_cfg:
            enc_cfg = {**enc_cfg, "base_model": model_cfg["base_model"]}
        if "freeze_feature_extractor" not in enc_cfg and "freeze_feature_extractor" in model_cfg:
            enc_cfg = {**enc_cfg, "freeze_feature_extractor": model_cfg["freeze_feature_extractor"]}
    else:
        # Old flat config format
        enc_cfg = {
            "base_model": model_cfg.get("base_model", "microsoft/wavlm-base-plus"),
            "freeze_feature_extractor": model_cfg.get("freeze_feature_extractor", True),
        }

    if encoder_type == "wavlm":
        return WavLMEncoder(
            base_model=enc_cfg.get("base_model", "microsoft/wavlm-base-plus"),
            freeze_feature_extractor=enc_cfg.get("freeze_feature_extractor", True),
        )
    elif encoder_type == "ecapa":
        return ECAPAEncoder(
            source=enc_cfg.get("source", "speechbrain/spkrec-ecapa-voxceleb"),
            freeze_encoder=enc_cfg.get("freeze_encoder", True),
        )
    elif encoder_type == "hubert":
        return HuBERTEncoder(
            base_model=enc_cfg.get("base_model", "facebook/hubert-large-ls960-ft"),
            freeze_feature_extractor=enc_cfg.get("freeze_feature_extractor", True),
        )
    else:
        raise ValueError(
            f"Unknown encoder_type: '{encoder_type}'. "
            f"Expected 'wavlm', 'ecapa', or 'hubert'."
        )


# ═══════════════════════════════════════════════════════════
#  Smoke Test
# ═══════════════════════════════════════════════════════════

def _smoke_test():
    """Quick test of WavLMEncoder (requires internet for first model load)."""
    print("=" * 50)
    print("  Encoder Smoke Test")
    print("=" * 50)

    # Test with minimal config
    config = {
        "model": {
            "encoder_type": "wavlm",
            "encoder_config": {
                "wavlm": {
                    "base_model": "microsoft/wavlm-base-plus",
                    "freeze_feature_extractor": True,
                }
            }
        }
    }

    print("  Creating WavLMEncoder (will download model if not cached)...")
    encoder = create_encoder(config)
    print(f"  Encoder type: {type(encoder).__name__}")
    print(f"  Output dim:   {encoder.output_dim}")

    # Forward pass
    waveforms = torch.randn(2, 1, 80000)  # 2 × 5s @ 16kHz
    hidden, lengths = encoder(waveforms)
    print(f"  Input shape:  {waveforms.shape}")
    print(f"  Output shape: {hidden.shape}")
    print(f"  Lengths:      {lengths}")

    assert hidden.shape[0] == 2
    assert hidden.shape[2] == encoder.output_dim
    assert hidden.shape[1] > 0  # sequence length > 0

    print()
    print("  ALL ENCODER TESTS PASSED ✅")


if __name__ == "__main__":
    _smoke_test()
