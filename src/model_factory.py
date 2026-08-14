"""
Model Factory — builds TwoHeadedSpeakerModel from config.

Usage:
    from src.model_factory import create_model_from_config

    config = yaml.safe_load(open("configs/default_config.yaml"))
    model = create_model_from_config(config, num_known_speakers=446)
"""

from typing import Optional

import torch.nn as nn

from src.encoders import BaseEncoder, create_encoder
from src.pooling import create_pooling
from src.heads import create_ood_head, create_speaker_head
from src.model import TwoHeadedSpeakerModel


def create_model_from_config(
    config: dict,
    num_known_speakers: int = 446,
    encoder_override: Optional[BaseEncoder] = None,
) -> TwoHeadedSpeakerModel:
    """
    Build a complete TwoHeadedSpeakerModel from a config dict.

    Components are selected based on:
        model.encoder_type    → encoder
        model.pooling_type    → pooling
        model.speaker_head_type → speaker head
        model.ood_head_config → OOD head

    Args:
        config:               Full project config dict
        num_known_speakers:   Number of known speaker classes (excl. unknown)
        encoder_override:     Optional pre-built encoder (for ensemble/checkpoint loading)

    Returns:
        TwoHeadedSpeakerModel ready for training or inference
    """
    model_cfg = config["model"]

    # ── 1. Encoder ──
    if encoder_override is not None:
        encoder = encoder_override
        encoder_name = getattr(encoder, 'base_model_name', 'custom')
    else:
        encoder = create_encoder(config)
        encoder_name = model_cfg.get("encoder_type", "unknown")

    # ── 2. Pooling ──
    # Per-encoder pooling wins over the global default (the ensemble mixes
    # identity-pooled encoders with statistical-pooled WavLM).
    pooling_type = model_cfg.get("pooling_type", "statistical")
    enc_type_key = model_cfg.get("encoder_type", "unknown").lower().strip()
    enc_cfg = model_cfg.get("encoder_config", {}).get(enc_type_key, {})
    pooling_type = enc_cfg.get("pooling_type", pooling_type)
    pooling = create_pooling(pooling_type, encoder.output_dim)
    pooled_dim = encoder.output_dim * pooling.output_multiplier


    # ── 3. OOD Head ──
    ood_head = create_ood_head(config, pooled_dim)

    # ── 4. Speaker Head ──
    speaker_head = create_speaker_head(config, pooled_dim, num_known_speakers)
    head_type = model_cfg.get("speaker_head_type", "linear")

    # ── 5. Assemble ──
    model = TwoHeadedSpeakerModel(
        encoder=encoder,
        pooling=pooling,
        speaker_head=speaker_head,
        ood_head=ood_head,
        num_known_speakers=num_known_speakers,
        encoder_name=encoder_name,
    )

    return model


# ═══════════════════════════════════════════════════════════
#  Smoke Test
