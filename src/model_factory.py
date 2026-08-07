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
    num_known_speakers: int = 447,
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
    pooling_type = model_cfg.get("pooling_type", "statistical")
    pooling = create_pooling(pooling_type, encoder.output_dim)
    pooled_dim = encoder.output_dim * pooling.output_multiplier

    print(f"  📊 Encoder: {type(encoder).__name__} ({encoder.output_dim}d)")
    print(f"     Pooling: {pooling_type} ({pooled_dim}d)")

    # ── 3. OOD Head ──
    ood_head = create_ood_head(config, pooled_dim)
    print(f"     OOD Head: {type(ood_head).__name__} ({pooled_dim} → 1)")

    # ── 4. Speaker Head ──
    speaker_head = create_speaker_head(config, pooled_dim, num_known_speakers)
    head_type = model_cfg.get("speaker_head_type", "linear")
    print(f"     Speaker Head: {type(speaker_head).__name__} ({pooled_dim} → {num_known_speakers})")

    # ── 5. Assemble ──
    model = TwoHeadedSpeakerModel(
        encoder=encoder,
        pooling=pooling,
        speaker_head=speaker_head,
        ood_head=ood_head,
        num_known_speakers=num_known_speakers,
        encoder_name=encoder_name,
    )

    model.print_summary()
    return model


# ═══════════════════════════════════════════════════════════
#  Smoke Test
# ═══════════════════════════════════════════════════════════

def _smoke_test():
    import torch

    print("=" * 55)
    print("  Model Factory Smoke Test")
    print("=" * 55)

    config = {
        "hardware": {
            "mode": "local",
            "profiles": {
                "local": {"device": "cpu", "batch_size": 4, "num_workers": 0, "mixed_precision": False}
            },
        },
        "audio": {"sample_rate": 16000, "duration_seconds": 5.0},
        "model": {
            "encoder_type": "wavlm",
            "encoder_config": {
                "wavlm": {
                    "base_model": "microsoft/wavlm-base-plus",
                    "freeze_feature_extractor": True,
                },
            },
            "pooling_type": "attentive",
            "speaker_head_type": "linear",
            "speaker_head_config": {
                "arcface": {"embedding_dim": 192, "margin": 0.3, "scale": 15.0}
            },
            "ood_head_config": {"hidden_dim": 256},
            "fusion": {"ensemble_method": "none"},
        },
    }

    num_known = 10
    model = create_model_from_config(config, num_known_speakers=num_known)

    # Forward pass
    waveforms = torch.randn(2, 1, 80000)
    ood, spk = model(waveforms)
    assert ood.shape == (2, 1)
    assert spk.shape == (2, num_known)

    # Predict proba
    probs = model.predict_proba(waveforms)
    assert probs.shape == (2, num_known + 1)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)

    print(f"\n  ALL FACTORY TESTS PASSED ✅")


if __name__ == "__main__":
    _smoke_test()
