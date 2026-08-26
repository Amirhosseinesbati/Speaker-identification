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


def resolve_speaker_head_layout(
    config: dict,
    num_metric_classes: int,
) -> tuple[int, int, str]:
    """Resolve speaker-head width independently from metric pseudo-labels.

    ``metric`` preserves the existing 446+k ArcFace head and collapses its
    pseudo columns at inference. ``known`` builds a strict 446-way head while
    retaining the k pseudo labels for OOD BCE and an optional metric auxiliary
    loss. The latter is the known-first architecture used by the new gate.

    Returns ``(speaker_head_classes, output_pseudo_classes, scope)``.
    """
    model_cfg = config.get("model", {}) or {}
    scope = str(model_cfg.get("speaker_target_scope", "metric")).lower().strip()
    if scope not in {"metric", "known"}:
        raise ValueError(
            "model.speaker_target_scope must be 'metric' or 'known', "
            f"got {scope!r}"
        )

    competition_known = int(model_cfg.get("competition_num_known", 446))
    if num_metric_classes < competition_known:
        raise ValueError(
            f"Metric class map has {num_metric_classes} classes, fewer than "
            f"competition_num_known={competition_known}."
        )
    if scope == "known":
        return competition_known, 0, scope

    output_pseudo = max(0, int(num_metric_classes) - competition_known)
    return int(num_metric_classes), output_pseudo, scope
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
    # The class map may carry 554 pseudo identities even when the primary
    # speaker head is intentionally restricted to the 446 competition-known
    # identities.  Keep those two widths independent.
    metric_num_classes = int(num_known_speakers)
    speaker_head_classes, output_pseudo_classes, speaker_target_scope = (
        resolve_speaker_head_layout(config, metric_num_classes)
    )
    speaker_head = create_speaker_head(config, pooled_dim, speaker_head_classes)
    head_type = model_cfg.get("speaker_head_type", "linear")

    # ── 5. Assemble ──
    # Closed-set 1000-class experiment (default 0 = legacy 447-way). The
    # speaker-head width is the SOURCE OF TRUTH for how many pseudo-identity
    # cluster columns exist: whatever sits beyond the competition-known
    # speakers is a cluster column. Deriving it here (instead of trusting the
    # raw config value) keeps the collapse math exact even when a few
    # clusters' files are dropped by the corruption/duplicate filter (the
    # class map shrinks) or the map was rebuilt at a different k — the model
    # always outputs the fixed 447-way competition vector.
    num_unknown_clusters = output_pseudo_classes

    model = TwoHeadedSpeakerModel(
        encoder=encoder,
        pooling=pooling,
        speaker_head=speaker_head,
        ood_head=ood_head,
        num_known_speakers=speaker_head_classes,
        encoder_name=encoder_name,
        num_unknown_clusters=num_unknown_clusters,
    )
    # Non-parameter metadata: useful to checkpoints, diagnostics and tests;
    # state-dict compatibility is unchanged.
    model.metric_num_classes = metric_num_classes
    model.speaker_target_scope = speaker_target_scope

    return model


# ═══════════════════════════════════════════════════════════
#  Smoke Test
