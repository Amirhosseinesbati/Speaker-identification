from __future__ import annotations

from src.deploy.deploy_app import _encoder_save_config


def test_wavlm_save_preserves_layer_adapter_contract() -> None:
    old = {
        "frozen_backbone_eval": True,
        "layer_aggregation": "layer_adapter",
        "layer_adapter_dim": 512,
        "layer_adapter_activation": "relu",
        "layer_adapter_layer_norm": True,
        "layer_adapter_init_std": 1.0e-3,
        "layer_adapter_tune_backbone_layer_norms": True,
        "unfreeze_last_n_blocks": 7,
    }

    updated = _encoder_save_config(
        encoder_type="wavlm",
        old_enc=old,
        ft_mode="Frozen",
        unfreeze_n=0,
        wavlm_variant="microsoft/wavlm-base-plus",
    )
    resolved = {**old, **updated}

    assert resolved["base_model"] == "microsoft/wavlm-base-plus"
    assert resolved["local_path"] == "weights/wavlm_base_plus"
    assert resolved["freeze_encoder"] is True
    assert resolved["freeze_feature_extractor"] is True
    assert resolved["frozen_backbone_eval"] is True
    assert resolved["layer_aggregation"] == "layer_adapter"
    assert resolved["layer_adapter_dim"] == 512
    assert resolved["layer_adapter_tune_backbone_layer_norms"] is True
    assert "unfreeze_last_n_blocks" not in resolved


def test_wavlm_full_mode_explicitly_unfreezes_backbone() -> None:
    old = {"freeze_encoder": True, "freeze_feature_extractor": True}

    updated = _encoder_save_config(
        encoder_type="wavlm",
        old_enc=old,
        ft_mode="Full",
        unfreeze_n=0,
        wavlm_variant="microsoft/wavlm-base",
    )
    resolved = {**old, **updated}

    assert resolved["freeze_encoder"] is False
    assert resolved["freeze_feature_extractor"] is False
