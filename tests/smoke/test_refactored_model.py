"""Smoke tests for refactored modular architecture (Phase 2.1).

Tests: encoders, heads, model factory, backward compat.
CPU-only, no downloads needed for heads/pooling.
"""

import pytest
import torch

from src.encoders import WavLMEncoder, create_encoder
from src.heads import (
    OODHead, LinearSpeakerHead, ArcFaceHead,
    create_ood_head, create_speaker_head,
)
from src.model import TwoHeadedSpeakerModel, TwoHeadedWavLM
from src.model_factory import create_model_from_config
from src.pooling import StatisticalPooling, AttentiveStatisticalPooling


# ═══════════════════════════════════════════════════════
#  Helper
# ═══════════════════════════════════════════════════════

def _make_minimal_config(encoder_type="wavlm", head_type="linear", pooling="attentive"):
    return {
        "hardware": {
            "mode": "local",
            "profiles": {"local": {"device": "cpu", "batch_size": 2, "num_workers": 0, "mixed_precision": False}},
        },
        "audio": {"sample_rate": 16000, "duration_seconds": 5.0},
        "model": {
            "encoder_type": encoder_type,
            "encoder_config": {
                "wavlm": {"base_model": "microsoft/wavlm-base-plus", "freeze_feature_extractor": True},
                "ecapa": {"source": "speechbrain/spkrec-ecapa-voxceleb", "freeze_encoder": True},
                "hubert": {"base_model": "facebook/hubert-large-ls960-ft", "freeze_feature_extractor": True},
            },
            "pooling_type": pooling,
            "speaker_head_type": head_type,
            "speaker_head_config": {"arcface": {"embedding_dim": 192, "margin": 0.3, "scale": 15.0}},
            "ood_head_config": {"hidden_dim": 256},
            "fusion": {"ensemble_method": "none"},
        },
    }


# ═══════════════════════════════════════════════════════
#  Encoder Tests
# ═══════════════════════════════════════════════════════

class TestEncoderFactory:
    def test_create_wavlm_encoder(self):
        config = _make_minimal_config("wavlm")
        encoder = create_encoder(config)
        assert isinstance(encoder, WavLMEncoder)
        assert encoder.output_dim == 768

    def test_unknown_encoder_raises(self):
        config = _make_minimal_config("wavlm")
        config["model"]["encoder_type"] = "resnet"  # not supported
        with pytest.raises(ValueError):
            create_encoder(config)


# ═══════════════════════════════════════════════════════
#  Head Tests
# ═══════════════════════════════════════════════════════

class TestOODHead:
    def test_output_shape(self):
        head = OODHead(512, hidden_dim=128)
        x = torch.randn(4, 512)
        out = head(x)
        assert out.shape == (4, 1)

    def test_no_hidden(self):
        head = OODHead(256, hidden_dim=0)
        x = torch.randn(4, 256)
        out = head(x)
        assert out.shape == (4, 1)


class TestLinearHead:
    def test_output_shape(self):
        head = LinearSpeakerHead(512, 20)
        x = torch.randn(4, 512)
        out = head(x)
        assert out.shape == (4, 20)


class TestArcFaceHead:
    def test_train_mode(self):
        head = ArcFaceHead(512, 10, embedding_dim=128)
        x = torch.randn(4, 512)
        labels = torch.randint(0, 10, (4,))
        out = head(x, labels)
        assert out.shape == (4, 10)

    def test_inference_mode(self):
        head = ArcFaceHead(512, 10, embedding_dim=128)
        x = torch.randn(4, 512)
        out = head(x, labels=None)
        assert out.shape == (4, 10)

    def test_head_factories(self):
        config = _make_minimal_config(head_type="linear")
        ood = create_ood_head(config, 512)
        spk = create_speaker_head(config, 512, 10)
        assert isinstance(ood, OODHead)
        assert isinstance(spk, LinearSpeakerHead)

        config["model"]["speaker_head_type"] = "arcface"
        spk_arc = create_speaker_head(config, 512, 10)
        assert isinstance(spk_arc, ArcFaceHead)


# ═══════════════════════════════════════════════════════
#  Modular Model Tests
# ═══════════════════════════════════════════════════════

class TestModularModel:
    def test_build_from_components(self):
        """Manually assemble a model from components."""
        config = _make_minimal_config()
        encoder = create_encoder(config)
        pooling = AttentiveStatisticalPooling(encoder.output_dim)
        pooled_dim = encoder.output_dim * 2
        ood_head = OODHead(pooled_dim, 128)
        spk_head = LinearSpeakerHead(pooled_dim, 10)

        model = TwoHeadedSpeakerModel(
            encoder=encoder, pooling=pooling,
            speaker_head=spk_head, ood_head=ood_head,
            num_known_speakers=10, encoder_name="test",
        )
        assert model.num_known_speakers == 10

    def test_predict_proba_sums_to_one(self):
        config = _make_minimal_config()
        model = create_model_from_config(config, num_known_speakers=10)
        waveforms = torch.randn(2, 1, 80000)
        probs = model.predict_proba(waveforms)
        assert probs.shape == (2, 11)
        assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)

    def test_trainable_params_positive(self):
        config = _make_minimal_config()
        model = create_model_from_config(config, num_known_speakers=10)
        assert model.get_trainable_params() > 0

    def test_backward_compat_two_headed_wavlm(self):
        """TwoHeadedWavLM should work exactly like before."""
        config = _make_minimal_config()
        model = TwoHeadedWavLM(config, num_known_speakers=10)
        assert hasattr(model, 'wavlm')
        assert hasattr(model, '_freeze_feature_extractor')
        assert hasattr(model, 'unfreeze_feature_extractor')

    def test_forward_with_labels_for_arcface(self):
        config = _make_minimal_config(head_type="arcface")
        model = create_model_from_config(config, num_known_speakers=10)
        waveforms = torch.randn(2, 1, 80000)
        labels = torch.tensor([0, 5])
        ood, spk = model(waveforms, labels=labels)
        assert ood.shape == (2, 1)
        assert spk.shape == (2, 10)

    def test_forward_without_labels(self):
        config = _make_minimal_config(head_type="arcface")
        model = create_model_from_config(config, num_known_speakers=10)
        waveforms = torch.randn(2, 1, 80000)
        ood, spk = model(waveforms)  # no labels
        assert ood.shape == (2, 1)
        assert spk.shape == (2, 10)
