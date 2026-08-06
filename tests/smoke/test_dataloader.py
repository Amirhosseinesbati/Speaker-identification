"""Smoke test for data pipeline: augmentation, dataset, filtering."""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import pytest

from src.data_pipeline import (
    AudioAugmentation,
    SpeakerDataset,
    create_class_mapping,
)


class TestAudioAugmentation:
    def test_augmentation_preserves_shape(self):
        aug = AudioAugmentation(sample_rate=16000)
        waveform = torch.randn(1, 80000)  # 5 seconds
        out = aug(waveform)
        assert out.shape == waveform.shape
        assert isinstance(out, torch.Tensor)

    def test_augmentation_does_not_crash_on_short_audio(self):
        aug = AudioAugmentation(sample_rate=16000)
        waveform = torch.randn(1, 8000)  # 0.5 seconds
        out = aug(waveform)
        assert out.shape == waveform.shape


class TestSpeakerDataset:
    def _make_dummy_df(self, n_samples=10):
        return pd.DataFrame({
            "speaker_id": ["spk_1"] * 5 + ["spk_2"] * 3 + ["unknown"] * 2,
            "audio_file": [f"dummy_{i}.wav" for i in range(10)],
            "label": [1] * 5 + [2] * 3 + [0] * 2,
        })

    def _make_dummy_audio(self, seconds=3.0):
        """Create a fake WAV file with given duration."""
        import wave
        import struct
        sr = 16000
        n_samples = int(sr * seconds)
        samples = (np.random.randn(n_samples) * 32767 * 0.1).astype(np.int16)

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        with wave.open(tmp.name, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(samples.tobytes())
        return tmp.name

    def test_dataset_creates_without_error(self):
        df = self._make_dummy_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy audio files
            for i in range(10):
                path = Path(tmpdir) / f"dummy_{i}.wav"
                import wave
                import struct
                sr = 16000
                samples = (np.random.randn(sr * 3) * 32767 * 0.1).astype(np.int16)
                with wave.open(str(path), "w") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sr)
                    wf.writeframes(samples.tobytes())

            ds = SpeakerDataset(df, tmpdir, duration_seconds=3.0, augment=False)
            assert len(ds) == 10

    def test_getitem_shape(self):
        df = self._make_dummy_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            sr = 16000
            duration = 3
            for i in range(10):
                path = Path(tmpdir) / f"dummy_{i}.wav"
                import wave
                import struct
                samples = (np.random.randn(sr * duration) * 32767 * 0.1).astype(np.int16)
                with wave.open(str(path), "w") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sr)
                    wf.writeframes(samples.tobytes())

            ds = SpeakerDataset(df, tmpdir, duration_seconds=duration, augment=False)
            waveform, label = ds[0]
            assert waveform.shape == (1, sr * duration)
            assert isinstance(label, torch.Tensor)

    def test_duration_from_config(self):
        """Test that 5s config produces 80000-sample tensors."""
        df = self._make_dummy_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            sr = 16000
            for i in range(10):
                path = Path(tmpdir) / f"dummy_{i}.wav"
                import wave, struct
                samples = (np.random.randn(sr * 6) * 32767 * 0.1).astype(np.int16)
                with wave.open(str(path), "w") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sr)
                    wf.writeframes(samples.tobytes())

            ds = SpeakerDataset(df, tmpdir, duration_seconds=5.0, augment=False)
            waveform, label = ds[0]
            assert waveform.shape == (1, 80000), f"Expected (1,80000), got {waveform.shape}"


class TestClassMapping:
    def test_mapping(self):
        df = pd.DataFrame({
            "speaker_id": ["spk_a", "spk_b", "unknown", "spk_a"],
            "audio_file": ["a.wav", "b.wav", "c.wav", "d.wav"],
        })
        mapping = create_class_mapping(df)
        assert mapping["unknown"] == 0
        assert mapping["spk_a"] == 1
        assert mapping["spk_b"] == 2
