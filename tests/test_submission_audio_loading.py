from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from submission.inference import _load_waveform


def test_corrupt_riff_does_not_fall_through_to_mpeg_decoder(
    monkeypatch, tmp_path: Path
) -> None:
    audio = tmp_path / "corrupt.mp3"
    audio.write_bytes(b"RIFF" + b"\x00" * 4 + b"WAVE" + b"broken")

    def fail_soundfile(*args, **kwargs):
        raise RuntimeError("corrupt RIFF")

    def forbidden_mpeg(*args, **kwargs):
        raise AssertionError("corrupt RIFF must not reach mpg123/librosa")

    monkeypatch.setattr(sf, "read", fail_soundfile)
    monkeypatch.setattr(librosa, "load", forbidden_mpeg)
    assert _load_waveform(audio, 16_000) is None


def test_real_mpeg_signature_keeps_decoder_fallback(monkeypatch, tmp_path: Path) -> None:
    audio = tmp_path / "real.mp3"
    audio.write_bytes(b"ID3" + b"\x00" * 32)

    def fail_soundfile(*args, **kwargs):
        raise RuntimeError("use MPEG decoder")

    monkeypatch.setattr(sf, "read", fail_soundfile)
    monkeypatch.setattr(
        librosa,
        "load",
        lambda *args, **kwargs: (np.zeros(16_000, dtype=np.float32), 16_000),
    )
    waveform = _load_waveform(audio, 16_000)
    assert waveform is not None
    assert tuple(waveform.shape) == (1, 16_000)


def test_pcm_bytes_that_resemble_frame_sync_do_not_reach_mpeg_decoder(
    monkeypatch, tmp_path: Path
) -> None:
    audio = tmp_path / "headerless-pcm.mp3"
    audio.write_bytes(b"\xff\xe3\xff\xe3" + b"\x00" * 32)

    monkeypatch.setattr(
        sf, "read", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError())
    )
    monkeypatch.setattr(
        librosa,
        "load",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("headerless PCM must not reach mpg123/librosa")
        ),
    )
    assert _load_waveform(audio, 16_000) is None
