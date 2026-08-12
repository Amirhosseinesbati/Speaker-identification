"""
Tests for the unified MP3→WAV preprocessing (`src/audio_preprocessing.py`).

Verify that local conversion (`scripts/convert_mp3_to_wav.py`) and the server
pipeline step (`src/pipelines/steps.py::convert_audio`) — which now share the
same `convert_all` code path — produce deterministic, correctly-formatted
(16 kHz mono PCM-16) WAV files and a correct labels CSV.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio_preprocessing import TARGET_SR, convert_all, convert_one

RAW_DIR = PROJECT_ROOT / "data" / "raw"

pytestmark = pytest.mark.skipif(
    not list(RAW_DIR.glob("*.mp3")),
    reason="data/raw has no MP3 files (data not present)",
)


@pytest.fixture()
def sample_mp3s() -> list[Path]:
    """Up to 3 real MP3s from data/raw (module is skipped if data is absent)."""
    return sorted(RAW_DIR.glob("*.mp3"))[:3]


@pytest.fixture()
def raw_fixture(tmp_path, sample_mp3s) -> tuple[Path, list[Path]]:
    """A tmp `data/raw`-shaped dir with labels.csv referencing copied MP3s."""
    raw = tmp_path / "raw"
    raw.mkdir()
    for src in sample_mp3s:
        (raw / src.name).write_bytes(src.read_bytes())
    rows = "\n".join(
        f"speaker_{i % 3},{mp3.name}" for i, mp3 in enumerate(sample_mp3s)
    )
    (raw / "labels.csv").write_text(
        "speaker_id,audio_file\n" + rows + "\n", encoding="utf-8"
    )
    return raw, sample_mp3s


def test_convert_one_format(sample_mp3s, tmp_path):
    import soundfile as sf

    dst = tmp_path / "out.wav"
    result = convert_one(sample_mp3s[0], dst)
    assert result["status"] == "ok"
    info = sf.info(str(dst))
    assert info.samplerate == TARGET_SR
    assert info.channels == 1
    assert info.subtype == "PCM_16"


def test_convert_one_deterministic(sample_mp3s, tmp_path):
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    assert convert_one(sample_mp3s[0], a)["status"] == "ok"
    assert convert_one(sample_mp3s[0], b)["status"] == "ok"
    assert a.read_bytes() == b.read_bytes()


def test_convert_all_skips_existing(raw_fixture, tmp_path):
    raw, samples = raw_fixture
    wav_dir = tmp_path / "audio_wav"
    s1 = convert_all(raw, wav_dir, progress=False)
    assert s1["converted"] == len(samples)

    first_wav = sorted(wav_dir.glob("*.wav"))[0]
    before = first_wav.read_bytes()

    s2 = convert_all(raw, wav_dir, progress=False)
    assert s2["converted"] == 0
    assert s2["skipped"] == len(samples)
    assert first_wav.read_bytes() == before  # untouched


def test_convert_all_force_reconverts(raw_fixture, tmp_path):
    raw, samples = raw_fixture
    wav_dir = tmp_path / "audio_wav"
    convert_all(raw, wav_dir, progress=False)
    s = convert_all(raw, wav_dir, force=True, progress=False)
    assert s["converted"] == len(samples)


def test_convert_all_labels_point_to_wav(raw_fixture, tmp_path):
    import pandas as pd

    raw, _ = raw_fixture
    wav_dir = tmp_path / "audio_wav"
    s = convert_all(raw, wav_dir, progress=False)

    labels = pd.read_csv(s["labels_path"])
    assert len(labels) == len(pd.read_csv(raw / "labels.csv"))
    for name in labels["audio_file"]:
        assert name.endswith(".wav")
        assert (wav_dir / name).exists()


def test_convert_all_two_roots_identical(raw_fixture, tmp_path):
    """Simulated local + server runs must produce byte-identical WAVs."""
    raw, _ = raw_fixture
    convert_all(raw, tmp_path / "wav_a", progress=False)
    convert_all(raw, tmp_path / "wav_b", progress=False)
    a_files = sorted((tmp_path / "wav_a").glob("*.wav"))
    b_files = sorted((tmp_path / "wav_b").glob("*.wav"))
    assert len(a_files) == len(b_files)
    for a, b in zip(a_files, b_files):
        assert a.read_bytes() == b.read_bytes()


def test_convert_all_missing_labels_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        convert_all(tmp_path / "raw", tmp_path / "audio_wav")


def test_zenml_convert_audio_step_uses_same_code(tmp_path, monkeypatch):
    """The server step drives the same convert_all code path as the local script."""
    import yaml

    from src.pipelines import steps as steps_module
    from src.pipelines.steps import convert_audio

    raw = tmp_path / "raw"
    wav_dir = tmp_path / "processed" / "audio_wav"
    raw.mkdir(parents=True)
    mp3s = sorted(RAW_DIR.glob("*.mp3"))[:2]
    for src in mp3s:
        (raw / src.name).write_bytes(src.read_bytes())
    (raw / "labels.csv").write_text(
        "speaker_id,audio_file\n"
        + "\n".join(f"sp{i},{mp3.name}" for i, mp3 in enumerate(mp3s))
        + "\n",
        encoding="utf-8",
    )

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "data": {"raw_dir": str(raw), "audio_dir": str(wav_dir)},
                "model": {"encoder_type": "test"},
            }
        ),
        encoding="utf-8",
    )

    # Stub MLflow tracking so the step runs without a real tracker.
    class StubTracker:
        def start_run(self, run_name=None):
            pass

        def log_params(self, params):
            pass

        def log_metrics(self, metrics, step=None):
            pass

        def log_artifact(self, path, artifact_path=None):
            pass

        def log_code_snapshot(self):
            pass

        def log_config_snapshot(self):
            pass

    monkeypatch.setattr(steps_module, "get_tracker", lambda config: StubTracker())

    result = convert_audio(config_path=str(cfg_path))

    assert result["data"]["audio_dir"] == str(wav_dir)
    assert result["data"]["labels_path"] == str(wav_dir.parent / "audio_wav_labels.csv")
    assert len(list(wav_dir.glob("*.wav"))) == len(mp3s)
    assert (wav_dir.parent / "audio_wav_labels.csv").exists()
