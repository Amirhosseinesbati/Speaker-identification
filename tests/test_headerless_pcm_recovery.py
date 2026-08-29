import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_headerless_pcm_recovery.py"
SPEC = importlib.util.spec_from_file_location("evaluate_headerless_pcm_recovery", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_inspect_pcm16_stereo_accepts_duplicate_channels(tmp_path: Path) -> None:
    path = tmp_path / "raw.mp3"
    mono = (np.sin(np.linspace(0, 20, 16_000)) * 10_000).astype("<i2")
    path.write_bytes(np.column_stack([mono, mono]).tobytes())
    result = MODULE.inspect_pcm16_stereo(path)
    assert result["eligible"]
    assert result["channel_equal_fraction"] == 1.0
    assert result["duration_seconds"] == 1.0


def test_inspect_pcm16_stereo_rejects_container_and_misalignment(tmp_path: Path) -> None:
    riff = tmp_path / "riff.mp3"
    riff.write_bytes(b"RIFF" + b"\x00" * 12)
    assert not MODULE.inspect_pcm16_stereo(riff)["eligible"]

    broken = tmp_path / "broken.mp3"
    broken.write_bytes(b"abc")
    assert MODULE.inspect_pcm16_stereo(broken)["reason"] == "not_stereo_pcm16_aligned"


def test_metrics_show_recovery_tradeoff() -> None:
    truth = ["known-a", "unknown", "known-b"]
    baseline = MODULE.evaluate(truth, ["unknown"] * 3)
    recovered = MODULE.evaluate(truth, ["known-a", "known-x", "known-b"])
    assert baseline["known_accuracy"] == 0.0
    assert baseline["unknown_accuracy"] == 1.0
    assert recovered["known_accuracy"] == 1.0
    assert recovered["unknown_accuracy"] == 0.0
