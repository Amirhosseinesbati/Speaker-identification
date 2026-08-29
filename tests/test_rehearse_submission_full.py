import csv
import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "rehearse_submission_full.py"
SPEC = importlib.util.spec_from_file_location("rehearse_submission_full", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _write(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)


def test_validate_predictions_requires_exact_coverage(tmp_path: Path) -> None:
    output = tmp_path / "predictions.csv"
    speaker = "12345678-1234-1234-1234-123456789abc"
    _write(
        output,
        [
            ["audio_file", "speaker_id"],
            ["a.mp3", "unknown"],
            ["b.mp3", speaker],
        ],
    )
    result = MODULE.validate_predictions(output, ["a.mp3", "b.mp3"])
    assert result["valid"]
    assert result["unknown_predictions"] == 1

    result = MODULE.validate_predictions(output, ["a.mp3", "c.mp3"])
    assert not result["valid"]
    assert result["missing"] == ["c.mp3"]
    assert result["unexpected"] == ["b.mp3"]


def test_validate_predictions_rejects_duplicates_and_bad_labels(tmp_path: Path) -> None:
    output = tmp_path / "predictions.csv"
    _write(
        output,
        [
            ["audio_file", "speaker_id"],
            ["a.mp3", "not-a-speaker"],
            ["a.mp3", "unknown"],
        ],
    )
    result = MODULE.validate_predictions(output, ["a.mp3", "b.mp3"])
    assert not result["valid"]
    assert result["duplicates"] == ["a.mp3"]
    assert result["invalid_labels"] == ["not-a-speaker"]
