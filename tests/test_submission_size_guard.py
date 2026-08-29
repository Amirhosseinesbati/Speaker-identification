from pathlib import Path

import pytest

from scripts.build_submission import assert_submission_zip_size
from scripts.verify_submission import submission_zip_size_ok


def _write_bytes(path: Path, size: int) -> Path:
    path.write_bytes(b"x" * size)
    return path


def test_submission_zip_size_guard_accepts_exact_limit(tmp_path: Path) -> None:
    archive = _write_bytes(tmp_path / "submission.zip", 16)
    assert assert_submission_zip_size(archive, maximum_bytes=16) == 16
    assert submission_zip_size_ok(archive, maximum_bytes=16)


def test_submission_zip_size_guard_rejects_one_byte_over(tmp_path: Path) -> None:
    archive = _write_bytes(tmp_path / "submission.zip", 17)
    with pytest.raises(RuntimeError, match="exceeds the 1 GB limit"):
        assert_submission_zip_size(archive, maximum_bytes=16)
    assert not submission_zip_size_ok(archive, maximum_bytes=16)
