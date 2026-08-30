from pathlib import Path
import zipfile

import pytest

from scripts.build_submission import (
    _rm_artifacts,
    assert_no_sensitive_submission_paths,
    assert_submission_zip_size,
    sensitive_submission_paths,
)
from scripts.verify_submission import sensitive_zip_entry_names, submission_zip_size_ok


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


def test_submission_secret_guard_rejects_model_cache_session(
    tmp_path: Path,
) -> None:
    credential = tmp_path / "weights" / "campp" / "credentials" / "session"
    credential.parent.mkdir(parents=True)
    credential.write_text("do-not-package", encoding="utf-8")

    assert sensitive_submission_paths(tmp_path) == [
        Path("weights/campp/credentials/session")
    ]
    with pytest.raises(RuntimeError, match="sensitive credential/cache paths"):
        assert_no_sensitive_submission_paths(tmp_path)


def test_submission_cleanup_removes_auth_and_lock_cache_only(
    tmp_path: Path,
) -> None:
    credential = tmp_path / "weights" / "campp" / "credentials" / "session"
    lock_file = tmp_path / "weights" / "campp" / ".lock" / "model.lock"
    source_file = tmp_path / "src" / "session.py"
    for path in (credential, lock_file, source_file):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")

    _rm_artifacts(tmp_path)

    assert not credential.exists()
    assert not lock_file.exists()
    assert source_file.exists()
    assert_no_sensitive_submission_paths(tmp_path)


def test_zip_secret_guard_reads_names_without_reading_payload(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "submission.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("submission.py", "print('ok')")
        output.writestr("weights/campp/credentials/session", "do-not-read")

    assert sensitive_zip_entry_names(archive) == [
        "weights/campp/credentials/session"
    ]
