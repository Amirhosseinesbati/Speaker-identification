import hashlib
import json
from pathlib import Path

import yaml

from scripts.audit_dvc_integrity import audit_dvc_directory


def _md5(payload: bytes) -> str:
    return hashlib.md5(payload).hexdigest()


def _fixture(tmp_path: Path, workspace_payload: bytes, cache_payload: bytes):
    root = tmp_path
    dvc_file = root / "data" / "raw.dvc"
    workspace = root / "data" / "raw"
    cache_root = root / ".dvc" / "cache" / "files" / "md5"
    dvc_file.parent.mkdir(parents=True)
    workspace.mkdir(parents=True)

    expected_payload = b"authoritative-audio"
    expected_md5 = _md5(expected_payload)
    directory_payload = json.dumps(
        [{"md5": expected_md5, "relpath": "sample.mp3"}]
    ).encode()
    directory_md5 = _md5(directory_payload)
    dvc_file.write_text(
        yaml.safe_dump({"outs": [{"md5": f"{directory_md5}.dir", "path": "raw"}]}),
        encoding="utf-8",
    )
    manifest_path = cache_root / directory_md5[:2] / f"{directory_md5[2:]}.dir"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_bytes(directory_payload)

    (workspace / "sample.mp3").write_bytes(workspace_payload)
    cache_path = cache_root / expected_md5[:2] / expected_md5[2:]
    cache_path.parent.mkdir(parents=True)
    cache_path.write_bytes(cache_payload)
    return root, dvc_file, cache_root, expected_payload


def test_clean_workspace_and_cache(tmp_path):
    root, dvc_file, cache_root, expected = _fixture(
        tmp_path, b"authoritative-audio", b"authoritative-audio"
    )
    report = audit_dvc_directory(root, dvc_file, cache_root)
    assert expected == b"authoritative-audio"
    assert report["status"] == "clean"
    assert report["scanned_files"] == 1
    assert report["issues"] == []


def test_nonzero_checksum_corruption_is_reported(tmp_path):
    root, dvc_file, cache_root, _ = _fixture(
        tmp_path, b"truncated-but-not-empty", b"wrong-cache-content"
    )
    report = audit_dvc_directory(root, dvc_file, cache_root)
    assert report["status"] == "corrupt"
    assert report["workspace_mismatch"] == 1
    assert report["cache_mismatch"] == 1
    assert report["workspace_missing"] == 0
    assert report["cache_missing"] == 0
    assert report["issues"][0]["relpath"] == "sample.mp3"


def test_selected_path_must_exist_in_manifest(tmp_path):
    root, dvc_file, cache_root, _ = _fixture(
        tmp_path, b"authoritative-audio", b"authoritative-audio"
    )
    try:
        audit_dvc_directory(root, dvc_file, cache_root, ["absent.mp3"])
    except ValueError as exc:
        assert "absent.mp3" in str(exc)
    else:
        raise AssertionError("Unknown manifest path should fail closed")
