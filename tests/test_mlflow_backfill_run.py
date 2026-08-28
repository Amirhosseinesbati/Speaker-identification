from pathlib import Path

from scripts.mlflow_backfill_run import (
    ArtifactEntry,
    _log_artifact_with_retries,
    build_manifest,
    collect_artifacts,
    is_safe_artifact,
    is_secret_key,
    read_filestore_metrics,
    read_filestore_values,
    validate_remote_artifact_path,
)


def test_secret_keys_and_files_are_rejected():
    assert is_secret_key("DAGSHUB_USER_TOKEN")
    assert is_secret_key("api-key")
    assert is_secret_key("db.password")
    assert not is_secret_key("learning_rate")
    assert not is_safe_artifact(Path("nested/.env"))
    assert not is_safe_artifact(Path("config/api_token.txt"))
    assert is_safe_artifact(Path("configs/resolved.yaml"))


def test_remote_artifact_path_is_normalized_and_rejects_unsafe_values():
    assert (
        validate_remote_artifact_path(
            "provenance/auxmetric_fold0_analysis_manifest.json"
        )
        == "provenance/auxmetric_fold0_analysis_manifest.json"
    )
    for unsafe in ("", "../escape.json", "/absolute.json", "secrets/api_token.json"):
        try:
            validate_remote_artifact_path(unsafe)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe path accepted: {unsafe}")


def test_read_filestore_values_omits_secrets(tmp_path):
    params = tmp_path / "params"
    params.mkdir()
    (params / "batch_size").write_text("32", encoding="utf-8")
    (params / "api_token").write_text("do-not-copy", encoding="utf-8")

    assert read_filestore_values(params) == {"batch_size": "32"}


def test_read_filestore_metrics_preserves_history(tmp_path):
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    (metrics / "val_macro_f1").write_text(
        "1000 0.91 1\n2000 0.93 2\n", encoding="utf-8"
    )

    parsed = read_filestore_metrics(metrics)

    assert list(parsed) == ["val_macro_f1"]
    assert [metric.value for metric in parsed["val_macro_f1"]] == [0.91, 0.93]
    assert [metric.step for metric in parsed["val_macro_f1"]] == [1, 2]


def test_manifest_has_hashes_counts_and_no_local_paths(tmp_path):
    artifacts = tmp_path / "artifacts"
    (artifacts / "models").mkdir(parents=True)
    (artifacts / "models" / "best.pt").write_bytes(b"model-bytes")
    (artifacts / "config.yaml").write_text("epochs: 3\n", encoding="utf-8")

    entries = collect_artifacts(artifacts, source="filestore_run")
    manifest = build_manifest(
        entries,
        remote_run_id="remote-1",
        local_run_id="local-1",
        local_model_id=None,
        profile="profile-1",
        git_commit="abc123",
        config_sha256="def456",
    )

    assert manifest["artifact_count"] == 2
    assert manifest["total_bytes"] == sum(path.stat().st_size for path in artifacts.rglob("*") if path.is_file())
    assert {row["remote_path"] for row in manifest["artifacts"]} == {
        "config.yaml",
        "models/best.pt",
    }
    assert all(len(row["sha256"]) == 64 for row in manifest["artifacts"])
    assert all("local_path" not in row for row in manifest["artifacts"])


def test_artifact_upload_can_assign_a_stable_remote_name(tmp_path):
    source = tmp_path / "timestamped-control.log"
    source.write_text("complete log", encoding="utf-8")
    entry = ArtifactEntry(
        local_path=source,
        remote_path="campaign/logs/control_fold0.log",
        size_bytes=source.stat().st_size,
        sha256="unused-in-this-test",
        source="campaign",
    )

    class RecordingClient:
        call = None

        def log_artifact(self, run_id, local_path, artifact_path):
            uploaded = Path(local_path)
            self.call = (run_id, uploaded.name, uploaded.read_text(), artifact_path)

    client = RecordingClient()
    _log_artifact_with_retries(client, "run-1", entry)

    assert client.call == (
        "run-1",
        "control_fold0.log",
        "complete log",
        "campaign/logs",
    )
