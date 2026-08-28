import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.mlflow_append_analysis import (
    CANONICAL_MANIFEST,
    build_append_manifest,
    validate_append_inputs,
)
from scripts.mlflow_backfill_run import collect_extra_artifacts


def _entries(tmp_path: Path):
    report = tmp_path / "report.md"
    report.write_text("scientific result\n", encoding="utf-8")
    diagnostic = tmp_path / "diagnostic.json"
    diagnostic.write_text(json.dumps({"macro_f1": 0.94}), encoding="utf-8")
    return collect_extra_artifacts(
        [
            (report, "analysis/auxmetric_fold0_result.md"),
            (diagnostic, "analysis/auxmetric_fold0_diagnostic.json"),
        ]
    )


def test_append_manifest_is_hash_complete_and_has_no_local_paths(tmp_path):
    entries = _entries(tmp_path)
    remote_path = validate_append_inputs(
        entries, "provenance/auxmetric_fold0_analysis_manifest.json"
    )
    manifest = build_append_manifest(
        entries,
        remote_run_id="run-1",
        manifest_remote_path=remote_path,
    )

    assert manifest["kind"] == "immutable_analysis_append"
    assert manifest["artifact_count"] == 2
    assert all(len(row["sha256"]) == 64 for row in manifest["artifacts"])
    assert all("local_path" not in row for row in manifest["artifacts"])


def test_append_refuses_canonical_collision_and_duplicate_paths(tmp_path):
    entries = _entries(tmp_path)
    with pytest.raises(ValueError, match="canonical"):
        validate_append_inputs(entries, CANONICAL_MANIFEST)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_append_inputs(
            [entries[0], entries[0]],
            "provenance/unique_analysis_manifest.json",
        )
    with pytest.raises(ValueError, match="collides"):
        validate_append_inputs(
            entries, "analysis/auxmetric_fold0_result.md"
        )


def test_append_script_help_runs_from_project_root():
    result = subprocess.run(
        [sys.executable, "scripts/mlflow_append_analysis.py", "--help"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--manifest-remote-path" in result.stdout
