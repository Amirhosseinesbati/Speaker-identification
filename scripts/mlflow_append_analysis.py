"""Append immutable analysis artifacts to a terminal MLflow run.

Unlike the full FileStore recovery utility, this command needs no local MLflow
run directory.  It is intended for post-run reports and diagnostics.  Existing
remote paths are accepted only when their downloaded SHA-256 is identical;
conflicting content is refused.  The caller must provide a unique manifest path
so the canonical recovery manifest can never be overwritten accidentally.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from mlflow.entities import RunTag
from mlflow.tracking import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mlflow_backfill_run import (
    TERMINAL_STATUSES,
    ArtifactEntry,
    _log_artifact_with_retries,
    _remote_artifacts,
    _verify_remote_hashes,
    collect_extra_artifacts,
    load_tracking_environment,
    sha256_file,
    validate_remote_artifact_path,
)


CANONICAL_MANIFEST = "provenance/artifact_manifest.json"


def build_append_manifest(
    entries: Sequence[ArtifactEntry],
    *,
    remote_run_id: str,
    manifest_remote_path: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "immutable_analysis_append",
        "remote_run_id": remote_run_id,
        "manifest_remote_path": manifest_remote_path,
        "artifact_count": len(entries),
        "total_bytes": sum(entry.size_bytes for entry in entries),
        "artifacts": [entry.public_dict() for entry in entries],
    }


def validate_append_inputs(
    entries: Sequence[ArtifactEntry], manifest_remote_path: str
) -> str:
    normalized_manifest = validate_remote_artifact_path(manifest_remote_path)
    if normalized_manifest == CANONICAL_MANIFEST:
        raise ValueError(
            "Analysis append refuses the canonical recovery manifest path"
        )
    if not entries:
        raise ValueError("At least one analysis artifact is required")
    remote_paths = [entry.remote_path for entry in entries]
    if len(remote_paths) != len(set(remote_paths)):
        raise ValueError("Duplicate analysis artifact remote paths")
    if normalized_manifest in set(remote_paths):
        raise ValueError("Manifest path collides with an analysis artifact")
    return normalized_manifest


def _downloaded_sha256(
    client: MlflowClient, run_id: str, remote_path: str
) -> str:
    with tempfile.TemporaryDirectory(prefix="mlflow_append_probe_") as tmp:
        downloaded = Path(client.download_artifacts(run_id, remote_path, tmp))
        return sha256_file(downloaded)


def _classify_remote_entries(
    client: MlflowClient,
    run_id: str,
    entries: Sequence[ArtifactEntry],
    remote_inventory: dict[str, int | None],
) -> tuple[list[ArtifactEntry], list[str], dict[str, dict[str, str]]]:
    upload: list[ArtifactEntry] = []
    identical: list[str] = []
    conflicts: dict[str, dict[str, str]] = {}
    for entry in entries:
        if entry.remote_path not in remote_inventory:
            upload.append(entry)
            continue
        remote_sha = _downloaded_sha256(client, run_id, entry.remote_path)
        if remote_sha == entry.sha256:
            identical.append(entry.remote_path)
        else:
            conflicts[entry.remote_path] = {
                "local_sha256": entry.sha256,
                "remote_sha256": remote_sha,
            }
    return upload, identical, conflicts


def append_analysis(args: argparse.Namespace) -> dict[str, Any]:
    entries = collect_extra_artifacts(args.artifact)
    manifest_remote_path = validate_append_inputs(
        entries, args.manifest_remote_path
    )
    client = MlflowClient()
    run = client.get_run(args.remote_run_id)
    if run.info.status not in TERMINAL_STATUSES:
        raise RuntimeError(
            f"Refusing to append to non-terminal run: {run.info.status}"
        )

    remote_before = _remote_artifacts(client, args.remote_run_id)
    upload, identical, conflicts = _classify_remote_entries(
        client,
        args.remote_run_id,
        entries,
        remote_before,
    )
    if conflicts:
        raise RuntimeError(
            "Immutable analysis artifact conflict: "
            + json.dumps(conflicts, ensure_ascii=False, sort_keys=True)
        )

    manifest = build_append_manifest(
        entries,
        remote_run_id=args.remote_run_id,
        manifest_remote_path=manifest_remote_path,
    )
    with tempfile.TemporaryDirectory(prefix="mlflow_analysis_append_") as tmp:
        manifest_local = Path(tmp) / Path(manifest_remote_path).name
        manifest_local.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest_entry = ArtifactEntry(
            local_path=manifest_local,
            remote_path=manifest_remote_path,
            size_bytes=manifest_local.stat().st_size,
            sha256=sha256_file(manifest_local),
            source="analysis_manifest",
        )

        manifest_upload, manifest_identical, manifest_conflicts = (
            _classify_remote_entries(
                client,
                args.remote_run_id,
                [manifest_entry],
                remote_before,
            )
        )
        if manifest_conflicts:
            raise RuntimeError(
                "Analysis manifest path already contains different content: "
                + json.dumps(manifest_conflicts, ensure_ascii=False, sort_keys=True)
            )

        if args.dry_run:
            return {
                "status": "dry_run",
                "remote_run_id": args.remote_run_id,
                "remote_run_status": run.info.status,
                "artifacts": len(entries),
                "artifacts_to_upload": len(upload),
                "artifacts_already_identical": identical,
                "manifest_remote_path": manifest_remote_path,
                "manifest_to_upload": bool(manifest_upload),
                "manifest_already_identical": manifest_identical,
            }

        for entry in upload:
            _log_artifact_with_retries(client, args.remote_run_id, entry)
        if manifest_upload:
            _log_artifact_with_retries(
                client, args.remote_run_id, manifest_entry
            )

        expected = [*entries, manifest_entry]
        remote_after = _remote_artifacts(client, args.remote_run_id)
        missing = [
            entry.remote_path
            for entry in expected
            if entry.remote_path not in remote_after
        ]
        size_mismatches = {
            entry.remote_path: {
                "local": entry.size_bytes,
                "remote": remote_after.get(entry.remote_path),
            }
            for entry in expected
            if entry.remote_path in remote_after
            and remote_after[entry.remote_path] is not None
            and remote_after[entry.remote_path] != entry.size_bytes
        }
        hash_mismatches = (
            {}
            if missing or size_mismatches
            else _verify_remote_hashes(
                client, args.remote_run_id, expected
            )
        )
        if missing or size_mismatches or hash_mismatches:
            raise RuntimeError(
                json.dumps(
                    {
                        "missing": missing,
                        "size_mismatches": size_mismatches,
                        "hash_mismatches": hash_mismatches,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )

        client.log_batch(
            args.remote_run_id,
            tags=[
                RunTag("campaign.analysis_append_verified", "true"),
                RunTag("campaign.latest_analysis_manifest", manifest_remote_path),
                RunTag(
                    "campaign.analysis_append_verified_at_utc",
                    datetime.now(timezone.utc).isoformat(),
                ),
            ],
        )
        return {
            "status": "verified",
            "remote_run_id": args.remote_run_id,
            "remote_run_status": run.info.status,
            "artifact_count": len(entries),
            "artifact_bytes": sum(entry.size_bytes for entry in entries),
            "uploaded": [entry.remote_path for entry in upload],
            "already_identical": identical,
            "manifest_remote_path": manifest_remote_path,
            "manifest_sha256": manifest_entry.sha256,
            "hash_mismatches": {},
        }


def _parse_extra(value: str) -> tuple[Path, str]:
    try:
        local, remote = value.split("::", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "artifact must be LOCAL::REMOTE"
        ) from exc
    return Path(local), remote


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-run-id", required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument(
        "--artifact",
        action="append",
        required=True,
        type=_parse_extra,
        metavar="LOCAL::REMOTE",
    )
    parser.add_argument(
        "--manifest-remote-path",
        required=True,
        help="Unique immutable path; canonical provenance/artifact_manifest.json is refused",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    load_tracking_environment(args.env_file)
    result = append_analysis(args)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
