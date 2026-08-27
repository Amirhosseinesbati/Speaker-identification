"""Backfill and verify a DagsHub/MLflow run from a preserved FileStore run.

This recovery utility is intentionally independent from the training process.
It copies the immutable scientific record (parameters, missing metric series,
artifacts, and an optional MLflow model flavor) into an existing remote run,
then verifies every uploaded artifact by relative path and byte size.

Secrets are never copied from the local FileStore.  Only explicitly allowed
tracking variables are loaded from ``--env-file`` and their values are never
printed or written to the generated manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import mlflow
from mlflow.entities import Metric, Param, RunTag
from mlflow.tracking import MlflowClient


TRACKING_ENV_KEYS = {
    "MLFLOW_TRACKING_URI",
    "DAGSHUB_TRACKING_URI",
    "DAGSHUB_REPO_OWNER",
    "DAGSHUB_USERNAME",
    "DAGSHUB_USER_TOKEN",
    "DAGSHUB_TOKEN",
}
SECRET_KEY_RE = re.compile(
    r"(?:^|[_\-.])(token|secret|password|passwd|api[_-]?key|credential)(?:$|[_\-.])",
    re.IGNORECASE,
)
SECRET_FILE_NAMES = {
    ".env",
    "credentials",
    "credentials.json",
    "secrets.json",
    "id_rsa",
    "id_ed25519",
}
TERMINAL_STATUSES = {"FINISHED", "FAILED", "KILLED"}


@dataclass(frozen=True)
class ArtifactEntry:
    local_path: Path
    remote_path: str
    size_bytes: int
    sha256: str
    source: str

    def public_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("local_path")
        return data


def _clean_env(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_tracking_environment(path: Path | None) -> None:
    """Load only the tracking allowlist from a dotenv-style file."""
    if path is not None:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key in TRACKING_ENV_KEYS and not os.environ.get(key):
                os.environ[key] = _clean_env(value)

    for key in TRACKING_ENV_KEYS:
        if key in os.environ:
            os.environ[key] = _clean_env(os.environ[key])

    uri = os.environ.get("MLFLOW_TRACKING_URI") or os.environ.get(
        "DAGSHUB_TRACKING_URI", ""
    )
    owner = os.environ.get("DAGSHUB_REPO_OWNER") or os.environ.get(
        "DAGSHUB_USERNAME", ""
    )
    token = os.environ.get("DAGSHUB_USER_TOKEN") or os.environ.get(
        "DAGSHUB_TOKEN", ""
    )
    if not uri or not owner or not token:
        raise RuntimeError("DagsHub tracking environment is incomplete")
    os.environ["MLFLOW_TRACKING_USERNAME"] = owner
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    mlflow.set_tracking_uri(uri)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_secret_key(key: str) -> bool:
    return bool(SECRET_KEY_RE.search(key))


def is_safe_artifact(path: Path) -> bool:
    lowered = {part.lower() for part in path.parts}
    if lowered & SECRET_FILE_NAMES:
        return False
    return not any(is_secret_key(part) for part in path.parts)


def read_filestore_values(directory: Path) -> dict[str, str]:
    if not directory.is_dir():
        return {}
    values: dict[str, str] = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and not is_secret_key(path.name):
            values[path.name] = path.read_text(encoding="utf-8").strip()
    return values


def read_filestore_metrics(directory: Path) -> dict[str, list[Metric]]:
    """Read ``timestamp value step`` rows from an MLflow FileStore."""
    if not directory.is_dir():
        return {}
    result: dict[str, list[Metric]] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or is_secret_key(path.name):
            continue
        rows: list[Metric] = []
        for line_number, raw_line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            fields = raw_line.split()
            if len(fields) != 3:
                raise ValueError(
                    f"Malformed metric row in {path.name}:{line_number}"
                )
            timestamp, value, step = fields
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                raise ValueError(
                    f"Non-finite metric in {path.name}:{line_number}"
                )
            rows.append(
                Metric(
                    key=path.name,
                    value=numeric_value,
                    timestamp=int(timestamp),
                    step=int(step),
                )
            )
        if rows:
            result[path.name] = rows
    return result


def _iter_safe_files(directory: Path) -> Iterator[Path]:
    if not directory.is_dir():
        return
    for path in sorted(directory.rglob("*")):
        if path.is_file() and is_safe_artifact(path.relative_to(directory)):
            yield path


def collect_artifacts(
    directory: Path,
    *,
    remote_prefix: str = "",
    source: str,
) -> list[ArtifactEntry]:
    entries: list[ArtifactEntry] = []
    for path in _iter_safe_files(directory):
        relative = path.relative_to(directory).as_posix()
        remote_path = "/".join(part for part in (remote_prefix, relative) if part)
        entries.append(
            ArtifactEntry(
                local_path=path,
                remote_path=remote_path,
                size_bytes=path.stat().st_size,
                sha256=sha256_file(path),
                source=source,
            )
        )
    return entries


def collect_extra_artifacts(
    specifications: Sequence[tuple[Path, str]],
) -> list[ArtifactEntry]:
    entries: list[ArtifactEntry] = []
    for local_path, remote_path in specifications:
        if not local_path.is_file():
            continue
        relative = Path(remote_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe remote artifact path: {remote_path}")
        if not is_safe_artifact(relative):
            raise ValueError(f"Potential secret artifact refused: {remote_path}")
        entries.append(
            ArtifactEntry(
                local_path=local_path,
                remote_path=relative.as_posix(),
                size_bytes=local_path.stat().st_size,
                sha256=sha256_file(local_path),
                source="campaign",
            )
        )
    return entries


def build_manifest(
    entries: Sequence[ArtifactEntry],
    *,
    remote_run_id: str,
    local_run_id: str,
    local_model_id: str | None,
    profile: str | None,
    git_commit: str | None,
    config_sha256: str | None,
) -> dict[str, Any]:
    source_counts: dict[str, int] = {}
    source_bytes: dict[str, int] = {}
    for entry in entries:
        source_counts[entry.source] = source_counts.get(entry.source, 0) + 1
        source_bytes[entry.source] = source_bytes.get(entry.source, 0) + entry.size_bytes
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "remote_run_id": remote_run_id,
        "source_local_run_id": local_run_id,
        "source_local_model_id": local_model_id,
        "profile": profile,
        "git_commit": git_commit,
        "config_sha256": config_sha256,
        "artifact_count": len(entries),
        "total_bytes": sum(entry.size_bytes for entry in entries),
        "source_counts": source_counts,
        "source_bytes": source_bytes,
        "artifacts": [entry.public_dict() for entry in entries],
    }


def _remote_artifacts(
    client: MlflowClient, run_id: str, path: str | None = None
) -> dict[str, int | None]:
    inventory: dict[str, int | None] = {}
    pending = [path]
    while pending:
        current = pending.pop()
        for item in client.list_artifacts(run_id, current):
            if item.is_dir:
                pending.append(item.path)
            else:
                inventory[item.path] = getattr(item, "file_size", None)
    return inventory


def _chunks(values: Sequence[Metric], size: int = 500) -> Iterable[Sequence[Metric]]:
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


def _log_artifact_with_retries(
    client: MlflowClient,
    run_id: str,
    entry: ArtifactEntry,
    attempts: int = 3,
) -> None:
    parent = str(Path(entry.remote_path).parent).replace("\\", "/")
    artifact_path = None if parent in {"", "."} else parent
    desired_name = Path(entry.remote_path).name
    with tempfile.TemporaryDirectory(prefix="mlflow_artifact_stage_") as tmp:
        upload_path = entry.local_path
        if desired_name != entry.local_path.name:
            upload_path = Path(tmp) / desired_name
            shutil.copyfile(entry.local_path, upload_path)
        for attempt in range(1, attempts + 1):
            try:
                client.log_artifact(run_id, str(upload_path), artifact_path)
                return
            except Exception:
                if attempt >= attempts:
                    raise
                time.sleep(float(attempt * 2))


def _verify_remote_hashes(
    client: MlflowClient,
    run_id: str,
    entries: Sequence[ArtifactEntry],
) -> dict[str, dict[str, str]]:
    """Download each expected artifact and compare its SHA-256 digest."""
    mismatches: dict[str, dict[str, str]] = {}
    with tempfile.TemporaryDirectory(prefix="mlflow_remote_verify_") as tmp:
        destination = Path(tmp)
        for index, entry in enumerate(entries, start=1):
            downloaded = Path(
                client.download_artifacts(run_id, entry.remote_path, str(destination))
            )
            actual = sha256_file(downloaded)
            if actual != entry.sha256:
                mismatches[entry.remote_path] = {
                    "expected": entry.sha256,
                    "remote": actual,
                }
            print(
                json.dumps(
                    {
                        "status": "artifact_hash_verified",
                        "index": index,
                        "total": len(entries),
                        "path": entry.remote_path,
                        "match": actual == entry.sha256,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return mismatches


def _parse_extra(value: str) -> tuple[Path, str]:
    try:
        local, remote = value.split("::", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "extra artifact must be LOCAL::REMOTE"
        ) from exc
    return Path(local), remote


def backfill(args: argparse.Namespace) -> dict[str, Any]:
    local_run_dir = args.local_run_dir.resolve()
    artifacts_dir = local_run_dir / "artifacts"
    if not artifacts_dir.is_dir():
        raise FileNotFoundError(f"Local run artifacts missing: {artifacts_dir}")

    local_model_dir = args.local_model_dir.resolve() if args.local_model_dir else None
    model_artifacts_dir = local_model_dir / "artifacts" if local_model_dir else None
    if model_artifacts_dir is not None and not model_artifacts_dir.is_dir():
        raise FileNotFoundError(
            f"Local model flavor artifacts missing: {model_artifacts_dir}"
        )

    params = read_filestore_values(local_run_dir / "params")
    metrics = read_filestore_metrics(local_run_dir / "metrics")
    local_tags = read_filestore_values(local_run_dir / "tags")

    entries = collect_artifacts(
        artifacts_dir,
        source="filestore_run",
    )
    if model_artifacts_dir is not None:
        entries.extend(
            collect_artifacts(
                model_artifacts_dir,
                remote_prefix="model_flavor",
                source="filestore_model",
            )
        )
    entries.extend(collect_extra_artifacts(args.extra_artifact))

    duplicate_paths = {
        entry.remote_path
        for entry in entries
        if sum(other.remote_path == entry.remote_path for other in entries) > 1
    }
    if duplicate_paths:
        raise ValueError(f"Duplicate remote artifact paths: {sorted(duplicate_paths)}")

    manifest = build_manifest(
        entries,
        remote_run_id=args.remote_run_id,
        local_run_id=local_run_dir.name,
        local_model_id=local_model_dir.name if local_model_dir else None,
        profile=args.profile,
        git_commit=args.git_commit,
        config_sha256=args.config_sha256,
    )

    client = MlflowClient()
    run = client.get_run(args.remote_run_id)
    original_status = run.info.status
    remote_params = dict(run.data.params)
    remote_metric_keys = set(run.data.metrics)
    remote_before = _remote_artifacts(client, args.remote_run_id)

    new_params = {key: value for key, value in params.items() if key not in remote_params}
    param_conflicts = {
        key: {"remote": remote_params[key], "local": value}
        for key, value in params.items()
        if key in remote_params and remote_params[key] != value
    }
    missing_metric_series = {
        key: values for key, values in metrics.items() if key not in remote_metric_keys
    }
    safe_tags = {
        key: value
        for key, value in local_tags.items()
        if not key.startswith("mlflow.")
        and not is_secret_key(key)
        and key not in run.data.tags
    }
    artifacts_to_upload = [
        entry
        for entry in entries
        if entry.remote_path not in remote_before
        or remote_before[entry.remote_path] != entry.size_bytes
    ]

    summary: dict[str, Any] = {
        "status": "dry_run" if args.dry_run else "pending",
        "remote_run_id": args.remote_run_id,
        "remote_run_status_before": original_status,
        "local_params": len(params),
        "new_params": len(new_params),
        "param_conflicts": param_conflicts,
        "local_metric_keys": len(metrics),
        "new_metric_keys": len(missing_metric_series),
        "new_metric_points": sum(len(rows) for rows in missing_metric_series.values()),
        "safe_local_tags": len(safe_tags),
        "manifest_artifact_count": len(entries),
        "manifest_total_bytes": manifest["total_bytes"],
        "artifacts_to_upload": len(artifacts_to_upload),
        "bytes_to_upload": sum(entry.size_bytes for entry in artifacts_to_upload),
        "remote_artifacts_before": len(remote_before),
    }
    if args.dry_run:
        return summary

    if original_status not in TERMINAL_STATUSES and not args.allow_running:
        raise RuntimeError(
            f"Refusing to mutate non-terminal run with status {original_status}"
        )

    active = mlflow.start_run(run_id=args.remote_run_id)
    try:
        new_param_entities = [Param(key, value) for key, value in new_params.items()]
        for offset in range(0, len(new_param_entities), 100):
            client.log_batch(
                args.remote_run_id,
                params=new_param_entities[offset : offset + 100],
            )
        for rows in missing_metric_series.values():
            for batch in _chunks(rows):
                client.log_batch(args.remote_run_id, metrics=list(batch))
        safe_tag_entities = [RunTag(key, value) for key, value in safe_tags.items()]
        for offset in range(0, len(safe_tag_entities), 100):
            client.log_batch(
                args.remote_run_id,
                tags=safe_tag_entities[offset : offset + 100],
            )

        for index, entry in enumerate(artifacts_to_upload, start=1):
            _log_artifact_with_retries(client, args.remote_run_id, entry)
            print(
                json.dumps(
                    {
                        "status": "artifact_uploaded",
                        "index": index,
                        "total": len(artifacts_to_upload),
                        "path": entry.remote_path,
                        "size_bytes": entry.size_bytes,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

        with tempfile.TemporaryDirectory(prefix="mlflow_backfill_") as tmp:
            manifest_path = Path(tmp) / "artifact_manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            client.log_artifact(
                args.remote_run_id,
                str(manifest_path),
                artifact_path="provenance",
            )

        completed_at = datetime.now(timezone.utc).isoformat()
        recovery_tags = {
            "campaign.artifacts_backfilled": "true",
            "campaign.backfill_complete_at_utc": completed_at,
            "campaign.source_local_run_id": local_run_dir.name,
            "campaign.backfill_artifact_count": str(len(entries)),
            "campaign.backfill_total_bytes": str(manifest["total_bytes"]),
        }
        if local_model_dir:
            recovery_tags["campaign.source_local_model_id"] = local_model_dir.name
        if args.profile:
            recovery_tags["campaign.profile"] = args.profile
        if args.git_commit:
            recovery_tags["campaign.git_commit"] = args.git_commit
        if args.config_sha256:
            recovery_tags["campaign.config_sha256"] = args.config_sha256
        client.log_batch(
            args.remote_run_id,
            tags=[RunTag(key, value) for key, value in recovery_tags.items()],
        )

        mlflow.end_run(status=original_status if original_status in TERMINAL_STATUSES else "FINISHED")
        active = None
    except Exception:
        if active is not None and mlflow.active_run() is not None:
            mlflow.end_run(
                status=original_status if original_status in TERMINAL_STATUSES else "KILLED"
            )
        raise

    verified_run = client.get_run(args.remote_run_id)
    remote_after = _remote_artifacts(client, args.remote_run_id)
    missing_paths = [entry.remote_path for entry in entries if entry.remote_path not in remote_after]
    size_mismatches = {
        entry.remote_path: {
            "expected": entry.size_bytes,
            "remote": remote_after.get(entry.remote_path),
        }
        for entry in entries
        if entry.remote_path in remote_after
        and remote_after[entry.remote_path] is not None
        and remote_after[entry.remote_path] != entry.size_bytes
    }
    manifest_path = "provenance/artifact_manifest.json"
    if manifest_path not in remote_after:
        missing_paths.append(manifest_path)

    verified_params = verified_run.data.params
    missing_params = [key for key in new_params if key not in verified_params]
    verified_metric_keys = set(verified_run.data.metrics)
    missing_metrics = [key for key in missing_metric_series if key not in verified_metric_keys]
    hash_mismatches = (
        {}
        if args.skip_hash_verification or missing_paths or size_mismatches
        else _verify_remote_hashes(client, args.remote_run_id, entries)
    )
    verified = not (
        missing_paths
        or size_mismatches
        or hash_mismatches
        or missing_params
        or missing_metrics
    )
    summary.update(
        {
            "status": "verified" if verified else "verification_failed",
            "remote_run_status_after": verified_run.info.status,
            "remote_params_after": len(verified_params),
            "remote_metric_keys_after": len(verified_metric_keys),
            "remote_artifacts_after": len(remote_after),
            "remote_artifact_bytes_after": sum(
                size for size in remote_after.values() if size is not None
            ),
            "missing_paths": missing_paths,
            "size_mismatches": size_mismatches,
            "hash_verification": "skipped" if args.skip_hash_verification else "sha256",
            "hash_mismatches": hash_mismatches,
            "missing_params": missing_params,
            "missing_metrics": missing_metrics,
        }
    )
    if not verified:
        raise RuntimeError(json.dumps(summary, ensure_ascii=False))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-run-id", required=True)
    parser.add_argument("--local-run-dir", type=Path, required=True)
    parser.add_argument("--local-model-dir", type=Path)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--profile")
    parser.add_argument("--git-commit")
    parser.add_argument("--config-sha256")
    parser.add_argument(
        "--extra-artifact",
        action="append",
        default=[],
        type=_parse_extra,
        metavar="LOCAL::REMOTE",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-running", action="store_true")
    parser.add_argument(
        "--skip-hash-verification",
        action="store_true",
        help="Verify only path and byte size; default also downloads and hashes every file.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    load_tracking_environment(args.env_file)
    result = backfill(args)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
