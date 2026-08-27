"""Audit a DVC directory output against its manifest without changing data.

Unlike a zero-byte scan, this catches truncated or otherwise non-empty objects
whose content no longer matches the MD5 recorded by DVC.  The audit checks both
the materialised workspace files and the corresponding local cache objects.
Use ``--path`` to limit an investigation to known-suspicious relative paths;
without it, the complete directory output is verified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import yaml


ROOT = Path(__file__).resolve().parent.parent


def _cache_object(cache_root: Path, md5: str) -> Path:
    return cache_root / md5[:2] / md5[2:]


def _file_md5(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _same_file(left: Path, right: Path) -> bool:
    """Return true for a DVC hardlink/reflink that resolves to one inode."""
    try:
        left_stat = left.stat()
        right_stat = right.stat()
    except OSError:
        return False
    return (
        left_stat.st_dev == right_stat.st_dev
        and left_stat.st_ino != 0
        and left_stat.st_ino == right_stat.st_ino
    )


def load_directory_manifest(
    root: Path,
    dvc_file: Path,
    cache_root: Path,
) -> tuple[Path, dict[str, str]]:
    spec = yaml.safe_load(dvc_file.read_text(encoding="utf-8"))
    outs = spec.get("outs") or []
    if len(outs) != 1 or not str(outs[0].get("md5", "")).endswith(".dir"):
        raise ValueError("Expected exactly one DVC directory output")

    out = outs[0]
    workspace = (dvc_file.parent / out["path"]).resolve()
    directory_md5 = str(out["md5"]).removesuffix(".dir")
    manifest_path = _cache_object(cache_root, directory_md5).with_suffix(".dir")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_relpath = {str(entry["relpath"]): str(entry["md5"]) for entry in manifest}
    if not by_relpath:
        raise ValueError("DVC directory manifest is empty")
    return workspace, by_relpath


def audit_dvc_directory(
    root: Path,
    dvc_file: Path,
    cache_root: Path,
    selected_paths: Iterable[str] | None = None,
) -> dict:
    workspace, manifest = load_directory_manifest(root, dvc_file, cache_root)
    selected = sorted(set(selected_paths or manifest.keys()))
    unknown = sorted(set(selected) - manifest.keys())
    if unknown:
        raise ValueError(
            "Paths absent from DVC manifest: " + ", ".join(unknown[:10])
        )

    issues: list[dict[str, object]] = []
    counts = {
        "workspace_missing": 0,
        "workspace_mismatch": 0,
        "cache_missing": 0,
        "cache_mismatch": 0,
    }
    cache_results: dict[str, tuple[bool, str | None]] = {}
    bytes_hashed = 0

    for relpath in selected:
        expected = manifest[relpath]
        if expected.endswith(".dir"):
            raise ValueError(f"Nested directory object is unsupported: {relpath}")
        workspace_path = workspace / relpath
        cache_path = _cache_object(cache_root, expected)
        row: dict[str, object] = {"relpath": relpath, "expected_md5": expected}

        workspace_actual: str | None = None
        if not workspace_path.is_file():
            counts["workspace_missing"] += 1
            row["workspace"] = "missing"
        else:
            workspace_actual = _file_md5(workspace_path)
            bytes_hashed += workspace_path.stat().st_size
            if workspace_actual != expected:
                counts["workspace_mismatch"] += 1
                row["workspace"] = "checksum_mismatch"
                row["workspace_actual_md5"] = workspace_actual

        if expected not in cache_results:
            if not cache_path.is_file():
                cache_results[expected] = (False, None)
            elif workspace_actual is not None and _same_file(workspace_path, cache_path):
                cache_results[expected] = (workspace_actual == expected, workspace_actual)
            else:
                cache_actual = _file_md5(cache_path)
                bytes_hashed += cache_path.stat().st_size
                cache_results[expected] = (cache_actual == expected, cache_actual)

        cache_ok, cache_actual = cache_results[expected]
        if cache_actual is None:
            counts["cache_missing"] += 1
            row["cache"] = "missing"
        elif not cache_ok:
            counts["cache_mismatch"] += 1
            row["cache"] = "checksum_mismatch"
            row["cache_actual_md5"] = cache_actual

        if len(row) > 2:
            issues.append(row)

    report = {
        "status": "corrupt" if issues else "clean",
        "workspace": str(workspace),
        "scanned_files": len(selected),
        "unique_cache_objects": len({manifest[path] for path in selected}),
        "bytes_hashed": bytes_hashed,
        **counts,
        "issues": issues,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dvc-file", default="data/raw.dvc")
    parser.add_argument("--cache-root", default=".dvc/cache/files/md5")
    parser.add_argument(
        "--path",
        action="append",
        default=None,
        help="Manifest-relative path to audit; repeat for multiple paths.",
    )
    parser.add_argument("--output", default=None, help="Optional JSON report path")
    args = parser.parse_args()

    root = ROOT.resolve()
    dvc_file = (root / args.dvc_file).resolve()
    cache_root = (root / args.cache_root).resolve()
    report = audit_dvc_directory(root, dvc_file, cache_root, args.path)
    payload = json.dumps(report, indent=2) + "\n"
    print(payload, end="")
    if args.output:
        output = (root / args.output).resolve()
        if root != output and root not in output.parents:
            raise SystemExit(f"Output must remain inside project root: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    return 1 if report["status"] != "clean" else 0


if __name__ == "__main__":
    raise SystemExit(main())
