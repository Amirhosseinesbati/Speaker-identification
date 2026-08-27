"""Quarantine zero-byte DVC workspace/cache objects for targeted re-pull.

This is a deliberately narrow recovery tool for interrupted/corrupt transfers.
It never deletes data: with ``--apply`` it moves only workspace files that are
currently zero bytes and their matching zero-byte cache objects into a
timestamped quarantine directory.  Run ``dvc pull`` and ``dvc checkout`` after
it, then re-run this script without ``--apply`` to verify that no zeros remain.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent


def _cache_object(cache_root: Path, md5: str) -> Path:
    return cache_root / md5[:2] / md5[2:]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dvc-file", default="data/raw.dvc")
    parser.add_argument("--cache-root", default=".dvc/cache/files/md5")
    parser.add_argument(
        "--quarantine-root", default="data/experiments/dvc_zero_quarantine",
    )
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    dvc_file = (ROOT / args.dvc_file).resolve()
    spec = yaml.safe_load(dvc_file.read_text(encoding="utf-8"))
    outs = spec.get("outs") or []
    if len(outs) != 1 or not str(outs[0].get("md5", "")).endswith(".dir"):
        raise SystemExit("Expected exactly one DVC directory output")
    out = outs[0]
    workspace = (dvc_file.parent / out["path"]).resolve()
    cache_root = (ROOT / args.cache_root).resolve()
    directory_md5 = str(out["md5"]).removesuffix(".dir")
    manifest_path = _cache_object(cache_root, directory_md5).with_suffix(".dir")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_relpath = {entry["relpath"]: entry["md5"] for entry in manifest}

    zeros = sorted(
        path for path in workspace.rglob("*")
        if path.is_file() and path.stat().st_size == 0
    )
    rows = []
    for path in zeros:
        relpath = path.relative_to(workspace).as_posix()
        md5 = by_relpath.get(relpath)
        if not md5:
            raise SystemExit(f"Zero-byte workspace file is absent from DVC manifest: {relpath}")
        cache_path = _cache_object(cache_root, md5)
        if not cache_path.is_file():
            raise SystemExit(f"Expected cache object is missing: {cache_path}")
        if cache_path.stat().st_size != 0:
            raise SystemExit(
                f"Refusing repair: workspace is zero but cache is non-zero: {relpath}"
            )
        rows.append({"relpath": relpath, "md5": md5, "cache": cache_path})

    unique_cache_objects = {row["md5"]: row["cache"] for row in rows}
    report = {
        "status": "needs_repair" if rows else "clean",
        "apply": bool(args.apply),
        "workspace": str(workspace),
        "zero_workspace_files": len(rows),
        "zero_cache_objects": len(unique_cache_objects),
    }
    print(json.dumps(report, indent=2))
    if not rows or not args.apply:
        return 1 if rows else 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    quarantine = (ROOT / args.quarantine_root / stamp).resolve()
    if ROOT not in quarantine.parents:
        raise SystemExit(f"Quarantine must remain inside project root: {quarantine}")
    workspace_quarantine = quarantine / "workspace"
    cache_quarantine = quarantine / "cache"

    for md5, cache_path in unique_cache_objects.items():
        destination = cache_quarantine / md5[:2] / md5[2:]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(cache_path), str(destination))

    for row in rows:
        source = workspace / row["relpath"]
        destination = workspace_quarantine / row["relpath"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))

    receipt = {
        **report,
        "status": "quarantined",
        "quarantine": str(quarantine.relative_to(ROOT)),
        "files": [{"relpath": row["relpath"], "md5": row["md5"]} for row in rows],
    }
    receipt_path = quarantine / "receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8",
    )
    os.chmod(receipt_path, 0o600)
    print(f"Quarantined exact zero-byte objects: {quarantine.relative_to(ROOT)}")
    print("Next: dvc pull data/raw.dvc --force && dvc checkout data/raw.dvc --relink")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
