"""Fail-fast preflight for the controlled CAM++ OOF experiment block.

The six runs form two independent recipe families.  Inside each family the
only scientific variable is the validation fold; the cluster-map path is a
required fold-derived artifact and logging paths are operational only.

Usage:
    uv run --no-sync python scripts/verify_oof_experiments.py
"""

from __future__ import annotations

import argparse
import copy
import fnmatch
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment_config import load_profile  # noqa: E402


FAMILIES = {
    "no_proto": {
        "profiles": [
            f"p0-campp-no-proto-repro-es21-oof-f{i}" for i in range(3)
        ],
        "invariant_sha256": "3220bc14bd729ff199db908c9cf1b65ab9d8457e81be4d0cd278d1bb86ec4b0d",
        "baseline": ROOT / "checkpoints" / "campp_best (4).pt",
        "baseline_sha256": "92893c7642901dc2e1bc4eb1d70d9b51c8ed7b03c286b0c89f7340a46475ad40",
    },
    "metric_only": {
        "profiles": [f"p0-campp-metric-only-repro-oof-f{i}" for i in range(3)],
        "invariant_sha256": "6c6960fce4f727b1ba9783add54a093a1525830f066ad5995d9e7dc26db711e5",
        "baseline": ROOT / "checkpoints" / "campp_best (5).pt",
        "baseline_sha256": "ead5d1b7af290271db356c9ecf5e980513693d1f793a0c989e98094e8f0f37e5",
    },
}

CLUSTER_MAPS = {
    0: ("321dd54d57c27e7ac8ac7beea8211aca1b3f236dfe7dfba35d49f97e562ab5f9", 1482),
    1: ("e6cf053878275cd6d74ec643a73725888bb92d44eea01af3ae0f1dbd4d868d1f", 1482),
    2: ("a7a8987cbace55cac08dd5d8fa601b8a7beddb3de7072fee0fa47460a20186bd", 1483),
}

FOLD_DERIVED_OR_OPERATIONAL = {
    "data.split.fold",
    "model.unknown_cluster_path",
    "logging.checkpoint_dir",
    "logging.log_dir",
}

BASELINE_ALLOWED_PATTERNS = (
    "experiment.*",
    "logging.*",
    "mlops.tracking.username",
    "mlops.tracking.password",
    "training.seed",
    "training.deterministic_algorithms",
    # Operational safeguard added after the first two engineering-invalid
    # epochs: patience must not be consumed while the encoder is frozen.
    "training.early_stopping_start_epoch",
    # Explicit default added by the known-first implementation; ``metric`` is
    # exactly the pre-existing 446+k behavior of both baseline checkpoints.
    "model.speaker_target_scope",
    # Added later as a disabled-by-default P6 capability.  The explicit checks
    # below make this allowance fail closed if it is ever enabled here.
    "training.loss.speaker.inter_class.*",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_text_sha256(path: Path) -> str:
    """Hash text with LF newlines so Windows/Linux checkouts agree."""
    normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    normalized = normalized.replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix.rstrip("."): value}
    out: dict[str, Any] = {}
    for key, item in value.items():
        path = f"{prefix}{key}"
        if isinstance(item, dict):
            out.update(_flatten(item, f"{path}."))
        else:
            out[path] = item
    return out


def _diff_paths(left: dict, right: dict) -> set[str]:
    a, b = _flatten(left), _flatten(right)
    return {key for key in set(a) | set(b) if a.get(key) != b.get(key)}


def _family_invariant(config: dict) -> str:
    normalized = copy.deepcopy(config)
    normalized.pop("experiment", None)
    normalized.pop("logging", None)
    normalized["data"]["split"].pop("fold", None)
    normalized["model"].pop("unknown_cluster_path", None)
    payload = json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _allowed_baseline(path: str) -> bool:
    return any(fnmatch.fnmatch(path, pattern)
               for pattern in BASELINE_ALLOWED_PATTERNS)


def verify(skip_checkpoints: bool = False) -> dict:
    errors: list[str] = []
    report: dict[str, Any] = {"status": "ok", "families": {}, "cluster_maps": {}}

    for fold, (expected_hash, expected_files) in CLUSTER_MAPS.items():
        path = ROOT / "data" / "processed" / f"unknown_clusters_oof_f{fold}.json"
        if not path.is_file():
            errors.append(f"missing cluster map: {path}")
            continue
        actual_hash = _normalized_text_sha256(path)
        mapping = json.loads(path.read_text(encoding="utf-8"))
        clusters = len(set(mapping.values()))
        if actual_hash != expected_hash:
            errors.append(f"cluster map f{fold} hash drift: {actual_hash}")
        if len(mapping) != expected_files or clusters != 554:
            errors.append(
                f"cluster map f{fold} shape drift: files={len(mapping)}, clusters={clusters}")
        report["cluster_maps"][str(fold)] = {
            "sha256": actual_hash, "files": len(mapping), "clusters": clusters,
        }

    for family_name, spec in FAMILIES.items():
        configs = [load_profile(name) for name in spec["profiles"]]
        family_rows = []
        for fold, (name, config) in enumerate(zip(spec["profiles"], configs)):
            invariant = _family_invariant(config)
            if invariant != spec["invariant_sha256"]:
                errors.append(f"{name}: invariant hash drift: {invariant}")

            expected_cluster = f"data/processed/unknown_clusters_oof_f{fold}.json"
            checks = {
                "fold": config["data"]["split"].get("fold") == fold,
                "folds": config["data"]["split"].get("folds") == 3,
                "split_seed": config["data"]["split"].get("seed") == 42,
                "training_seed": config["training"].get("seed") == 42,
                "deterministic": config["training"].get("deterministic_algorithms") is True,
                "early_stopping_after_freeze": (
                    family_name != "no_proto"
                    or (
                        config["training"].get("freeze_epochs") == 20
                        and config["training"].get(
                            "early_stopping_start_epoch"
                        ) == 21
                        and config["training"].get(
                            "early_stopping_patience"
                        ) == 20
                    )
                ),
                "cluster_path": config["model"].get("unknown_cluster_path") == expected_cluster,
                "hardware_mode": config["hardware"].get("mode") == "vastai_3060",
                "batch_size": config["hardware"]["profiles"]["vastai_3060"].get("batch_size") == 16,
                "checkpoint_isolated": config["logging"].get("checkpoint_dir", "").endswith(name),
                "inter_class_disabled": (
                    config["training"]["loss"]["speaker"]
                    .get("inter_class", {})
                    .get("enabled") is False
                ),
            }
            for check, passed in checks.items():
                if not passed:
                    errors.append(f"{name}: failed check {check}")
            family_rows.append({"profile": name, "invariant_sha256": invariant,
                                "checks": checks})

        reference = configs[0]
        for name, config in zip(spec["profiles"][1:], configs[1:]):
            unexpected = _diff_paths(reference, config) - FOLD_DERIVED_OR_OPERATIONAL
            if unexpected:
                errors.append(f"{name}: unexpected within-family changes: {sorted(unexpected)}")

        baseline_info: dict[str, Any] = {"checked": False}
        baseline_path: Path = spec["baseline"]
        if not skip_checkpoints and baseline_path.is_file():
            actual_checkpoint_hash = _sha256(baseline_path)
            if actual_checkpoint_hash != spec["baseline_sha256"]:
                errors.append(
                    f"{family_name}: baseline checkpoint hash drift: {actual_checkpoint_hash}")
            checkpoint = torch.load(
                baseline_path, map_location="cpu", weights_only=False)
            baseline_diffs = _diff_paths(checkpoint["config"], reference)
            unexpected = {p for p in baseline_diffs if not _allowed_baseline(p)}
            if unexpected:
                errors.append(
                    f"{family_name}: recipe differs from baseline checkpoint: {sorted(unexpected)}")
            baseline_info = {
                "checked": True,
                "path": str(baseline_path.relative_to(ROOT)),
                "sha256": actual_checkpoint_hash,
                "allowed_operational_additions": sorted(baseline_diffs - unexpected),
            }

        report["families"][family_name] = {
            "expected_invariant_sha256": spec["invariant_sha256"],
            "runs": family_rows,
            "baseline": baseline_info,
        }

    if errors:
        report["status"] = "failed"
        report["errors"] = errors
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-checkpoints", action="store_true",
                        help="Skip optional comparison with the two local baseline .pt files")
    parser.add_argument("--json-out", default=None,
                        help="Optional path for a machine-readable preflight report")
    args = parser.parse_args()

    report = verify(skip_checkpoints=args.skip_checkpoints)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
