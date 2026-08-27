"""Fail-fast preflight for the gated known-first CAM++ experiment block.

The gate compares two fold-0 runs that are identical except for a 5% metric
auxiliary over the 446 known + 554 pseudo identities.  Only the winning family
is allowed to continue to folds 1 and 2.

Usage:
    uv run --no-sync python scripts/verify_known_first_experiments.py
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment_config import load_profile  # noqa: E402
from src.model_factory import resolve_speaker_head_layout  # noqa: E402


FAMILIES = {
    "control": [f"p0-campp-known446-ood-control-oof-f{i}" for i in range(3)],
    "auxmetric": [f"p0-campp-known446-ood-auxmetric-oof-f{i}" for i in range(3)],
}

CLUSTER_MAPS = {
    0: ("321dd54d57c27e7ac8ac7beea8211aca1b3f236dfe7dfba35d49f97e562ab5f9", 1482),
    1: ("e6cf053878275cd6d74ec643a73725888bb92d44eea01af3ae0f1dbd4d868d1f", 1482),
    2: ("a7a8987cbace55cac08dd5d8fa601b8a7beddb3de7072fee0fa47460a20186bd", 1483),
}

EXPECTED_HARDWARE = {
    "mode": "vastai_3090_campp",
    "batch_size": 48,
    "num_workers": 8,
    "mixed_precision": True,
    "device": "cuda",
}


def _sha256(path: Path) -> str:
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


def _normalise_fold(config: dict) -> dict:
    value = copy.deepcopy(config)
    value.pop("experiment", None)
    value.pop("logging", None)
    value["data"]["split"].pop("fold", None)
    value["model"].pop("unknown_cluster_path", None)
    return value


def _hardware_checks(config: dict) -> dict[str, bool]:
    """Assert the measured RTX 3090 recipe, not merely cross-run equality."""
    hardware = config.get("hardware", {})
    mode = hardware.get("mode")
    profile = hardware.get("profiles", {}).get(mode, {})
    return {
        "hardware_mode": mode == EXPECTED_HARDWARE["mode"],
        "batch_size": profile.get("batch_size") == EXPECTED_HARDWARE["batch_size"],
        "num_workers": profile.get("num_workers") == EXPECTED_HARDWARE["num_workers"],
        "mixed_precision": profile.get("mixed_precision")
        is EXPECTED_HARDWARE["mixed_precision"],
        "device": profile.get("device") == EXPECTED_HARDWARE["device"],
    }


def verify() -> dict:
    errors: list[str] = []
    report: dict[str, Any] = {"status": "ok", "families": {}, "cluster_maps": {}}

    for fold, (expected_hash, expected_files) in CLUSTER_MAPS.items():
        path = ROOT / "data" / "processed" / f"unknown_clusters_oof_f{fold}.json"
        if not path.is_file():
            errors.append(f"missing cluster map: {path}")
            continue
        mapping = json.loads(path.read_text(encoding="utf-8"))
        actual_hash = _sha256(path)
        clusters = len(set(mapping.values()))
        if actual_hash != expected_hash:
            errors.append(f"cluster map f{fold} hash drift: {actual_hash}")
        if len(mapping) != expected_files or clusters != 554:
            errors.append(
                f"cluster map f{fold} shape drift: files={len(mapping)}, clusters={clusters}"
            )
        report["cluster_maps"][str(fold)] = {
            "sha256": actual_hash, "files": len(mapping), "clusters": clusters,
        }

    loaded: dict[str, list[dict]] = {}
    for family, profiles in FAMILIES.items():
        configs = [load_profile(name) for name in profiles]
        loaded[family] = configs
        reference = _normalise_fold(configs[0])
        rows = []
        for fold, (name, config) in enumerate(zip(profiles, configs)):
            model_cfg = config["model"]
            loss_cfg = config["training"]["loss"]
            head_classes, output_pseudo, scope = resolve_speaker_head_layout(
                config, 1000,
            )
            weights = (
                float(loss_cfg["speaker"]["weight"])
                + float(loss_cfg["ood"]["weight"])
                + float(loss_cfg["proto"].get("weight", 0.0)
                        if loss_cfg["proto"].get("enabled", False) else 0.0)
            )
            checks = {
                "fold": config["data"]["split"].get("fold") == fold,
                "cluster_path": model_cfg.get("unknown_cluster_path")
                == f"data/processed/unknown_clusters_oof_f{fold}.json",
                "known_scope": scope == "known",
                "head_classes": head_classes == 446,
                "output_pseudo_classes": output_pseudo == 0,
                "metric_pseudo_classes": model_cfg.get("num_unknown_clusters") == 554,
                "ood_head": model_cfg.get("ood_head") is True,
                "loss_weights_sum_to_one": abs(weights - 1.0) < 1e-9,
                "checkpoint_isolated": config["logging"]["checkpoint_dir"].endswith(name),
            }
            checks.update(_hardware_checks(config))
            expected_proto = family == "auxmetric"
            checks["proto_gate"] = bool(loss_cfg["proto"].get("enabled")) is expected_proto
            if expected_proto:
                checks["proto_scope_metric"] = loss_cfg["proto"].get("scope") == "metric"
            for check, passed in checks.items():
                if not passed:
                    errors.append(f"{name}: failed check {check}")

            unexpected = _diff_paths(reference, _normalise_fold(config))
            if unexpected:
                errors.append(f"{name}: unexpected within-family drift: {sorted(unexpected)}")
            rows.append({"profile": name, "checks": checks})
        report["families"][family] = rows

    # Across the two fold-0 gate configs, only experiment metadata, logging and
    # the three loss weights/proto settings may differ.
    control = copy.deepcopy(loaded["control"][0])
    auxiliary = copy.deepcopy(loaded["auxmetric"][0])
    control.pop("experiment", None)
    auxiliary.pop("experiment", None)
    control.pop("logging", None)
    auxiliary.pop("logging", None)
    allowed = {
        "training.loss.speaker.weight",
        "training.loss.ood.weight",
        "training.loss.proto.enabled",
        "training.loss.proto.scope",
        "training.loss.proto.weight",
        "training.loss.proto.scale",
        "training.loss.proto.margin",
        "training.loss.proto.decay",
    }
    unexpected_gate = _diff_paths(control, auxiliary) - allowed
    if unexpected_gate:
        errors.append(f"fold-0 gate has confounders: {sorted(unexpected_gate)}")
    report["fold0_gate_diff"] = sorted(_diff_paths(control, auxiliary))

    if errors:
        report["status"] = "failed"
        report["errors"] = errors
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()
    report = verify()
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
