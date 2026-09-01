"""Mechanically preregister the conditional P9 ECAPA SE/BN follow-up.

The script is intentionally unusable until the P8 parent run is terminal.  It
locks the selected parent checkpoint SHA/size/metrics into a single P9 config
and a separate JSON contract before any P9 metric can exist.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
import yaml


PARENT_PROFILE = "p8-ecapa-frozen-known446-ood-complement-oof-f0"
P9_PROFILE = "p9-ecapa-sebn-known446-ood-adapter-oof-f0"
P8_STANDALONE_GATE = 0.9269211906147802
EXPECTED_SPLIT = {"scheme": "kfold", "folds": 3, "fold": 0, "seed": 42}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _parent_receipt(checkpoint_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    required = {"model_state_dict", "config", "class_map", "epoch"}
    missing = required.difference(checkpoint)
    if missing:
        raise RuntimeError(f"P8 checkpoint missing keys: {sorted(missing)}")

    config = checkpoint["config"]
    embedded_name = (config.get("_meta", {}) or {}).get("name")
    checkpoint_dir = str(
        (config.get("logging", {}) or {}).get("checkpoint_dir", "")
    ).replace("\\", "/")
    if embedded_name not in (None, PARENT_PROFILE):
        raise RuntimeError("source checkpoint embeds a different profile name")
    if not checkpoint_dir.endswith(f"checkpoints/{PARENT_PROFILE}"):
        raise RuntimeError("source checkpoint is not bound to the P8 directory")
    model = config.get("model", {}) or {}
    ecapa = (model.get("encoder_config", {}) or {}).get("ecapa", {}) or {}
    split = (config.get("data", {}) or {}).get("split", {}) or {}
    if model.get("encoder_type") != "ecapa" or not ecapa.get("freeze_encoder"):
        raise RuntimeError("P8 source is not a fully frozen ECAPA checkpoint")
    if ecapa.get("adapter_mode") not in (None, "none"):
        raise RuntimeError("P8 source already contains an encoder adapter")
    if split != EXPECTED_SPLIT:
        raise RuntimeError(f"P8 split mismatch: {split}")
    if len(checkpoint["class_map"]) != 1001:
        raise RuntimeError("P8 class-map size is not 1001")
    if str(checkpoint.get("weight_variant", "raw")).lower() != "raw":
        raise RuntimeError("P9 must warm-start the selected Raw checkpoint")

    selected_epoch = int(checkpoint["epoch"])
    history = checkpoint.get("training_history")
    if not isinstance(history, list) or not history:
        raise RuntimeError("P8 checkpoint has no embedded training history")
    selected_rows = [
        row for row in history if int(row.get("epoch", -1)) == selected_epoch
    ]
    if len(selected_rows) != 1:
        raise RuntimeError("P8 selected epoch is absent or duplicated in history")
    selected = selected_rows[0]
    raw = float(selected["val_macro_f1"])
    history_best = max(float(row["val_macro_f1"]) for row in history)
    if abs(raw - history_best) > 1e-12:
        raise RuntimeError("P8 selected Raw checkpoint is not the history maximum")
    if abs(raw - float(checkpoint["val_macro_f1"])) > 1e-12:
        raise RuntimeError("P8 checkpoint metric disagrees with embedded history")
    if not (0.90 <= raw < P8_STANDALONE_GATE):
        raise RuntimeError(
            "conditional P9 trigger is false: terminal P8 Raw must be in "
            f"[0.90, {P8_STANDALONE_GATE})"
        )

    return {
        "profile": PARENT_PROFILE,
        "checkpoint_path": checkpoint_path.as_posix(),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "selected_epoch": selected_epoch,
        "raw_macro_f1": raw,
        "known_accuracy": float(selected["val_known_acc"]),
        "ood_f1": float(selected["val_ood_f1"]),
        "logit_macro_f1": float(selected["val_logit_avg_macro_f1"]),
        "ema_macro_f1": float(selected["val_ema_macro_f1"]),
        "history_points_at_selection": len(history),
        "split": split,
        "class_map_size": len(checkpoint["class_map"]),
    }


def build_p9_profile(parent_profile: dict[str, Any], parent: dict[str, Any]) -> dict:
    profile = copy.deepcopy(parent_profile)
    profile["_meta"]["name"] = P9_PROFILE
    profile["experiment"] = {
        "priority": "P9-CONDITIONAL-DOMAIN-ADAPTER",
        "family": "ecapa-sebn-known446-ood-adapter-oof",
        "purpose": (
            "Test whether all-group SE/BN target-domain adaptation lifts the "
            "sub-gate frozen P8 representation without changing its heads"
        ),
        "expected_decision": (
            "Fold0 gate only; no later fold, tuned fusion, or submission is "
            "authorized by this profile"
        ),
        "decision_policy": "raw_probability_average_argmax",
        "diagnostic_ensemble": (
            "fixed_probability_average_50_50_with_campp_control_fold0"
        ),
        "adapter_evidence": {
            "paper": "https://doi.org/10.21437/Interspeech.2024-2476",
            "arxiv": "https://arxiv.org/abs/2406.07832",
            "implementation_commit": "e125ce6",
            "scope": "all_native_ecapa_se_blocks_and_bn_affine_statistics",
            "optimizer_values_are_campaign_preregistered_not_paper_reported": True,
        },
        "preregistered_parent": parent,
        "preregistered_gate": {
            "standalone_min_raw_macro_f1": max(
                P8_STANDALONE_GATE, parent["raw_macro_f1"] + 0.002
            ),
            "minimum_raw_gain_vs_parent": 0.002,
            "max_known_accuracy_drop_vs_parent": 0.001,
            "max_ood_f1_drop_vs_parent": 0.001,
            "minimum_supporting_epochs": 2,
            "fixed_50_50_min_macro_gain_vs_campp": 0.002,
            "fixed_50_50_max_known_accuracy_drop": 0.001,
            "fixed_50_50_max_ood_f1_drop": 0.001,
            "min_campp_error_rescue_rate": 0.25,
            "require_rescued_gt_introduced": True,
        },
        "stop_rules": {
            "futility_epoch": 12,
            "maximum_raw_deficit_vs_parent": 0.003,
            "early_stopping_start_epoch": 5,
            "early_stopping_patience": 12,
            "max_epochs": 45,
            "max_run_hours": 2.5,
            "max_incremental_cost_usd": 0.50,
        },
    }

    ecapa = profile["model"]["encoder_config"]["ecapa"]
    ecapa["freeze_encoder"] = True
    ecapa["unfreeze_last_n_blocks"] = 0
    ecapa["adapter_mode"] = "se_bn"

    source_hardware = profile["hardware"]["profiles"].pop(
        profile["hardware"]["mode"]
    )
    profile["hardware"]["mode"] = "vastai_3090_ecapa_se_bn"
    source_hardware["description"] = (
        "Conditional P9 SE/BN adapter profile; must pass a measured RTX 3090 "
        "preflight before launch"
    )
    profile["hardware"]["profiles"] = {
        "vastai_3090_ecapa_se_bn": source_hardware
    }
    profile["logging"] = {
        "checkpoint_dir": f"checkpoints/{P9_PROFILE}",
        "log_dir": f"logs/{P9_PROFILE}",
    }

    training = profile["training"]
    training.update({
        "epochs": 45,
        "freeze_epochs": 0,
        "early_stopping_start_epoch": 5,
        "early_stopping_patience": 12,
        "learning_rate": 0.0,
        "encoder_lr": 1e-5,
        "weight_decay": 1e-4,
        "schedule": "cosine",
        "warmup_ratio": 0.05,
        "min_lr_ratio": 0.05,
        "warm_start_checkpoint": parent["checkpoint_path"],
        "selection_variant": "raw",
    })
    return profile


def preregister(
    parent_checkpoint: Path,
    parent_profile_path: Path,
    output_profile_path: Path,
    output_contract_path: Path,
) -> dict[str, Any]:
    parent = _parent_receipt(parent_checkpoint)
    source_profile = yaml.safe_load(parent_profile_path.read_text(encoding="utf-8"))
    if source_profile.get("_meta", {}).get("name") != PARENT_PROFILE:
        raise RuntimeError("parent YAML is not the locked P8 profile")
    profile = build_p9_profile(source_profile, parent)
    profile_text = yaml.safe_dump(
        profile, allow_unicode=True, sort_keys=False, width=100
    )
    _atomic_text(output_profile_path, profile_text)
    profile_sha = sha256_file(output_profile_path)

    contract = {
        "status": "preregistered_after_parent_terminal_before_p9_metrics",
        "profile": P9_PROFILE,
        "profile_path": output_profile_path.as_posix(),
        "profile_sha256": profile_sha,
        "parent": parent,
        "single_variable": "enable_all_group_ecapa_se_bn_adapter",
        "trainable_encoder_parameters": "SEBlock parameters plus BN affine parameters",
        "heads": "warm_started_and_fixed_by_zero_lr",
        "adapter_lr": 1e-5,
        "head_lr": 0.0,
        "max_epochs": 45,
        "early_stopping_start_epoch": 5,
        "early_stopping_patience": 12,
        "no_hyperparameter_sweep": True,
        "no_leaderboard_selection": True,
        "fold0_pass_authorizes_only_separate_fold1_fold2_preregistration": True,
        "gate": profile["experiment"]["preregistered_gate"],
        "stop_rules": profile["experiment"]["stop_rules"],
    }
    _atomic_text(
        output_contract_path,
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n",
    )
    return {
        "profile": P9_PROFILE,
        "profile_sha256": profile_sha,
        "parent_checkpoint_sha256": parent["checkpoint_sha256"],
        "parent_epoch": parent["selected_epoch"],
        "parent_raw_macro_f1": parent["raw_macro_f1"],
        "standalone_gate": profile["experiment"]["preregistered_gate"][
            "standalone_min_raw_macro_f1"
        ],
        "profile_path": output_profile_path.as_posix(),
        "contract_path": output_contract_path.as_posix(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--parent-profile",
        type=Path,
        default=Path(f"configs/experiments/{PARENT_PROFILE}.yaml"),
    )
    parser.add_argument(
        "--output-profile",
        type=Path,
        default=Path(f"configs/experiments/{P9_PROFILE}.yaml"),
    )
    parser.add_argument(
        "--output-contract",
        type=Path,
        default=Path(
            "configs/analyses/"
            "ecapa-sebn-known446-ood-adapter-oof-f0.prereg.json"
        ),
    )
    args = parser.parse_args()
    receipt = preregister(
        args.parent_checkpoint,
        args.parent_profile,
        args.output_profile,
        args.output_contract,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
