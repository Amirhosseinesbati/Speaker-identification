"""Measure safe CAM++ training batch sizes on the actual worker and recipe.

The probe runs real forward/backward/optimizer steps through ``train_epoch``;
it does not estimate VRAM from parameter counts.  Results include peak memory,
files/s and windows/s.  The recommendation maximizes measured file throughput
subject to a configurable VRAM headroom, but it remains a recipe decision: a
larger batch changes optimizer-step frequency and must be recorded in a new
experiment profile before scientific training.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_pipeline import get_active_profile, get_dataloaders  # noqa: E402
from src.batch_probe import select_recommended_batch  # noqa: E402
from src.experiment_config import load_profile  # noqa: E402
from src.heads import ood_head_enabled  # noqa: E402
from src.model_factory import create_model_from_config  # noqa: E402
from src.train import PrototypicalLoss, build_criterion, train_epoch  # noqa: E402
from src.training_utils import (  # noqa: E402
    apply_encoder_finetune_mode,
    build_amp,
    seed_everything,
)


def _optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    train_cfg = config["training"]
    encoder_params = [
        parameter for name, parameter in model.named_parameters()
        if "encoder" in name and parameter.requires_grad
    ]
    head_params = [
        parameter for name, parameter in model.named_parameters()
        if "encoder" not in name and parameter.requires_grad
    ]
    groups = [{"params": head_params, "lr": train_cfg["learning_rate"]}]
    if encoder_params:
        groups.insert(0, {
            "params": encoder_params,
            "lr": train_cfg.get("encoder_lr", 1e-5),
        })
    return torch.optim.AdamW(groups, weight_decay=train_cfg["weight_decay"])


def _proto_loss(model: torch.nn.Module, config: dict, num_metric_classes: int):
    proto_cfg = (config["training"].get("loss", {}) or {}).get("proto", {}) or {}
    if not bool(proto_cfg.get("enabled", False)):
        return None, 0.0
    competition_known = int(config.get("model", {}).get("competition_num_known", 446))
    scope = str(proto_cfg.get("scope", "metric")).lower().strip()
    num_classes = competition_known if scope == "known" else num_metric_classes
    embedding_dim = getattr(model.head_speaker, "embedding_dim", None)
    if embedding_dim is None:
        embedding_dim = model.encoder.output_dim * model.pooling.output_multiplier
    criterion = PrototypicalLoss(
        num_classes=num_classes,
        embedding_dim=int(embedding_dim),
        scale=float(proto_cfg.get("scale", 30.0)),
        margin=float(proto_cfg.get("margin", 0.2)),
        decay=float(proto_cfg.get("decay", 0.9)),
    ).cuda()
    return criterion, float(proto_cfg.get("weight", 0.1))


def _training_view(batch: tuple[Any, Any]) -> torch.Tensor:
    """Return the supervised tensor from ordinary or paired loader batches."""
    views = batch[0]
    if isinstance(views, dict):
        if set(views) != {"augmented", "clean"}:
            raise ValueError(
                "Paired probe batch must contain exactly 'augmented' and 'clean'"
            )
        views = views["augmented"]
    if not isinstance(views, torch.Tensor):
        raise TypeError("Probe batch view must be a tensor or paired-view dict")
    return views


def _consistency_weight(config: dict) -> float:
    consistency = (
        ((config.get("training", {}).get("loss", {}) or {})
         .get("consistency", {}) or {})
    )
    return (
        float(consistency.get("weight", 0.0))
        if bool(consistency.get("enabled", False))
        else 0.0
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--candidates", nargs="+", type=int,
                        default=[16, 24, 32, 40, 48, 64])
    parser.add_argument("--encoder-mode", choices=["frozen", "configured"],
                        default="configured")
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--timed-steps", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--headroom-fraction", type=float, default=0.10)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the training batch probe")
    if not 0.0 <= args.headroom_fraction < 1.0:
        raise SystemExit("--headroom-fraction must be in [0, 1)")

    base_config = load_profile(args.profile)
    seed_everything(
        base_config["training"].get("seed", 42),
        deterministic=base_config["training"].get("deterministic_algorithms", True),
    )
    device = torch.device("cuda")

    # Build the model once. DataLoaders are rebuilt per candidate because the
    # balanced sampler's batch boundary is part of what is being measured.
    initial = copy.deepcopy(base_config)
    active_name = initial["hardware"]["mode"]
    initial["hardware"]["profiles"][active_name]["batch_size"] = args.candidates[0]
    initial["hardware"]["profiles"][active_name]["num_workers"] = args.num_workers
    initial_loader, initial_val_loader, class_map = get_dataloaders(initial)
    del initial_loader, initial_val_loader
    num_metric_classes = len(class_map) - 1
    model = create_model_from_config(
        base_config, num_known_speakers=num_metric_classes,
    ).to(device)
    if args.encoder_mode == "frozen":
        model.encoder.freeze()
    else:
        apply_encoder_finetune_mode(model, base_config)

    competition_known = int(base_config.get("model", {}).get("competition_num_known", 446))
    criterion = build_criterion(
        base_config["training"],
        use_ood=ood_head_enabled(base_config),
        competition_known_count=competition_known,
        speaker_target_scope=getattr(model, "speaker_target_scope", "metric"),
    )
    proto_criterion, proto_weight = _proto_loss(
        model, base_config, num_metric_classes,
    )
    total_vram_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    rows: list[dict[str, Any]] = []

    for batch_size in sorted(set(args.candidates)):
        config = copy.deepcopy(base_config)
        active_name = config["hardware"]["mode"]
        profile = config["hardware"]["profiles"][active_name]
        profile["batch_size"] = int(batch_size)
        profile["num_workers"] = int(args.num_workers)
        row: dict[str, Any] = {"batch_size": int(batch_size)}
        train_loader = None
        optimizer = None
        batch = None
        scaler = None
        try:
            train_loader, _, _ = get_dataloaders(config)
            batch = next(iter(train_loader))
            supervised_view = _training_view(batch)
            actual_batch = int(supervised_view.shape[0])
            num_windows = (
                int(supervised_view.shape[1])
                if supervised_view.dim() == 4 else 1
            )
            consistency_weight = _consistency_weight(config)
            optimizer = _optimizer(model, config)
            hw_profile = get_active_profile(config)
            autocast_fn, scaler = build_amp(
                amp_enabled=hw_profile["mixed_precision"],
                amp_dtype=config["training"].get("amp_dtype", "fp16"),
                device=device,
            )
            one_batch_loader = [batch]
            torch.cuda.empty_cache()
            for _ in range(max(0, args.warmup_steps)):
                train_epoch(
                    model, one_batch_loader, optimizer, criterion, scaler, device,
                    config["training"]["max_grad_norm"],
                    ood_grad_norm=config["training"].get("ood_grad_norm", 1.0),
                    autocast_fn=autocast_fn,
                    proto_criterion=proto_criterion,
                    proto_weight=proto_weight,
                    consistency_weight=consistency_weight,
                )
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            for _ in range(max(1, args.timed_steps)):
                train_epoch(
                    model, one_batch_loader, optimizer, criterion, scaler, device,
                    config["training"]["max_grad_norm"],
                    ood_grad_norm=config["training"].get("ood_grad_norm", 1.0),
                    autocast_fn=autocast_fn,
                    proto_criterion=proto_criterion,
                    proto_weight=proto_weight,
                    consistency_weight=consistency_weight,
                )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            files = actual_batch * max(1, args.timed_steps)
            row.update({
                "status": "ok",
                "actual_batch_size": actual_batch,
                "num_windows": num_windows,
                "elapsed_seconds": elapsed,
                "files_per_second": files / elapsed,
                "windows_per_second": files * num_windows / elapsed,
                "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
                "reserved_vram_gib": torch.cuda.max_memory_reserved() / 2**30,
            })
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()
            if not is_oom:
                raise
            row.update({"status": "oom", "error_type": type(exc).__name__})
        finally:
            model.zero_grad(set_to_none=True)
            del optimizer, train_loader, batch, scaler
            gc.collect()
            torch.cuda.empty_cache()
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    recommended = select_recommended_batch(
        rows, total_vram_gib, args.headroom_fraction,
    )
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "encoder_mode": args.encoder_mode,
        "gpu": torch.cuda.get_device_name(0),
        "total_vram_gib": total_vram_gib,
        "headroom_fraction": args.headroom_fraction,
        "results": rows,
        "recommended_batch_size": recommended,
        "warning": (
            "This is an operational throughput recommendation. Commit a derived "
            "profile and account for changed optimizer-step frequency before training."
        ),
    }
    output = Path(args.output) if args.output else (
        ROOT / "data" / "experiments" / "batch_probe" /
        f"{args.profile}_{args.encoder_mode}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved: {output}")
    return 0 if recommended is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
