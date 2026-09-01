"""Measure exact multi-window validation throughput and VRAM by batch size."""

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.batch_probe import select_recommended_batch  # noqa: E402
from src.data_pipeline import get_dataloaders  # noqa: E402
from src.experiment_config import load_profile  # noqa: E402
from src.model_factory import create_model_from_config  # noqa: E402
from src.train import forward_multi_window_evaluation  # noqa: E402
from src.training_utils import apply_encoder_finetune_mode, seed_everything  # noqa: E402


def _model_invariants(model: torch.nn.Module) -> dict[str, Any]:
    encoder = model.encoder
    wavlm = getattr(encoder, "wavlm", None)
    layer_weights = getattr(encoder, "layer_weights", None)
    if wavlm is None or layer_weights is None:
        raise RuntimeError(
            "evaluation probe requires a multilayer WavLM encoder"
        )
    layer_adapters = getattr(encoder, "layer_adapters", None)
    adapter_parameters = (
        list(layer_adapters.parameters())
        if layer_adapters is not None else []
    )
    transformer_layer_norm_parameters = [
        parameter
        for name, parameter in wavlm.named_parameters()
        if name.startswith("encoder.layers.") and "layer_norm" in name
    ]
    wavlm_trainable_parameters = sum(
        parameter.numel() for parameter in wavlm.parameters()
        if parameter.requires_grad
    )
    transformer_layer_norm_trainable_parameters = sum(
        parameter.numel()
        for parameter in transformer_layer_norm_parameters
        if parameter.requires_grad
    )
    return {
        "layer_aggregation": str(
            getattr(encoder, "layer_aggregation", "unknown")
        ),
        "wavlm_parameters": sum(parameter.numel() for parameter in wavlm.parameters()),
        "wavlm_trainable_parameters": wavlm_trainable_parameters,
        "transformer_layer_norm_parameters": sum(
            parameter.numel()
            for parameter in transformer_layer_norm_parameters
        ),
        "transformer_layer_norm_trainable_parameters": (
            transformer_layer_norm_trainable_parameters
        ),
        "wavlm_other_trainable_parameters": (
            wavlm_trainable_parameters
            - transformer_layer_norm_trainable_parameters
        ),
        "layer_weight_count": int(layer_weights.numel()),
        "layer_weight_trainable": bool(layer_weights.requires_grad),
        "layer_adapter_count": (
            len(layer_adapters) if layer_adapters is not None else 0
        ),
        "layer_adapter_parameters": sum(
            parameter.numel() for parameter in adapter_parameters
        ),
        "layer_adapter_trainable_parameters": sum(
            parameter.numel() for parameter in adapter_parameters
            if parameter.requires_grad
        ),
    }


def _validate_model_invariants(model: torch.nn.Module, invariants: dict) -> None:
    encoder = model.encoder
    aggregation = invariants["layer_aggregation"]
    expected_transformer_layers = int(encoder.wavlm.config.num_hidden_layers)
    if not invariants["layer_weight_trainable"]:
        raise RuntimeError("WavLM multilayer weights are unexpectedly frozen")

    if aggregation == "weighted_sum":
        if invariants["wavlm_trainable_parameters"] != 0:
            raise RuntimeError("WavLM backbone is unexpectedly trainable")
        if invariants["layer_weight_count"] != expected_transformer_layers + 1:
            raise RuntimeError("weighted-sum layer parameter invariant failed")
        if invariants["layer_adapter_parameters"] != 0:
            raise RuntimeError("weighted-sum encoder unexpectedly has L-adapters")
        return

    if aggregation == "layer_adapter":
        if invariants["layer_weight_count"] != expected_transformer_layers:
            raise RuntimeError("L-adapter layer-weight invariant failed")
        if invariants["layer_adapter_count"] != expected_transformer_layers:
            raise RuntimeError("L-adapter transformer-layer coverage failed")
        if invariants["layer_adapter_parameters"] <= 0:
            raise RuntimeError("L-adapter parameters are missing")
        if (invariants["layer_adapter_trainable_parameters"] !=
                invariants["layer_adapter_parameters"]):
            raise RuntimeError("some L-adapter parameters are unexpectedly frozen")
        if invariants["wavlm_other_trainable_parameters"] != 0:
            raise RuntimeError(
                "WavLM parameters outside transformer LayerNorm are trainable"
            )
        tune_layer_norms = bool(
            getattr(
                encoder,
                "layer_adapter_tune_backbone_layer_norms",
                False,
            )
        )
        observed_layer_norms = invariants[
            "transformer_layer_norm_trainable_parameters"
        ]
        if tune_layer_norms and observed_layer_norms <= 0:
            raise RuntimeError("transformer LayerNorm tuning is missing")
        if not tune_layer_norms and observed_layer_norms != 0:
            raise RuntimeError("transformer LayerNorm is unexpectedly trainable")
        return

    raise RuntimeError(f"unsupported WavLM layer aggregation: {aggregation!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--candidates", nargs="+", type=int, required=True)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--timed-batches", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--headroom-fraction", type=float, default=0.10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the evaluation batch probe")

    base = load_profile(args.profile)
    seed_everything(
        int(base["training"].get("seed", 42)),
        deterministic=bool(base["training"].get("deterministic_algorithms", True)),
    )
    device = torch.device("cuda")
    initial = copy.deepcopy(base)
    active = initial["hardware"]["mode"]
    initial["hardware"]["profiles"][active]["batch_size"] = min(args.candidates)
    initial["hardware"]["profiles"][active]["num_workers"] = args.num_workers
    train_loader, val_loader, class_map = get_dataloaders(initial)
    del train_loader, val_loader
    model = create_model_from_config(base, num_known_speakers=len(class_map) - 1)
    apply_encoder_finetune_mode(model, base)
    invariants = _model_invariants(model)
    _validate_model_invariants(model, invariants)
    model = model.to(device).eval()
    total_vram_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    rows: list[dict[str, Any]] = []

    for batch_size in sorted(set(args.candidates)):
        config = copy.deepcopy(base)
        active = config["hardware"]["mode"]
        config["hardware"]["profiles"][active]["batch_size"] = batch_size
        config["hardware"]["profiles"][active]["num_workers"] = args.num_workers
        val_loader = None
        batches = None
        row: dict[str, Any] = {"batch_size": batch_size}
        try:
            train_loader, val_loader, _ = get_dataloaders(config)
            del train_loader
            iterator = iter(val_loader)
            count = max(1, args.warmup_batches) + max(1, args.timed_batches)
            batches = [next(iterator) for _ in range(count)]
            with torch.no_grad():
                for waveforms, _ in batches[: args.warmup_batches]:
                    forward_multi_window_evaluation(
                        model, waveforms.to(device, non_blocking=True)
                    )
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                files = 0
                windows = 0
                for waveforms, _ in batches[args.warmup_batches :]:
                    files += int(waveforms.shape[0])
                    windows += int(waveforms.shape[0]) * (
                        int(waveforms.shape[1]) if waveforms.dim() == 4 else 1
                    )
                    forward_multi_window_evaluation(
                        model, waveforms.to(device, non_blocking=True)
                    )
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
            row.update({
                "status": "ok",
                "files": files,
                "windows": windows,
                "elapsed_seconds": elapsed,
                "files_per_second": files / elapsed,
                "windows_per_second": windows / elapsed,
                "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
                "reserved_vram_gib": torch.cuda.max_memory_reserved() / 2**30,
            })
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or (
                "out of memory" in str(exc).lower()
            )
            if not is_oom:
                raise
            row.update({"status": "oom", "error_type": type(exc).__name__})
        finally:
            del val_loader, batches
            gc.collect()
            torch.cuda.empty_cache()
        rows.append(row)
        print(json.dumps(row), flush=True)

    recommended = select_recommended_batch(
        rows, total_vram_gib, args.headroom_fraction
    )
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "path": "exact_forward_multi_window_evaluation_without_amp",
        "gpu": torch.cuda.get_device_name(0),
        "total_vram_gib": total_vram_gib,
        "headroom_fraction": args.headroom_fraction,
        "model_invariants": invariants,
        "results": rows,
        "recommended_batch_size": recommended,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if recommended is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
