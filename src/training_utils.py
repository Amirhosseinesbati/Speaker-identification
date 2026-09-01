"""
Shared training utilities for Phase 2 (Full Fine-Tune + per-window loss).

These are used by BOTH training loops — ``src.train`` (standalone) and
``src.pipelines.steps`` (ZenML) — so the two paths can never diverge again
(root cause R6). Everything here is config-driven and dependency-light.

Contents:
    EMA             — exponential moving average of trainable weights (SWA-lite)
    build_scheduler — warmup-ratio + cosine / cosine-warm-restarts
    build_amp       — fp16 / bf16 / off mixed-precision context + GradScaler
    flatten_windows — per-window training: (B, W, 1, T) -> (B*W, 1, T) + label repeat
"""

from __future__ import annotations

import os
import random
import math
from contextlib import contextmanager
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch.amp import GradScaler


def seed_everything(seed: int, deterministic: bool = True) -> dict:
    """Seed every training RNG and record the effective reproducibility policy.

    ``data.split.seed`` only controls which files land in each fold.  Model
    initialisation, random crops, augmentation and DataLoader workers use the
    process RNGs, so they must be seeded separately for controlled experiments.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    deterministic = bool(deterministic)
    if deterministic:
        # Required by deterministic CUDA matrix multiplications on supported
        # toolchains.  setdefault preserves an explicit operator choice.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    return {
        "seed": seed,
        "deterministic_algorithms": deterministic,
        "cudnn_benchmark": False,
        "cudnn_deterministic": deterministic,
    }


def capture_rng_state() -> dict:
    """Capture every RNG that can affect the next training epoch.

    DataLoader workers are not persistent in the current pipeline. Their next
    epoch seeds are derived from Torch's CPU RNG when the iterator is created,
    so restoring the CPU state also restores their future base-seed stream.
    """
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def restore_rng_state(state: object) -> dict:
    """Restore a captured RNG payload and return an auditable receipt.

    ``None`` is accepted for legacy checkpoints and deliberately reports a
    reseeded branch. A present but malformed payload is a hard provenance
    failure rather than a silent partial restore.
    """
    if state is None:
        return {
            "restored": False,
            "python_restored": False,
            "numpy_restored": False,
            "torch_cpu_restored": False,
            "torch_cuda_restored": False,
            "policy": "reseeded_branch_from_training_seed",
        }
    if not isinstance(state, dict):
        raise ValueError("Checkpoint rng_state must be a mapping")
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    missing = sorted(required - set(state))
    if missing:
        raise ValueError("Checkpoint rng_state missing: " + ", ".join(missing))

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])

    cuda_restored = False
    if torch.cuda.is_available():
        cuda_states = state["torch_cuda"]
        if not isinstance(cuda_states, (list, tuple)):
            raise ValueError("Checkpoint rng_state lacks CUDA RNG states")
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError(
                "Checkpoint CUDA RNG device-count mismatch: "
                f"checkpoint={len(cuda_states)}, current={torch.cuda.device_count()}"
            )
        torch.cuda.set_rng_state_all(list(cuda_states))
        cuda_restored = True

    return {
        "restored": True,
        "python_restored": True,
        "numpy_restored": True,
        "torch_cpu_restored": True,
        "torch_cuda_restored": cuda_restored,
        "policy": "restored_checkpoint_rng_state",
    }


# ═══════════════════════════════════════════════════════════
#  Per-window training helper (root cause R2 fix)
# ═══════════════════════════════════════════════════════════

def flatten_windows(
    waveforms: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Flatten a multi-window batch into independent window-level samples.

    ``SpeakerDataset`` returns ``(B, W, 1, T)`` (``W`` random crops per file).
    The OLD bag-level path averaged the ``W`` logits BEFORE the loss, so each
    file contributed a single gradient signal (R2 — the data-multiplication
    factor was thrown away). This flattens to ``(B*W, 1, T)`` and repeats the
    file label ``W`` times, so the loss is computed **per window** (standard
    speaker-verification per-window training).

    Single-window batches ``(B, 1, T)`` pass through unchanged.
    """
    if waveforms.dim() == 4:
        B, W = waveforms.shape[0], waveforms.shape[1]
        flat = waveforms.reshape(B * W, *waveforms.shape[2:])
        if labels is not None:
            labels = labels.repeat_interleave(W)
        return flat, labels
    return waveforms, labels


# ═══════════════════════════════════════════════════════════
#  Exponential Moving Average (EMA) of model weights
# ═══════════════════════════════════════════════════════════

class EMA:
    """Exponential moving average of trainable parameters (``decay=0.999``).

    Keeps a float32 shadow copy of every trainable parameter, keyed by
    parameter name, and updates it after each optimizer step. The EMA weights
    are the ones saved into the best checkpoint (they are more stable and
    typically generalize better — an almost-free robustness win for few-shot
    fine-tuning).

    Shadows are looked up by name (not by list position), so ``extend`` can
    add newly-unfrozen encoder params out of module order without corrupting
    the pairing between shadows and parameters.

    ``state_dict(model)`` returns a copy of the model's state_dict with the
    EMA shadow substituted for every tracked parameter; parameters never
    added to the EMA (e.g. an encoder that stays frozen) pass through
    unchanged.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self._shadow: dict = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self._shadow[name] = p.detach().clone().float()

    @property
    def enabled(self) -> bool:
        return bool(self._shadow)

    def update(self, model: torch.nn.Module) -> None:
        """Step the shadows toward the current weights (call after optimizer.step)."""
        with torch.no_grad():
            for name, p in model.named_parameters():
                shadow = self._shadow.get(name)
                if shadow is not None:
                    shadow.mul_(self.decay).add_(p.data.float(), alpha=1.0 - self.decay)

    def state_dict(self, model: torch.nn.Module) -> dict:
        """Model state_dict with EMA weights substituted for tracked params."""
        sd = model.state_dict()
        for name, shadow in self._shadow.items():
            if name in sd:
                sd[name] = shadow.to(sd[name].dtype)
        return sd

    @contextmanager
    def average_parameters(self, model: torch.nn.Module):
        """Temporarily evaluate ``model`` with EMA parameters, then restore raw.

        Validation used to score raw weights but save EMA weights under the same
        metric.  This context manager makes the two variants independently
        measurable without perturbing optimizer state or the training path.
        """
        parameters = dict(model.named_parameters())
        backup = {}
        with torch.no_grad():
            for name, shadow in self._shadow.items():
                parameter = parameters.get(name)
                if parameter is None:
                    continue
                backup[name] = parameter.detach().clone()
                parameter.copy_(shadow.to(device=parameter.device,
                                          dtype=parameter.dtype))
        try:
            yield model
        finally:
            with torch.no_grad():
                for name, value in backup.items():
                    parameters[name].copy_(value)

    def extend(self, model: torch.nn.Module) -> None:
        """Add fresh shadows for params that became trainable after construction.

        Used by the progressive-unfreezing schedule: the encoder is frozen while
        EMA is built (so its params are skipped), then becomes trainable at the
        transition epoch. This registers shadows for those params, keeping the
        existing head shadows untouched.
        """
        for name, p in model.named_parameters():
            if p.requires_grad and name not in self._shadow:
                self._shadow[name] = p.detach().clone().float()


# ═══════════════════════════════════════════════════════════
#  Config-driven learning-rate scheduler
# ═══════════════════════════════════════════════════════════

def build_scheduler(
    optimizer: torch.optim.Optimizer,
    train_cfg: dict,
    num_epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Build a scheduler from ``training`` config.

    Supported ``training.schedule`` values:
      - ``"cosine"`` (default): linear warmup over ``warmup_ratio`` of epochs,
        then cosine anneal; each param group anneals to its own
        ``base_lr * min_lr_ratio``.
      - ``"exponential"``: exponentially anneal every parameter group from
        its own base LR to ``base_lr * min_lr_ratio`` at ``num_epochs``.  This
        is the schedule used by the published ALMFT fine-tuning protocol.
      - ``"cosine_warm_restarts"``: the legacy 3-epoch warmup + warm-restarts
        (kept for backward compatibility with existing checkpoints).

    ``warmup_ratio`` (0..1) replaces the old dead ``warmup_steps`` key.
    """
    schedule = str(train_cfg.get("schedule", "cosine")).lower().strip()
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.0))
    min_lr_ratio = float(train_cfg.get("min_lr_ratio", 0.0))

    if schedule == "exponential":
        if warmup_ratio != 0.0:
            raise ValueError(
                "The exponential schedule does not use warmup; set "
                "training.warmup_ratio=0."
            )
        if not (0.0 < min_lr_ratio <= 1.0):
            raise ValueError(
                "The exponential schedule requires 0 < min_lr_ratio <= 1."
            )
        gamma = min_lr_ratio ** (1.0 / max(1, int(num_epochs)))
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    if schedule == "cosine_warm_restarts":
        warmup_epochs = max(1, int(train_cfg.get("warmup_epochs", 3)))
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_epochs)
        restarts = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=int(train_cfg.get("t0", 10)), T_mult=1)
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, restarts], milestones=[warmup_epochs])

    # Default: cosine with optional warmup.
    warmup_epochs = max(0, int(round(num_epochs * warmup_ratio)))
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_epochs)
        cosine = _cosine_to_min_lr(
            optimizer, T_max=max(1, num_epochs - warmup_epochs),
            min_lr_ratio=min_lr_ratio)
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

    return _cosine_to_min_lr(
        optimizer, T_max=max(1, num_epochs), min_lr_ratio=min_lr_ratio)


def _cosine_to_min_lr(
    optimizer: torch.optim.Optimizer,
    T_max: int,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Cosine anneal with a PER-GROUP floor at ``base_lr * min_lr_ratio``.

    ``CosineAnnealingLR`` takes one absolute ``eta_min`` for every param
    group. The scheduler was building that floor from the HEAD LR only, so an
    encoder group whose ``encoder_lr`` sat below ``learning_rate * min_lr_ratio``
    annealed UPWARD toward the floor (inverted cosine). Scaling the whole
    schedule by a per-epoch factor keeps each group proportional to its own
    base LR and is mathematically identical to ``CosineAnnealingLR`` when a
    single group's floor equals ``base_lr * min_lr_ratio``.
    """

    def lr_lambda(last_epoch: int) -> float:
        factor = (1.0 + math.cos(math.pi * last_epoch / T_max)) / 2.0
        return min_lr_ratio + (1.0 - min_lr_ratio) * factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ═══════════════════════════════════════════════════════════
#  Mixed-precision helper (fp16 / bf16 / off)
# ═══════════════════════════════════════════════════════════

def build_amp(
    amp_enabled: bool,
    amp_dtype: str,
    device: torch.device,
) -> Tuple[Callable[[], object], GradScaler]:
    """Return ``(autocast_fn, scaler)`` for the training loop.

    - ``amp_dtype="bf16"``: bf16 autocast with NO GradScaler (bf16 keeps the
      fp32 exponent range, so gradient scaling is unnecessary and WavLM-Large
      no longer NaNs — root cause R5's fp16 instability).
    - ``amp_dtype="fp16"``: classic fp16 autocast + GradScaler (legacy default).
    - ``amp_enabled=False``: no-op autocast + disabled scaler.

    ``autocast_fn()`` returns a context manager to use as ``with autocast_fn():``.
    """
    use_amp = bool(amp_enabled) and device.type == "cuda"
    dtype_name = str(amp_dtype).lower()
    grad_scaler_device = "cuda" if device.type == "cuda" else "cpu"

    if use_amp and dtype_name == "bf16":
        def autocast_fn():
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)
        return autocast_fn, GradScaler(grad_scaler_device, enabled=False)

    if use_amp:
        def autocast_fn():
            return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
        return autocast_fn, GradScaler(grad_scaler_device, enabled=True)

    def autocast_fn():
        return torch.autocast(device_type=device.type, enabled=False)
    return autocast_fn, GradScaler(grad_scaler_device, enabled=False)


# ═══════════════════════════════════════════════════════════
#  Progressive unfreezing (two-phase fine-tuning schedule)
# ═══════════════════════════════════════════════════════════

def encoder_will_train(config: dict) -> bool:
    """True if the configured fine-tune mode leaves encoder params trainable.

    Progressive unfreezing is only meaningful for encoders whose whole trunk can
    be frozen (ecapa / campp / eres2net / titanet). WavLM keeps its transformer
    trainable even when ``freeze_feature_extractor`` is on, so it returns False.
    """
    enc_type = str(config.get("model", {}).get("encoder_type", "")).lower().strip()
    if enc_type == "wavlm":
        return False
    enc_cfg = (config.get("model", {}).get("encoder_config", {}) or {}).get(enc_type, {}) or {}
    if enc_type == "ecapa" and str(
            enc_cfg.get("adapter_mode", "none")).lower().strip() != "none":
        return True
    return not bool(enc_cfg.get("freeze_encoder", True))


def apply_encoder_finetune_mode(model: torch.nn.Module, config: dict) -> None:
    """Restore the configured fine-tune mode on ``model.encoder`` (idempotent).

    Used by the progressive-unfreezing schedule (``training.freeze_epochs``): the
    encoder is forced frozen during the warm-up phase, then this restores the
    configured mode (frozen / partial last-N / full) at the transition epoch.
    """
    enc_type = str(config.get("model", {}).get("encoder_type", "")).lower().strip()
    enc_cfg = (config.get("model", {}).get("encoder_config", {}) or {}).get(enc_type, {}) or {}
    encoder = model.encoder

    adapter_mode = str(enc_cfg.get("adapter_mode", "none")).lower().strip()
    if enc_type == "ecapa" and adapter_mode == "se_bn":
        encoder.enable_se_bn_adapter()
        return

    freeze_key = "freeze_feature_extractor" if enc_type == "wavlm" else "freeze_encoder"
    if enc_cfg.get(freeze_key, True):
        encoder.freeze()
        return

    n = int(enc_cfg.get("unfreeze_last_n_blocks", 0) or 0)
    if n > 0 and hasattr(encoder, "unfreeze_last_n_blocks"):
        encoder.unfreeze_last_n_blocks(n)
    else:
        encoder.unfreeze()
