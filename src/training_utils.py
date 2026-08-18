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

import math
from typing import Callable, Optional, Tuple

import torch
from torch.amp import GradScaler


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
      - ``"cosine_warm_restarts"``: the legacy 3-epoch warmup + warm-restarts
        (kept for backward compatibility with existing checkpoints).

    ``warmup_ratio`` (0..1) replaces the old dead ``warmup_steps`` key.
    """
    schedule = str(train_cfg.get("schedule", "cosine")).lower().strip()
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.0))
    min_lr_ratio = float(train_cfg.get("min_lr_ratio", 0.0))

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

    freeze_key = "freeze_feature_extractor" if enc_type == "wavlm" else "freeze_encoder"
    if enc_cfg.get(freeze_key, True):
        encoder.freeze()
        return

    n = int(enc_cfg.get("unfreeze_last_n_blocks", 0) or 0)
    if n > 0 and hasattr(encoder, "unfreeze_last_n_blocks"):
        encoder.unfreeze_last_n_blocks(n)
    else:
        encoder.unfreeze()
