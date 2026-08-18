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

    Keeps a float32 shadow copy of every trainable parameter and updates it
    after each optimizer step. The EMA weights are the ones saved into the
    best checkpoint (they are more stable and typically generalize better —
    an almost-free robustness win for few-shot fine-tuning).

    ``state_dict(model)`` returns a copy of the model's state_dict with the
    EMA shadow substituted for trainable parameters; frozen encoder weights
    (``requires_grad=False``) pass through unchanged.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self._names: list = []
        self._shadow: list = []
        for name, p in model.named_parameters():
            if p.requires_grad:
                self._names.append(name)
                self._shadow.append(p.detach().clone().float())

    @property
    def enabled(self) -> bool:
        return bool(self._names)

    def update(self, model: torch.nn.Module) -> None:
        """Step the shadow toward the current weights (call after optimizer.step)."""
        trainable = [p for p in model.parameters() if p.requires_grad]
        with torch.no_grad():
            for shadow, p in zip(self._shadow, trainable):
                shadow.mul_(self.decay).add_(p.data.float(), alpha=1.0 - self.decay)

    def state_dict(self, model: torch.nn.Module) -> dict:
        """Model state_dict with EMA weights substituted for trainable params."""
        sd = model.state_dict()
        trainable = [p for p in model.parameters() if p.requires_grad]
        for name, shadow, p in zip(self._names, self._shadow, trainable):
            sd[name] = shadow.to(p.dtype)
        return sd

    def extend(self, model: torch.nn.Module) -> None:
        """Add fresh shadows for params that became trainable after construction.

        Used by the progressive-unfreezing schedule: the encoder is frozen while
        EMA is built (so its params are skipped), then becomes trainable at the
        transition epoch. This appends shadows for those params in parameter
        order, preserving the existing head shadows untouched.
        """
        tracked = set(self._names)
        new_names: list = []
        new_shadow: list = []
        for name, p in model.named_parameters():
            if p.requires_grad and name not in tracked:
                new_names.append(name)
                new_shadow.append(p.detach().clone().float())
        self._names.extend(new_names)
        self._shadow.extend(new_shadow)


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
        then cosine anneal to ``learning_rate * min_lr_ratio``.
      - ``"cosine_warm_restarts"``: the legacy 3-epoch warmup + warm-restarts
        (kept for backward compatibility with existing checkpoints).

    ``warmup_ratio`` (0..1) replaces the old dead ``warmup_steps`` key.
    """
    schedule = str(train_cfg.get("schedule", "cosine")).lower().strip()
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.0))
    min_lr_ratio = float(train_cfg.get("min_lr_ratio", 0.0))
    head_lr = float(train_cfg.get("learning_rate", 1e-4))
    eta_min = head_lr * min_lr_ratio

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
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, num_epochs - warmup_epochs), eta_min=eta_min)
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, num_epochs), eta_min=eta_min)


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
