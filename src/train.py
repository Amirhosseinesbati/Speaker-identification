"""
Phase 3: Training Engine for Open-Set Speaker Identification.

Key features:
- Two-Part Loss: BCE (OOD) + CrossEntropy (Speaker, masked for unknown)
- Automatic Mixed Precision (AMP) for VRAM efficiency
- OOD Detection Accuracy + Known Speaker Accuracy metrics
- Best checkpoint saving based on Validation Loss
"""

import os
import sys
import time
import warnings
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

# Add parent to path for direct execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Windows cp1252 fix: force UTF-8 stdio so emoji output never crashes.
from src.cli_utils import setup_utf8_stdio
setup_utf8_stdio()

from src.data_pipeline import load_config, get_active_profile, get_dataloaders
from src.model_factory import create_model_from_config
from src.metrics import evaluate_macro_f1
from src.training_utils import (
    EMA,
    build_amp,
    build_scheduler,
)

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────

def setup_device(config: dict) -> torch.device:
    """Set up device based on config and availability."""
    hw_profile = get_active_profile(config)
    preferred = hw_profile.get("device", "cuda")
    if preferred == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  🖥️  GPU: {torch.cuda.get_device_name(0)}")
        print(f"     VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print(f"  🖥️  Device: CPU")
    return device


# ─────────────────────────────────────────────────────────
#  Focal Loss
# ─────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    FL(p_t) = -(1 - p_t)^γ * log(p_t)

    where p_t is the model's estimated probability for the target class.
    The modulating factor (1 - p_t)^γ down-weights easy examples and
    focuses training on hard, misclassified samples.

    Reference: Lin et al., "Focal Loss for Dense Object Detection" (2017)

    Args:
        gamma: Focusing parameter. γ=0 → standard cross-entropy.
               Recommended: 2.0 for strong imbalance, 1.0 for mild.
        ignore_index: Target value to ignore (default: -100)
        reduction: 'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        gamma: float = 2.0,
        ignore_index: int = -100,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: torch.Tensor,   # (batch, num_classes)
        targets: torch.Tensor,  # (batch,)
    ) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits:  Raw logits from the model (before softmax)
            targets: Integer class labels (ignored if == ignore_index)

        Returns:
            Scalar loss tensor
        """
        # Label smoothing: convert hard targets to soft targets
        if self.label_smoothing > 0 and self.training:
            num_classes = logits.size(-1)
            with torch.no_grad():
                # Mask ignored targets before scatter (avoid out-of-bounds)
                if self.ignore_index is not None:
                    ignore_mask = targets == self.ignore_index
                    safe_targets = targets.clone()
                    safe_targets[ignore_mask] = 0  # temporary valid index
                else:
                    ignore_mask = torch.zeros_like(targets, dtype=torch.bool)
                    safe_targets = targets

                smooth_targets = torch.full_like(logits, self.label_smoothing / (num_classes - 1))
                smooth_targets.scatter_(1, safe_targets.unsqueeze(1), 1.0 - self.label_smoothing)
                # Zero out ignored positions (they contribute 0 to CE)
                if self.ignore_index is not None:
                    smooth_targets[ignore_mask] = 0.0
            # CE with soft targets: -sum(target * log_softmax)
            log_probs = F.log_softmax(logits, dim=1)
            ce_loss = -(smooth_targets * log_probs).sum(dim=1)
        else:
            # Standard cross-entropy with no reduction
            ce_loss = F.cross_entropy(
                logits, targets,
                reduction="none",
                ignore_index=self.ignore_index,
            )

        # p_t = exp(-CE_loss)
        pt = torch.exp(-ce_loss)

        # Focal scaling factor
        focal_weight = (1.0 - pt) ** self.gamma

        # Apply focal weight
        focal_loss = focal_weight * ce_loss

        if self.reduction == "mean":
            # Average over valid (non-ignored) samples
            valid_mask = targets != self.ignore_index
            if valid_mask.sum() == 0:
                return torch.tensor(0.0, device=logits.device, requires_grad=True)
            return focal_loss[valid_mask].mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


# ─────────────────────────────────────────────────────────
#  Two-Part Loss Function
# ─────────────────────────────────────────────────────────

class TwoPartLoss(nn.Module):
    """
    Two-part loss for the two-headed architecture:

    Loss 1 (OOD): BCEWithLogitsLoss — target=1 if label==0 (unknown), else 0.
    Loss 2 (Speaker): CrossEntropyLoss or FocalLoss — ONLY for known samples.
                       Unknown samples are masked via ignore_index.

    total_loss = loss_ood + loss_speaker
    """

    def __init__(
        self,
        ignore_index: int = -100,
        use_focal: bool = True,
        focal_gamma: float = 2.0,
        ood_weight: float = 1.0,
        speaker_weight: float = 1.0,
        label_smoothing: float = 0.0,
        ood_pos_weight: float = 1.0,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.ood_weight = ood_weight
        self.speaker_weight = speaker_weight

        self.bce_loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(ood_pos_weight, dtype=torch.float32)
        )

        if use_focal:
            self.ce_loss = FocalLoss(
                gamma=focal_gamma, ignore_index=ignore_index,
                label_smoothing=label_smoothing,
            )
            print(f"  🎯 Speaker loss: FocalLoss (γ={focal_gamma}, smoothing={label_smoothing})")
        else:
            self.ce_loss = nn.CrossEntropyLoss(
                ignore_index=ignore_index, label_smoothing=label_smoothing,
            )
            print(f"  📊 Speaker loss: CrossEntropyLoss (smoothing={label_smoothing})")

        print(f"  ⚖️  Loss weights: OOD={ood_weight}, Speaker={speaker_weight} "
              f"(OOD BCE pos_weight={ood_pos_weight})")

    def forward(
        self,
        ood_logits: torch.Tensor,     # (batch, 1)
        speaker_logits: torch.Tensor,  # (batch, 446)
        labels: torch.Tensor,          # (batch,)  — 0 for unknown, 1..446 for known
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Returns:
            total_loss: scalar tensor
            loss_components: dict with individual loss values
        """
        batch_size = labels.size(0)

        # ── Loss 1: OOD Detection ──
        # Target: 1 if label == 0 (unknown), else 0
        ood_targets = (labels == 0).float().unsqueeze(1)  # (batch, 1)
        loss_ood = self.bce_loss(ood_logits, ood_targets)

        # ── Loss 2: Known Speaker Classification ──
        # Create masked labels: for unknown samples, set to ignore_index
        speaker_labels = labels.clone()
        speaker_labels[labels == 0] = self.ignore_index

        # Speaker labels are 1-indexed in the dataset, but CrossEntropy expects 0-indexed.
        # So we subtract 1 to map classes 1..446 → 0..445
        speaker_labels_mapped = speaker_labels.clone()
        mask_known = speaker_labels_mapped != self.ignore_index
        speaker_labels_mapped[mask_known] = speaker_labels_mapped[mask_known] - 1

        loss_speaker = self.ce_loss(speaker_logits, speaker_labels_mapped)

        # ── Total Loss (weighted) ──
        total_loss = self.ood_weight * loss_ood + self.speaker_weight * loss_speaker

        loss_dict = {
            "loss_ood": loss_ood.item(),
            "loss_speaker": loss_speaker.item(),
            "loss_total": total_loss.item(),
        }

        return total_loss, loss_dict


# ─────────────────────────────────────────────────────────
#  Config-driven Loss Factory
# ─────────────────────────────────────────────────────────

def build_criterion(train_cfg: dict) -> TwoPartLoss:
    """Build the two-part loss from the ``training`` config (root cause R9/C4).

    Reads an optional ``training.loss`` block:

        training.loss.speaker.type        → "focal" (legacy default) | "ce"
        training.loss.speaker.focal_gamma → γ for focal
        training.loss.speaker.label_smoothing
        training.loss.speaker.weight      → speaker loss weight
        training.loss.ood.weight          → OOD loss weight
        training.loss.ood.pos_weight      → BCE pos_weight

    Falls back to the flat legacy keys (``use_focal``, ``focal_gamma``,
    ``ood_loss_weight``, ``speaker_loss_weight``, ``label_smoothing``,
    ``ood_pos_weight``) so existing configs keep working unchanged.
    """
    loss_cfg = train_cfg.get("loss", {}) or {}
    speaker_cfg = loss_cfg.get("speaker", {}) or {}
    ood_cfg = loss_cfg.get("ood", {}) or {}

    speaker_type = str(speaker_cfg.get("type", train_cfg.get("use_focal", "focal"))).lower()
    use_focal = speaker_type == "focal"

    return TwoPartLoss(
        ignore_index=-100,
        use_focal=use_focal,
        focal_gamma=float(speaker_cfg.get(
            "focal_gamma", train_cfg.get("focal_gamma", 2.0))),
        ood_weight=float(ood_cfg.get(
            "weight", train_cfg.get("ood_loss_weight", 1.0))),
        speaker_weight=float(speaker_cfg.get(
            "weight", train_cfg.get("speaker_loss_weight", 1.0))),
        label_smoothing=float(speaker_cfg.get(
            "label_smoothing", train_cfg.get("label_smoothing", 0.0))),
        ood_pos_weight=float(ood_cfg.get(
            "pos_weight", train_cfg.get("ood_pos_weight", 1.0))),
    )


# ─────────────────────────────────────────────────────────
#  Metric Calculators
# ─────────────────────────────────────────────────────────

def compute_ood_accuracy(
    ood_logits: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """
    Compute OOD detection accuracy:
    - Correct if sigmoid(logit) > 0.5 AND label is 0 (unknown detected as unknown)
    - OR sigmoid(logit) <= 0.5 AND label is not 0 (known detected as known)
    """
    probs = torch.sigmoid(ood_logits).squeeze(1)  # (batch,)
    predictions = (probs > 0.5).long()  # 1 = predicted unknown, 0 = predicted known
    targets = (labels == 0).long()       # 1 = actual unknown, 0 = actual known
    correct = (predictions == targets).sum().item()
    return correct / labels.size(0)


def compute_speaker_accuracy(
    speaker_logits: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[float, int]:
    """
    Compute known speaker accuracy (only for samples with label != 0).
    """
    known_mask = labels != 0
    if known_mask.sum() == 0:
        return 0.0, 0

    known_logits = speaker_logits[known_mask]
    known_labels = labels[known_mask] - 1  # Map 1..446 → 0..445

    predictions = known_logits.argmax(dim=1)
    correct = (predictions == known_labels).sum().item()
    total = known_labels.size(0)

    return correct / total, total


def tune_ood_threshold(
    ood_logits: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Tune a binary OOD threshold on the val set for the checkpoint.

    Mirrors the ZenML evaluate step: sweep 0.1..0.9 by unknown-class binary
    F1 (fallback to the median P(unknown) if the head collapsed). Persisted
    as ckpt['ood_threshold'] so --apply-ood-threshold works from plain
    `python -m src.train` too.
    """
    from sklearn.metrics import f1_score

    ood_probs = torch.sigmoid(ood_logits.squeeze(1)).numpy()
    ood_targets = (labels == 0).numpy().astype(int)
    best_threshold = 0.5
    best_f1 = 0.0
    for thr in np.arange(0.1, 0.9, 0.05):
        preds = (ood_probs > thr).astype(int)
        f1 = f1_score(ood_targets, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thr
    if best_f1 == 0.0:
        best_threshold = float(np.median(ood_probs))
    return float(best_threshold)


# ─────────────────────────────────────────────────────────
#  Multi-Window Forward Helper (TTA)
# ─────────────────────────────────────────────────────────

def forward_multi_window(
    model: nn.Module,
    waveforms: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward pass with multi-window TTA.

    Accepts either:
      - (B, 1, T)   → plain single-window forward
      - (B, W, 1, T) → runs the model **once per window** (loop) and averages
                       the resulting logits over W (window-level TTA, so the
                       full file length is used).

    Looping (instead of one flattened (B*W, 1, T) forward) keeps the peak
    activation tensor at (B, 1, T) rather than (B*W, 1, T) — the plan's
    low-VRAM recommendation for the 6 GB local GPU. The math is identical:
    the mean of per-window logits (verified by an equivalence test).

    Args:
        model:      TwoHeadedSpeakerModel (or anything with forward(w, labels)).
        waveforms:  (B, 1, T) or (B, W, 1, T) raw audio, 16 kHz.
        labels:     Optional (B,) speaker labels (ArcFace margin applied per
                    window when given — equivalent to repeat_interleave(W)).

    Returns:
        ood_logits:     (B, 1)
        speaker_logits: (B, N)
    """
    if waveforms.dim() == 3:
        return model(waveforms, labels=labels)

    B, W = waveforms.shape[0], waveforms.shape[1]
    ood_sum = None
    spk_sum = None
    for w in range(W):
        ood, spk = model(waveforms[:, w], labels=labels)
        ood_sum = ood if ood_sum is None else ood_sum + ood
        spk_sum = spk if spk_sum is None else spk_sum + spk
    return ood_sum / W, spk_sum / W


# ─────────────────────────────────────────────────────────
#  Training Epoch
# ─────────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: TwoPartLoss,
    scaler: GradScaler,
    device: torch.device,
    max_grad_norm: float = 5.0,
    ood_grad_norm: float = 1.0,
    ood_head_attr: str = "head_ood",
    autocast_fn=None,
    ema: Optional[EMA] = None,
) -> Dict[str, float]:
    """
    Train for one epoch with AMP.

    **Per-window loss (root cause R2):** a ``(B, W, 1, T)`` batch is processed
    one window at a time and gradients are accumulated across the ``W`` windows
    (peak activation stays at ``(B, 1, T)``), so the loss is computed per window
    instead of averaging the ``W`` logits first — without the OOM of flattening
    to a single ``(B*W, 1, T)`` forward.

    Returns dict of average metrics.
    """
    if autocast_fn is None:
        def autocast_fn():
            return autocast()

    model.train()
    total_loss = 0.0
    total_ood_acc = 0.0
    total_speaker_acc = 0.0
    num_batches = len(dataloader)

    progress_bar = tqdm(dataloader, desc="  Train", leave=False)
    for step, (waveforms, labels) in enumerate(progress_bar):
        waveforms = waveforms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Per-window training with gradient accumulation (root cause R2).
        # Each window is a (B, 1, T) forward — peak activation stays at B, not
        # B*W — and gradients accumulate across the W windows. The old
        # flatten-to-(B*W,1,T) path OOM'd a 24 GB GPU at 32 files × 4 windows of
        # 8 s under full fine-tune. Scaling each window's loss by 1/W keeps the
        # step mathematically identical to the mean-over-windows loss.
        W = waveforms.shape[1] if waveforms.dim() == 4 else 1

        step_loss = 0.0
        step_ood_acc = 0.0
        step_spk_acc = 0.0
        for w in range(W):
            wf = waveforms[:, w] if W > 1 else waveforms  # (B, 1, T)
            with autocast_fn():
                ood_logits, speaker_logits = model(wf, labels=labels)
                loss, loss_dict = criterion(ood_logits, speaker_logits, labels)

            # Fail loudly on NaN/Inf instead of silently training a broken model
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Loss is NaN/Inf at step {step} window {w} of the current "
                    "epoch. This usually means the encoder is fully unfrozen "
                    "under fp16 AMP (try mixed_precision: false, a lower "
                    "learning_rate/encoder_lr, or freeze_encoder: true / a "
                    "smaller unfreeze_last_n_blocks)."
                )

            # AMP-scaled per-window backward; gradients accumulate over W.
            scaler.scale(loss / W).backward()

            step_loss += loss_dict["loss_total"]
            step_ood_acc += compute_ood_accuracy(ood_logits, labels)
            spk_acc, _ = compute_speaker_accuracy(speaker_logits, labels)
            step_spk_acc += spk_acc

        scaler.unscale_(optimizer)

        # Separate gradient clipping: tighter for OOD head
        ood_params = []
        other_params = []
        for name, param in model.named_parameters():
            if ood_head_attr in name and param.requires_grad:
                ood_params.append(param)
            elif param.requires_grad:
                other_params.append(param)

        if ood_params and ood_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(ood_params, ood_grad_norm)
        if other_params:
            torch.nn.utils.clip_grad_norm_(other_params, max_grad_norm)

        scaler.step(optimizer)
        scaler.update()

        # EMA shadow update (after the optimizer step).
        if ema is not None:
            ema.update(model)

        # Metrics (mean over windows)
        total_loss += step_loss / W
        total_ood_acc += step_ood_acc / W
        total_speaker_acc += step_spk_acc / W

        # Update progress bar
        progress_bar.set_postfix({
            "loss": f"{step_loss / W:.4f}",
            "ood": f"{step_ood_acc / W:.3f}",
            "spk": f"{step_spk_acc / W:.3f}",
        })

    return {
        "loss": total_loss / num_batches,
        "ood_acc": total_ood_acc / num_batches,
        "speaker_acc": total_speaker_acc / num_batches,
    }


# ─────────────────────────────────────────────────────────
#  Validation Epoch
# ─────────────────────────────────────────────────────────

@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: TwoPartLoss,
    device: torch.device,
) -> Dict[str, float]:
    """
    Validate for one epoch. No AMP needed for eval.

    Forward is called WITHOUT labels so the ArcFace margin is never applied at
    eval time (margin would otherwise under-report the honest accuracy).

    Also collects and returns the concatenated logits + labels so callers can
    compute the competition metric (Macro-F1) via src.metrics.evaluate_macro_f1.
    """
    model.eval()
    total_loss = 0.0
    total_ood_acc = 0.0
    total_speaker_acc = 0.0
    num_batches = len(dataloader)

    all_ood_logits, all_speaker_logits, all_labels = [], [], []

    progress_bar = tqdm(dataloader, desc="  Val", leave=False)
    for waveforms, labels in progress_bar:
        waveforms = waveforms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # No labels → no ArcFace margin → honest eval logits
        ood_logits, speaker_logits = forward_multi_window(model, waveforms, labels=None)
        loss, loss_dict = criterion(ood_logits, speaker_logits, labels)

        total_loss += loss_dict["loss_total"]
        total_ood_acc += compute_ood_accuracy(ood_logits, labels)
        speaker_acc, _ = compute_speaker_accuracy(speaker_logits, labels)
        total_speaker_acc += speaker_acc

        all_ood_logits.append(ood_logits.cpu())
        all_speaker_logits.append(speaker_logits.cpu())
        all_labels.append(labels.cpu())

        progress_bar.set_postfix({
            "loss": f"{loss_dict['loss_total']:.4f}",
            "ood": f"{compute_ood_accuracy(ood_logits, labels):.3f}",
            "spk": f"{speaker_acc:.3f}",
        })

    return {
        "loss": total_loss / num_batches,
        "ood_acc": total_ood_acc / num_batches,
        "speaker_acc": total_speaker_acc / num_batches,
        "ood_logits": torch.cat(all_ood_logits),
        "speaker_logits": torch.cat(all_speaker_logits),
        "labels": torch.cat(all_labels),
    }


# ─────────────────────────────────────────────────────────
#  Full Training Pipeline
# ─────────────────────────────────────────────────────────

def train(config_path: str = "configs/default_config.yaml"):
    """Main training function."""

    # ── Load Config ──
    config = load_config(config_path)
    hw_profile = get_active_profile(config)
    train_cfg = config["training"]
    log_cfg = config["logging"]
    encoder_type = str(config.get("model", {}).get("encoder_type", "model"))

    print("=" * 55)
    print("  Training Engine — Open-Set Speaker ID")
    print("=" * 55)
    print(f"  Hardware mode: {config['hardware']['mode']}")
    print(f"  Batch size: {hw_profile['batch_size']}")
    print(f"  Mixed precision: {hw_profile['mixed_precision']}")
    print(f"  Epochs: {train_cfg['epochs']}")
    print(f"  Learning rate: {train_cfg['learning_rate']}")
    print(f"  Weight decay: {train_cfg['weight_decay']}")
    print()

    # ── Device ──
    device = setup_device(config)

    # ── DataLoaders ──
    print("\n  [1/4] Preparing DataLoaders...")
    train_loader, val_loader, class_map = get_dataloaders(config)
    num_known = config.get("model", {}).get("competition_num_known", len(class_map) - 1)

    # ── Model ──
    # NOTE: use the config-driven factory, NOT the legacy TwoHeadedWavLM
    # wrapper (which hardcodes a WavLM encoder and ignores `encoder_type`).
    print(f"\n  [2/4] Building model ({num_known} known speakers)...")
    from src.model_factory import create_model_from_config
    model = create_model_from_config(config, num_known_speakers=num_known)
    model = model.to(device)
    model.print_summary()

    # ── Optimizer, Loss, Scaler ──
    print(f"\n  [3/4] Setting up optimizer & loss...")
    # Separate LR for unfrozen encoder blocks (fine-tuning) vs the heads
    encoder_params = [p for n, p in model.named_parameters()
                      if "encoder" in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters()
                   if "encoder" not in n and p.requires_grad]
    param_groups = [{"params": head_params, "lr": train_cfg["learning_rate"]}]
    if encoder_params:
        encoder_lr = train_cfg.get("encoder_lr", 1e-5)
        param_groups.insert(0, {"params": encoder_params, "lr": encoder_lr})
        print(f"  🔓 Encoder LR: {encoder_lr:.2e} ({len(encoder_params):,} tensors) | "
              f"Head LR: {train_cfg['learning_rate']:.2e}")
    else:
        print(f"  🔒 Encoder fully frozen — single LR {train_cfg['learning_rate']:.2e}")
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = build_scheduler(optimizer, train_cfg, train_cfg["epochs"])
    criterion = build_criterion(train_cfg)
    amp_dtype = train_cfg.get("amp_dtype", "fp16")
    autocast_fn, scaler = build_amp(
        amp_enabled=hw_profile["mixed_precision"], amp_dtype=amp_dtype, device=device,
    )
    ema = EMA(model, decay=float(train_cfg.get("ema_decay", 0.999))) \
        if train_cfg.get("ema_enabled", False) else None
    if ema is not None and ema.enabled:
        print(f"  🧊 EMA enabled (decay={ema.decay:.4f})")

    # ── Training Loop ──
    print(f"\n  [4/4] Starting training...")
    print(f"  {'='*50}\n")

    checkpoint_dir = Path(log_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_f1 = -float("inf")
    best_epoch = -1
    history = []

    for epoch in range(1, train_cfg["epochs"] + 1):
        epoch_start = time.time()

        print(f"\n  ── Epoch {epoch}/{train_cfg['epochs']} ──")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion,
            scaler, device, train_cfg["max_grad_norm"],
            ood_grad_norm=train_cfg.get("ood_grad_norm", 1.0),
            autocast_fn=autocast_fn,
            ema=ema,
        )

        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        val_m = evaluate_macro_f1(
            val_metrics["ood_logits"], val_metrics["speaker_logits"],
            val_metrics["labels"], num_classes=len(class_map),
        )
        val_metrics["macro_f1"] = val_m["macro_f1"]

        # LR scheduling
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_time = time.time() - epoch_start

        # Print epoch results
        print(f"\n  📊 Epoch {epoch:2d} — "
              f"Loss: {train_metrics['loss']:.4f} (train) / {val_metrics['loss']:.4f} (val)")
        print(f"     OOD Acc:  {train_metrics['ood_acc']:.3f} (train) / "
              f"{val_metrics['ood_acc']:.3f} (val)")
        print(f"     Spk Acc:  {train_metrics['speaker_acc']:.3f} (train) / "
              f"{val_metrics['speaker_acc']:.3f} (val)")
        print(f"     Macro-F1: {val_metrics['macro_f1']:.4f} (val)  |  "
              f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")

        # Save best checkpoint (based on competition Macro-F1). Named by
        # encoder (<enc>_best.pt) so the submission package can ship one per
        # encoder; best_model.pt is kept as a backward-compat copy.
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            ood_threshold = tune_ood_threshold(
                val_metrics["ood_logits"], val_metrics["labels"])
            ckpt = {
                "epoch": epoch,
                "model_state_dict": (
                    ema.state_dict(model) if ema is not None else model.state_dict()
                ),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": config,
                "class_map": class_map,
                "ood_threshold": ood_threshold,
                "val_loss": val_metrics["loss"],
                "val_ood_acc": val_metrics["ood_acc"],
                "val_speaker_acc": val_metrics["speaker_acc"],
                "val_macro_f1": val_metrics["macro_f1"],
            }
            torch.save(ckpt, checkpoint_dir / f"{encoder_type}_best.pt")
            torch.save(ckpt, checkpoint_dir / "best_model.pt")
            print(f"     💾 Saved new best model (val_macro_f1={best_val_f1:.4f}, "
                  f"ood_thr={ood_threshold:.2f})")

        # Save latest checkpoint (encoder-named + back-compat copy)
        latest_ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "class_map": class_map,
        }
        torch.save(latest_ckpt, checkpoint_dir / f"{encoder_type}_latest.pt")
        torch.save(latest_ckpt, checkpoint_dir / "latest_model.pt")

        # Record history (exclude tensors collected for Macro-F1)
        val_history = {k: v for k, v in val_metrics.items()
                       if not isinstance(v, torch.Tensor)}
        history.append({
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_history.items()},
            "lr": current_lr,
            "time": epoch_time,
        })

    # ── Final Summary ──
    print(f"\n  {'='*50}")
    print(f"  🏁 Training Complete!")
    print(f"  {'='*50}")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Best val Macro-F1: {best_val_f1:.4f}")
    if best_epoch > 0 and len(history) >= best_epoch:
        print(f"  Best val OOD acc: {history[best_epoch-1]['val_ood_acc']:.3f}")
        print(f"  Best val Speaker acc: {history[best_epoch-1]['val_speaker_acc']:.3f}")
    print(f"  Checkpoint saved: {checkpoint_dir / f'{encoder_type}_best.pt'}")
    print()

    return history


# ─────────────────────────────────────────────────────────
#  CLI Entry Point
# ─────────────────────────────────────────────────────────

def main():
    """Test training on a single batch to verify the pipeline."""
    import warnings
    warnings.filterwarnings("ignore")

    print("=" * 55)
    print("  Training Engine — Quick Test (1 batch)")
    print("=" * 55)
    print()

    # Load minimal config
    config = load_config()
    hw_profile = get_active_profile(config)

    device = setup_device(config)

    # Get a single batch from dataloader
    _, val_loader, class_map = get_dataloaders(config)
    num_known = len(class_map) - 1

    # Build model (config-driven factory — honors encoder_type)
    from src.model_factory import create_model_from_config
    model = create_model_from_config(config, num_known_speakers=num_known).to(device)

    # Loss & optimizer
    criterion = TwoPartLoss(ignore_index=-100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler(enabled=hw_profile["mixed_precision"])

    # Single training step
    print("\n  Running 1 training step...")
    model.train()
    waveforms, labels = next(iter(val_loader))
    waveforms, labels = waveforms.to(device), labels.to(device)

    optimizer.zero_grad()
    with autocast():
        ood_logits, speaker_logits = forward_multi_window(model, waveforms, labels=labels)
        loss, loss_dict = criterion(ood_logits, speaker_logits, labels)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    ood_acc = compute_ood_accuracy(ood_logits, labels)
    speaker_acc, n_known = compute_speaker_accuracy(speaker_logits, labels)

    print(f"  ✅ Training step completed!")
    print(f"  Loss components: {loss_dict}")
    print(f"  OOD Accuracy: {ood_acc:.3f}")
    print(f"  Speaker Accuracy (on {n_known} known samples): {speaker_acc:.3f}")
    print(f"\n  {'='*50}")
    print(f"  Training pipeline is ready!")
    print(f"  Run full training: python -m src.train")
    print(f"  {'='*50}")


if __name__ == "__main__":
    # Run full training (not the quick test)
    train()
