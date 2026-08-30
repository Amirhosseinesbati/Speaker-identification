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
from src.metrics import (
    evaluate_competition_probs,
    evaluate_macro_f1,
    fused_probs_from_logits,
)
from src.training_utils import (
    EMA,
    apply_encoder_finetune_mode,
    build_amp,
    build_scheduler,
    encoder_will_train,
    seed_everything,
)
from src.model_artifacts import enrich_checkpoint, create_training_bundle

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
        use_ood: bool = True,
        competition_known_count: int = 446,
        speaker_target_scope: str = "metric",
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.ood_weight = ood_weight
        self.speaker_weight = speaker_weight
        self.use_ood = use_ood
        self.competition_known_count = int(competition_known_count)
        self.speaker_target_scope = str(speaker_target_scope).lower().strip()
        if self.speaker_target_scope not in {"metric", "known"}:
            raise ValueError(
                "speaker_target_scope must be 'metric' or 'known', "
                f"got {self.speaker_target_scope!r}"
            )

        # Cluster mode (use_ood=False): the OOD head does not exist and its BCE
        # supervision would be distorted (unknown files are relabeled to cluster
        # ids, so the label==0 target is never positive) — speaker-only loss.
        self.bce_loss = (
            nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(ood_pos_weight, dtype=torch.float32)
            ) if use_ood else None
        )
        if not use_ood:
            print(f"  🚫 OOD head disabled (cluster mode) — speaker-only loss")

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
        ood_logits: Optional[torch.Tensor],   # (batch, 1) | None (cluster mode)
        speaker_logits: torch.Tensor,          # (batch, 446)
        labels: torch.Tensor,                  # (batch,)  — 0 for unknown, 1..446 for known
        speaker_regularizer: Optional[torch.Tensor] = None,
        speaker_regularizer_weight: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Returns:
            total_loss: scalar tensor
            loss_components: dict with individual loss values
        """
        # ── Loss 1: OOD Detection ── (skipped when the OOD head is disabled)
        # Target is independent from the metric identity.  Pseudo-OOD classes
        # occupy ids > competition_known_count and must remain positive for BCE.
        if self.use_ood and ood_logits is not None:
            ood_targets = (
                (labels == 0) | (labels > self.competition_known_count)
            ).float().unsqueeze(1)
            loss_ood = self.bce_loss(ood_logits, ood_targets)
        else:
            loss_ood = torch.zeros((), dtype=speaker_logits.dtype,
                                   device=speaker_logits.device)

        # ── Loss 2: Known Speaker Classification ──
        # Create masked labels. Metric-scope heads classify pseudo identities;
        # known-scope heads deliberately exclude them and let only OOD BCE plus
        # the optional auxiliary metric loss learn from those samples.
        speaker_labels = labels.clone()
        speaker_ignore = labels == 0
        if self.speaker_target_scope == "known":
            speaker_ignore = speaker_ignore | (labels > self.competition_known_count)
        speaker_labels[speaker_ignore] = self.ignore_index

        # Speaker labels are 1-indexed in the dataset, but CrossEntropy expects 0-indexed.
        # So we subtract 1 to map classes 1..446 → 0..445
        speaker_labels_mapped = speaker_labels.clone()
        mask_known = speaker_labels_mapped != self.ignore_index
        speaker_labels_mapped[mask_known] = speaker_labels_mapped[mask_known] - 1

        loss_speaker = self.ce_loss(speaker_logits, speaker_labels_mapped)

        # Optional literature-fixed exclusive inter-class angular energy. The
        # source paper combines angular classification and regularization as
        # (1-lambda)*L_speaker + lambda*L_inter, so preserve that convex mixture
        # rather than silently adding an unnormalised auxiliary objective.
        speaker_regularizer_weight = float(speaker_regularizer_weight)
        if not 0.0 <= speaker_regularizer_weight < 1.0:
            raise ValueError(
                "speaker_regularizer_weight must be in [0, 1)"
            )
        if speaker_regularizer_weight > 0.0:
            if speaker_regularizer is None:
                raise ValueError(
                    "A positive speaker_regularizer_weight requires a scalar "
                    "speaker_regularizer tensor"
                )
            if speaker_regularizer.numel() != 1:
                raise ValueError("speaker_regularizer must be scalar")
            if not torch.isfinite(speaker_regularizer):
                raise RuntimeError("speaker_regularizer is NaN/Inf")
            loss_speaker_effective = (
                (1.0 - speaker_regularizer_weight) * loss_speaker
                + speaker_regularizer_weight * speaker_regularizer
            )
        else:
            speaker_regularizer = torch.zeros_like(loss_speaker)
            loss_speaker_effective = loss_speaker

        # ── Total Loss (weighted) ──
        total_loss = (
            self.ood_weight * loss_ood
            + self.speaker_weight * loss_speaker_effective
        )

        loss_dict = {
            "loss_ood": loss_ood.item(),
            "loss_speaker": loss_speaker.item(),
            "loss_speaker_effective": loss_speaker_effective.item(),
            "loss_inter_class": speaker_regularizer.item(),
            "loss_inter_class_weighted": (
                self.speaker_weight
                * speaker_regularizer_weight
                * speaker_regularizer.item()
            ),
            "loss_total": total_loss.item(),
        }

        return total_loss, loss_dict


# ─────────────────────────────────────────────────────────
#  Config-driven Loss Factory
# ─────────────────────────────────────────────────────────

def build_criterion(
    train_cfg: dict,
    use_ood: bool = True,
    competition_known_count: int = 446,
    speaker_target_scope: str = "metric",
) -> TwoPartLoss:
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

    ``use_ood=False`` (cluster mode) drops the OOD BCE term entirely — the
    speaker head's pseudo-identity columns carry the unknown supervision.
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
        use_ood=use_ood,
        competition_known_count=competition_known_count,
        speaker_target_scope=str(
            speaker_cfg.get("scope", speaker_target_scope)
        ),
    )


def resolve_inter_class_regularizer(train_cfg: dict) -> Tuple[float, str]:
    """Validate the single preregistered speaker-head regularizer.

    Returns ``(effective_weight, type)``. Disabled configurations always use a
    zero effective weight while retaining their declared type for provenance.
    """
    loss_cfg = train_cfg.get("loss", {}) or {}
    speaker_cfg = loss_cfg.get("speaker", {}) or {}
    inter_cfg = speaker_cfg.get("inter_class", {}) or {}
    enabled = bool(inter_cfg.get("enabled", False))
    inter_type = str(
        inter_cfg.get("type", "exclusive_angular_energy")
    ).lower().strip()
    weight = float(inter_cfg.get("weight", 0.01))
    if inter_type != "exclusive_angular_energy":
        raise ValueError(
            "Only training.loss.speaker.inter_class.type="
            "exclusive_angular_energy is supported"
        )
    if enabled and not 0.0 < weight < 1.0:
        raise ValueError(
            "Enabled inter-class angular regularization requires weight in "
            "(0, 1)"
        )
    return (weight if enabled else 0.0), inter_type


def exclusive_inter_class_angular_loss(
    class_weights: torch.Tensor,
) -> torch.Tensor:
    """Exclusive positive-cosine energy between normalized class weights.

    Implements Chen, Ren & Xu (APSIPA ASC 2019):
    ``(1/C) * ||relu(W_n W_n^T) - I||_F^2``. ArcFace stores classes by row,
    whereas the paper uses column-oriented notation. Negative inter-class
    cosines are intentionally unpenalized. The computation stays in float32
    for stability under fp16 autocast.
    """
    if not isinstance(class_weights, torch.Tensor):
        raise TypeError("class_weights must be a torch.Tensor")
    if class_weights.ndim != 2:
        raise ValueError(
            "exclusive inter-class angular loss requires a 2-D ArcFace "
            "class-weight matrix"
        )
    if class_weights.shape[0] < 2:
        raise ValueError("at least two class weights are required")
    normalized = F.normalize(class_weights.float(), p=2, dim=1)
    gram_positive = torch.relu(normalized @ normalized.transpose(0, 1))
    identity = torch.eye(
        normalized.shape[0],
        dtype=gram_positive.dtype,
        device=gram_positive.device,
    )
    return (gram_positive - identity).square().sum() / normalized.shape[0]


# ─────────────────────────────────────────────────────────
#  Prototypical Loss (EMA centroids — matches the centroid readout)
# ─────────────────────────────────────────────────────────

class PrototypicalLoss(nn.Module):
    """
    EMA-centroid prototypical loss — aligns training with the nearest-centroid
    readout the decision layer uses at inference.

    Maintains an EMA prototype (L2-normalised) per known class in the ArcFace
    embedding space. Each known sample is scored by AM-softmax (subtractive
    margin) against all prototypes, so the projection learns a space where
    nearest-centroid classification works. The prototypes are data-derived
    (EMA of the actual train embeddings), unlike ArcFace's learned weight rows
    — this is the few-shot-safe complement that ties training to the readout.

    Labels are the ORIGINAL dataset ids (0 = unknown, 1..446 = known); unknown
    samples are masked out and remapped known classes use 0..445.
    """

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        scale: float = 30.0,
        margin: float = 0.2,
        decay: float = 0.9,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.embedding_dim = int(embedding_dim)
        self.scale = float(scale)
        self.margin = float(margin)
        self.decay = float(decay)

        # EMA prototypes: (num_classes, embedding_dim), L2-normalised.
        proto = torch.randn(self.num_classes, self.embedding_dim)
        self.register_buffer("prototypes", F.normalize(proto, p=2, dim=1))

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """embeddings: (N, D) L2-normalised; labels: (N,) original (0 = unknown)."""
        remapped = labels - 1
        known_mask = (remapped >= 0) & (remapped < self.num_classes)
        if not known_mask.any():
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        emb = embeddings[known_mask]
        lab = remapped[known_mask]

        # EMA prototype update (no grad — the prototypes are not parameters).
        with torch.no_grad():
            # Only classes present in this batch need an EMA update. Iterating
            # all 1000 metric classes for every window was a substantial Python
            # bottleneck in hybrid runs.
            for c in lab.unique().tolist():
                c = int(c)
                m = lab == c
                new = F.normalize(emb[m].mean(dim=0, keepdim=True), p=2, dim=1)[0]
                self.prototypes[c] = (
                    self.decay * self.prototypes[c] + (1.0 - self.decay) * new
                )

        cos = emb @ self.prototypes.T  # (N, num_classes)
        cos = cos.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # AM-softmax subtractive margin on the target class. The margin value
        # is gathered from the SAME tensor the scatter writes into, so the
        # scatter must be out-of-place — an in-place scatter_ bumped the tensor
        # to version 1 while the gather's backward still needed version 0.
        target_cos = cos.gather(1, lab.unsqueeze(1)) - self.margin
        logits = self.scale * cos.scatter(1, lab.unsqueeze(1), target_cos)
        return F.cross_entropy(logits, lab)


# ─────────────────────────────────────────────────────────
#  Metric Calculators
# ─────────────────────────────────────────────────────────

def compute_ood_accuracy(
    ood_logits: Optional[torch.Tensor],
    labels: torch.Tensor,
    competition_known_count: int = 446,
) -> float:
    """
    Compute OOD detection accuracy:
    - Correct if sigmoid(logit) > 0.5 AND label is 0 (unknown detected as unknown)
    - OR sigmoid(logit) <= 0.5 AND label is not 0 (known detected as known)

    With the OOD head disabled (``ood_logits is None``, cluster mode) the
    signal is a monotonic function of the speaker head only — return 0.0 so
    the metric is clearly marked as unavailable instead of crashing.
    """
    if ood_logits is None:
        return 0.0
    probs = torch.sigmoid(ood_logits).squeeze(1)  # (batch,)
    predictions = (probs > 0.5).long()  # 1 = predicted unknown, 0 = predicted known
    targets = ((labels == 0) | (labels > competition_known_count)).long()
    correct = (predictions == targets).sum().item()
    return correct / labels.size(0)


def compute_speaker_accuracy(
    speaker_logits: torch.Tensor,
    labels: torch.Tensor,
    competition_known_count: int = 446,
) -> Tuple[float, int]:
    """
    Compute known speaker accuracy (only for samples with label != 0).
    """
    known_mask = (labels > 0) & (labels <= competition_known_count)
    if known_mask.sum() == 0:
        return 0.0, 0

    known_logits = speaker_logits[known_mask]
    known_labels = labels[known_mask] - 1  # Map 1..446 → 0..445

    predictions = known_logits.argmax(dim=1)
    correct = (predictions == known_labels).sum().item()
    total = known_labels.size(0)

    return correct / total, total


def tune_ood_threshold(
    ood_logits: Optional[torch.Tensor],
    labels: torch.Tensor,
    competition_known_count: int = 446,
) -> Optional[float]:
    """Tune a binary OOD threshold on the val set for the checkpoint.

    Mirrors the ZenML evaluate step: sweep 0.1..0.9 by unknown-class binary
    F1 (fallback to the median P(unknown) if the head collapsed). Persisted
    as ckpt['ood_threshold'] so --apply-ood-threshold works from plain
    `python -m src.train` too.

    Returns ``None`` when the OOD head is disabled (cluster mode).
    """
    if ood_logits is None:
        return None
    from sklearn.metrics import f1_score

    ood_probs = torch.sigmoid(ood_logits.squeeze(1)).numpy()
    ood_targets = (
        (labels == 0) | (labels > competition_known_count)
    ).numpy().astype(int)
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
        if ood is not None:  # OOD head disabled (cluster mode)
            ood_sum = ood if ood_sum is None else ood_sum + ood
        spk_sum = spk if spk_sum is None else spk_sum + spk
    ood_logit = None if ood_sum is None else ood_sum / W
    return ood_logit, spk_sum / W


@torch.no_grad()
def forward_multi_window_evaluation(
    model: nn.Module,
    waveforms: torch.Tensor,
) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Return diagnostic mean logits and submission-consistent probabilities.

    The leaderboard path fuses each window's two heads into a competition
    probability vector *before* averaging windows. Fusing averaged logits is
    not equivalent because sigmoid and softmax are nonlinear. Keeping both
    outputs lets validation retain loss/threshold diagnostics while model
    selection uses exactly the distribution submitted to the competition.
    """
    num_unknown_clusters = int(getattr(model, "num_unknown_clusters", 0))
    if waveforms.dim() == 3:
        ood_logits, speaker_logits = model(waveforms, labels=None)
        probabilities = fused_probs_from_logits(
            ood_logits,
            speaker_logits,
            num_unknown_clusters=num_unknown_clusters,
        )
        return ood_logits, speaker_logits, probabilities

    if waveforms.dim() != 4:
        raise ValueError(
            "Expected waveforms shaped (B,1,T) or (B,W,1,T), got "
            f"{tuple(waveforms.shape)}"
        )

    num_windows = waveforms.shape[1]
    ood_sum = None
    speaker_sum = None
    probability_sum = None
    for window_idx in range(num_windows):
        ood_logits, speaker_logits = model(
            waveforms[:, window_idx], labels=None)
        probabilities = fused_probs_from_logits(
            ood_logits,
            speaker_logits,
            num_unknown_clusters=num_unknown_clusters,
        )
        if ood_logits is not None:
            ood_sum = ood_logits if ood_sum is None else ood_sum + ood_logits
        speaker_sum = (
            speaker_logits if speaker_sum is None else speaker_sum + speaker_logits
        )
        probability_sum = (
            probabilities
            if probability_sum is None
            else probability_sum + probabilities
        )

    mean_ood = None if ood_sum is None else ood_sum / num_windows
    return mean_ood, speaker_sum / num_windows, probability_sum / num_windows


# ─────────────────────────────────────────────────────────
#  Training Epoch
# ─────────────────────────────────────────────────────────

def cross_file_pair_consistency(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    competition_known_count: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cosine consistency for two distinct files of each known speaker.

    The paired batch sampler guarantees exactly two rows for every selected
    known speaker and keeps OOD rows unchanged.  The first row after the
    deterministic batch shuffle is the student and the second is a detached
    target.  Both rows still receive the ordinary supervised loss, so this
    auxiliary term changes only cross-file invariance pressure.
    """
    if embeddings.ndim != 2:
        raise ValueError(
            f"embeddings must be two-dimensional, got {embeddings.shape}"
        )
    if labels.ndim != 1 or len(labels) != len(embeddings):
        raise ValueError(
            "labels must be one-dimensional and align with embeddings"
        )
    known_mask = (labels > 0) & (labels <= int(competition_known_count))
    known_labels = labels[known_mask]
    if not len(known_labels):
        raise ValueError("cross-file consistency batch contains no known rows")

    unique_labels, counts = torch.unique(known_labels, return_counts=True)
    if not torch.all(counts == 2):
        bad = [
            (int(label.item()), int(count.item()))
            for label, count in zip(unique_labels, counts)
            if int(count.item()) != 2
        ]
        raise ValueError(
            "cross-file consistency requires exactly two files for every "
            f"selected known speaker; bad_counts={bad[:5]}"
        )

    anchor_indices = []
    target_indices = []
    for label in unique_labels:
        indices = torch.nonzero(labels == label, as_tuple=False).flatten()
        anchor_indices.append(indices[0])
        target_indices.append(indices[1])
    anchor = embeddings[torch.stack(anchor_indices)]
    target = embeddings[torch.stack(target_indices)].detach()
    cosines = F.cosine_similarity(anchor.float(), target.float(), dim=1)
    pair_cosine = cosines.mean()
    loss = 1.0 - pair_cosine
    anchor_std = anchor.float().std(dim=0, unbiased=False).mean()
    target_std = target.float().std(dim=0, unbiased=False).mean()
    return loss, pair_cosine, anchor_std, target_std

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
    proto_criterion: Optional[nn.Module] = None,
    proto_weight: float = 0.0,
    consistency_weight: float = 0.0,
    consistency_pairing: str = "clean_aug",
    inter_class_weight: float = 0.0,
) -> Dict[str, float]:
    """
    Train for one epoch with AMP.

    **Per-window loss (root cause R2):** a ``(B, W, 1, T)`` batch is processed
    one window at a time and gradients are accumulated across the ``W`` windows
    (peak activation stays at ``(B, 1, T)``), so the loss is computed per window
    instead of averaging the ``W`` logits first — without the OOM of flattening
    to a single ``(B*W, 1, T)`` forward.

    With paired clean/augmented dataset views, the supervised objective remains
    on the augmented branch and a stop-gradient clean embedding acts as the
    target for a cosine consistency term.  The clean branch is evaluated with
    dropout disabled so the auxiliary target measures channel distortion, not
    two unrelated dropout masks.  In ``cross_file_batch`` mode the balanced
    sampler instead places two distinct files from every selected known speaker
    in the same ordinary supervised batch; their already-computed embeddings
    form a stop-gradient positive pair and OOD rows are excluded.

    Returns dict of average metrics.
    """
    if consistency_weight < 0:
        raise ValueError("consistency_weight must be non-negative")
    if not 0.0 <= inter_class_weight < 1.0:
        raise ValueError("inter_class_weight must be in [0, 1)")
    if inter_class_weight > 0.0:
        speaker_head = getattr(model, "head_speaker", None)
        class_weights = getattr(speaker_head, "weight", None)
        if not isinstance(class_weights, torch.Tensor):
            raise ValueError(
                "inter-class angular regularization requires an ArcFace-like "
                "speaker head with a 2-D weight tensor"
            )
        if class_weights.ndim != 2:
            raise ValueError(
                "inter-class angular regularization supports only a single-"
                "center ArcFace weight matrix"
            )
    consistency_pairing = str(consistency_pairing).lower().strip()
    if consistency_pairing not in {"clean_aug", "cross_file_batch"}:
        raise ValueError(
            "consistency_pairing must be clean_aug or cross_file_batch"
        )
    if autocast_fn is None:
        def autocast_fn():
            return autocast()

    model.train()
    total_loss = 0.0
    total_loss_primary = 0.0
    total_loss_proto = 0.0
    total_loss_proto_weighted = 0.0
    total_loss_consistency = 0.0
    total_loss_consistency_weighted = 0.0
    total_pair_cosine = 0.0
    total_embedding_std_augmented = 0.0
    total_embedding_std_clean = 0.0
    total_ood_acc = 0.0
    total_speaker_acc = 0.0
    total_loss_ood = 0.0
    total_loss_spk = 0.0
    total_loss_spk_effective = 0.0
    total_loss_inter_class = 0.0
    total_loss_inter_class_weighted = 0.0
    num_batches = len(dataloader)

    progress_bar = tqdm(dataloader, desc="  Train", leave=False)
    for step, (batch_views, labels) in enumerate(progress_bar):
        clean_waveforms = None
        if isinstance(batch_views, dict):
            if consistency_pairing != "clean_aug":
                raise ValueError(
                    "Paired clean/aug dataset views require "
                    "consistency_pairing=clean_aug"
                )
            if set(batch_views) != {"augmented", "clean"}:
                raise ValueError(
                    "Paired training view dict must contain exactly "
                    "'augmented' and 'clean'"
                )
            waveforms = batch_views["augmented"]
            clean_waveforms = batch_views["clean"]
            if consistency_weight <= 0:
                raise ValueError(
                    "Dataset returned paired views but consistency_weight is not positive"
                )
        else:
            waveforms = batch_views
            if consistency_weight > 0 and consistency_pairing != "cross_file_batch":
                raise ValueError(
                    "consistency_weight > 0 requires paired clean/augmented dataset views"
                )

        waveforms = waveforms.to(device, non_blocking=True)
        if clean_waveforms is not None:
            clean_waveforms = clean_waveforms.to(device, non_blocking=True)
            if clean_waveforms.shape != waveforms.shape:
                raise ValueError(
                    "Clean and augmented view tensors must have identical shapes"
                )
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
        step_loss_primary = 0.0
        step_loss_proto = 0.0
        step_loss_proto_weighted = 0.0
        step_loss_consistency = 0.0
        step_loss_consistency_weighted = 0.0
        step_pair_cosine = 0.0
        step_embedding_std_augmented = 0.0
        step_embedding_std_clean = 0.0
        step_ood_acc = 0.0
        step_spk_acc = 0.0
        step_loss_ood = 0.0
        step_loss_spk = 0.0
        step_loss_spk_effective = 0.0
        step_loss_inter_class = 0.0
        step_loss_inter_class_weighted = 0.0
        for w in range(W):
            wf = waveforms[:, w] if W > 1 else waveforms  # (B, 1, T)

            clean_embedding = None
            if clean_waveforms is not None:
                clean_wf = (
                    clean_waveforms[:, w] if W > 1 else clean_waveforms
                )
                was_training = model.training
                model.eval()
                try:
                    with torch.no_grad(), autocast_fn():
                        clean_embedding = model.embed(clean_wf).detach()
                finally:
                    if was_training:
                        model.train()

            with autocast_fn():
                need_embedding = (
                    proto_criterion is not None
                    or clean_embedding is not None
                    or (consistency_weight > 0
                        and consistency_pairing == "cross_file_batch")
                )
                if need_embedding:
                    ood_logits, speaker_logits, emb = model(
                        wf, labels=labels, return_embedding=True)
                else:
                    ood_logits, speaker_logits = model(wf, labels=labels)
                inter_class_loss = (
                    exclusive_inter_class_angular_loss(
                        model.head_speaker.weight
                    )
                    if inter_class_weight > 0.0
                    else None
                )
                primary_loss, loss_dict = criterion(
                    ood_logits,
                    speaker_logits,
                    labels,
                    speaker_regularizer=inter_class_loss,
                    speaker_regularizer_weight=inter_class_weight,
                )
                proto_loss = torch.zeros_like(primary_loss)
                if proto_criterion is not None:
                    proto_loss = proto_criterion(emb, labels)
                weighted_proto_loss = proto_weight * proto_loss
                consistency_loss = torch.zeros_like(primary_loss)
                pair_cosine = torch.zeros_like(primary_loss)
                embedding_std_augmented = torch.zeros_like(primary_loss)
                embedding_std_clean = torch.zeros_like(primary_loss)
                if clean_embedding is not None:
                    pair_cosines = F.cosine_similarity(
                        emb.float(), clean_embedding.float(), dim=1,
                    )
                    pair_cosine = pair_cosines.mean()
                    consistency_loss = 1.0 - pair_cosine
                    embedding_std_augmented = emb.float().std(
                        dim=0, unbiased=False,
                    ).mean()
                    embedding_std_clean = clean_embedding.float().std(
                        dim=0, unbiased=False,
                    ).mean()
                elif (consistency_weight > 0
                      and consistency_pairing == "cross_file_batch"):
                    (
                        consistency_loss,
                        pair_cosine,
                        embedding_std_augmented,
                        embedding_std_clean,
                    ) = cross_file_pair_consistency(
                        emb, labels, criterion.competition_known_count,
                    )
                weighted_consistency_loss = (
                    consistency_weight * consistency_loss
                )
                loss = (
                    primary_loss
                    + weighted_proto_loss
                    + weighted_consistency_loss
                )

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

            # ``loss`` is the objective actually backpropagated.  The old
            # logger recorded only ``loss_dict['loss_total']`` and silently
            # omitted the auxiliary prototype term, making AuxMetric training
            # curves look artificially better than the control.
            step_loss += float(loss.detach().item())
            step_loss_primary += loss_dict["loss_total"]
            step_loss_proto += float(proto_loss.detach().item())
            step_loss_proto_weighted += float(weighted_proto_loss.detach().item())
            step_loss_consistency += float(consistency_loss.detach().item())
            step_loss_consistency_weighted += float(
                weighted_consistency_loss.detach().item()
            )
            step_pair_cosine += float(pair_cosine.detach().item())
            step_embedding_std_augmented += float(
                embedding_std_augmented.detach().item()
            )
            step_embedding_std_clean += float(
                embedding_std_clean.detach().item()
            )
            step_loss_ood += loss_dict["loss_ood"]
            step_loss_spk += loss_dict["loss_speaker"]
            step_loss_spk_effective += loss_dict["loss_speaker_effective"]
            step_loss_inter_class += loss_dict["loss_inter_class"]
            step_loss_inter_class_weighted += loss_dict[
                "loss_inter_class_weighted"
            ]
            step_ood_acc += compute_ood_accuracy(
                ood_logits, labels, criterion.competition_known_count)
            spk_acc, _ = compute_speaker_accuracy(
                speaker_logits, labels, criterion.competition_known_count)
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
        total_loss_primary += step_loss_primary / W
        total_loss_proto += step_loss_proto / W
        total_loss_proto_weighted += step_loss_proto_weighted / W
        total_loss_consistency += step_loss_consistency / W
        total_loss_consistency_weighted += (
            step_loss_consistency_weighted / W
        )
        total_pair_cosine += step_pair_cosine / W
        total_embedding_std_augmented += step_embedding_std_augmented / W
        total_embedding_std_clean += step_embedding_std_clean / W
        total_ood_acc += step_ood_acc / W
        total_speaker_acc += step_spk_acc / W
        total_loss_ood += step_loss_ood / W
        total_loss_spk += step_loss_spk / W
        total_loss_spk_effective += step_loss_spk_effective / W
        total_loss_inter_class += step_loss_inter_class / W
        total_loss_inter_class_weighted += (
            step_loss_inter_class_weighted / W
        )

        # Update progress bar
        progress_bar.set_postfix({
            "loss": f"{step_loss / W:.4f}",
            "ood": f"{step_ood_acc / W:.3f}",
            "spk": f"{step_spk_acc / W:.3f}",
            **({"pair": f"{step_pair_cosine / W:.3f}"}
               if clean_waveforms is not None else {}),
        })

    primary_weight_sum = criterion.ood_weight + criterion.speaker_weight
    mean_primary_loss = total_loss_primary / num_batches
    return {
        "loss": total_loss / num_batches,
        "loss_primary": mean_primary_loss,
        "loss_primary_normalized": (
            mean_primary_loss / primary_weight_sum
            if primary_weight_sum > 0
            else mean_primary_loss
        ),
        "loss_proto": total_loss_proto / num_batches,
        "loss_proto_weighted": total_loss_proto_weighted / num_batches,
        "loss_consistency": total_loss_consistency / num_batches,
        "loss_consistency_weighted": (
            total_loss_consistency_weighted / num_batches
        ),
        "pair_cosine": total_pair_cosine / num_batches,
        "embedding_std_augmented": (
            total_embedding_std_augmented / num_batches
        ),
        "embedding_std_clean": total_embedding_std_clean / num_batches,
        "loss_ood": total_loss_ood / num_batches,
        "loss_speaker": total_loss_spk / num_batches,
        "loss_speaker_effective": total_loss_spk_effective / num_batches,
        "loss_inter_class": total_loss_inter_class / num_batches,
        "loss_inter_class_weighted": (
            total_loss_inter_class_weighted / num_batches
        ),
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

    Also collects both mean logits (diagnostics/threshold tuning) and
    per-window-averaged competition probabilities. Callers must select models
    with the latter because it is the actual submission aggregation path.
    """
    model.eval()
    total_loss = 0.0
    total_loss_ood = 0.0
    total_loss_speaker = 0.0
    total_ood_acc = 0.0
    total_speaker_acc = 0.0
    num_batches = len(dataloader)

    all_ood_logits, all_speaker_logits, all_labels = [], [], []
    all_competition_probs = []

    progress_bar = tqdm(dataloader, desc="  Val", leave=False)
    for waveforms, labels in progress_bar:
        waveforms = waveforms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # No labels → no ArcFace margin → honest eval logits
        ood_logits, speaker_logits, competition_probs = (
            forward_multi_window_evaluation(model, waveforms)
        )
        loss, loss_dict = criterion(ood_logits, speaker_logits, labels)

        total_loss += loss_dict["loss_total"]
        total_loss_ood += loss_dict["loss_ood"]
        total_loss_speaker += loss_dict["loss_speaker"]
        total_ood_acc += compute_ood_accuracy(
            ood_logits, labels, criterion.competition_known_count)
        speaker_acc, _ = compute_speaker_accuracy(
            speaker_logits, labels, criterion.competition_known_count)
        total_speaker_acc += speaker_acc

        if ood_logits is not None:  # OOD head disabled (cluster mode)
            all_ood_logits.append(ood_logits.cpu())
        all_speaker_logits.append(speaker_logits.cpu())
        all_labels.append(labels.cpu())
        all_competition_probs.append(competition_probs.cpu())

        progress_bar.set_postfix({
            "loss": f"{loss_dict['loss_total']:.4f}",
            "ood": f"{compute_ood_accuracy(ood_logits, labels, criterion.competition_known_count):.3f}",
            "spk": f"{speaker_acc:.3f}",
        })

    primary_weight_sum = criterion.ood_weight + criterion.speaker_weight
    mean_loss = total_loss / num_batches
    return {
        "loss": mean_loss,
        "loss_primary_normalized": (
            mean_loss / primary_weight_sum
            if primary_weight_sum > 0
            else mean_loss
        ),
        "loss_ood": total_loss_ood / num_batches,
        "loss_speaker": total_loss_speaker / num_batches,
        "ood_acc": total_ood_acc / num_batches,
        "speaker_acc": total_speaker_acc / num_batches,
        "ood_logits": (torch.cat(all_ood_logits) if all_ood_logits else None),
        "speaker_logits": torch.cat(all_speaker_logits),
        "labels": torch.cat(all_labels),
        "competition_probs": torch.cat(all_competition_probs),
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
    consistency_cfg = (
        ((train_cfg.get("loss", {}) or {}).get("consistency", {}) or {})
    )
    consistency_enabled = bool(consistency_cfg.get("enabled", False))
    consistency_weight = float(consistency_cfg.get("weight", 0.0))
    consistency_type = str(consistency_cfg.get("type", "cosine")).lower()
    consistency_pairing = str(
        consistency_cfg.get("pairing", "clean_aug")
    ).lower().strip()
    inter_class_weight, inter_class_type = resolve_inter_class_regularizer(
        train_cfg
    )
    if consistency_enabled and consistency_weight <= 0:
        raise ValueError(
            "training.loss.consistency.enabled requires a positive weight"
        )
    if consistency_enabled and consistency_type != "cosine":
        raise ValueError(
            "Only training.loss.consistency.type=cosine is supported"
        )
    if consistency_pairing not in {"clean_aug", "cross_file_batch"}:
        raise ValueError(
            "training.loss.consistency.pairing must be clean_aug or "
            "cross_file_batch"
        )
    encoder_type = str(config.get("model", {}).get("encoder_type", "model"))
    reproducibility = seed_everything(
        train_cfg.get("seed", 42),
        deterministic=train_cfg.get("deterministic_algorithms", True),
    )

    print("=" * 55)
    print("  Training Engine — Open-Set Speaker ID")
    print("=" * 55)
    print(f"  Hardware mode: {config['hardware']['mode']}")
    print(f"  Batch size: {hw_profile['batch_size']}")
    print(f"  Mixed precision: {hw_profile['mixed_precision']}")
    print(f"  Epochs: {train_cfg['epochs']}")
    print(f"  Learning rate: {train_cfg['learning_rate']}")
    print(f"  Weight decay: {train_cfg['weight_decay']}")
    print(f"  Training seed: {reproducibility['seed']} | "
          f"deterministic: {reproducibility['deterministic_algorithms']}")
    if consistency_enabled:
        print(
            "  Embedding consistency: "
            f"pairing={consistency_pairing}, cosine weight="
            f"{consistency_weight:g} (target stop-gradient)"
        )
    if inter_class_weight > 0.0:
        print(
            "  Inter-class speaker regularizer: "
            f"type={inter_class_type}, convex weight={inter_class_weight:g}"
        )
    print()

    # ── Device ──
    device = setup_device(config)

    # ── DataLoaders ──
    print("\n  [1/4] Preparing DataLoaders...")
    train_loader, val_loader, class_map = get_dataloaders(config)
    # Metric labels may span 446+k identities while the known-first primary
    # speaker head intentionally spans only the 446 competition identities.
    num_known = len(class_map) - 1

    # ── Model ──
    # NOTE: use the config-driven factory, NOT the legacy TwoHeadedWavLM
    # wrapper (which hardcodes a WavLM encoder and ignores `encoder_type`).
    print(f"\n  [2/4] Building model ({num_known} known speakers)...")
    from src.model_factory import create_model_from_config
    model = create_model_from_config(config, num_known_speakers=num_known)
    # Number of pseudo columns emitted by the PRIMARY head. It is zero for
    # known-first models even when the metric class map contains pseudo ids.
    output_unknown_clusters = int(model.num_unknown_clusters)
    num_output_classes = int(model.num_output_classes)
    model = model.to(device)
    model.print_summary()

    # ── Optimizer, Loss, Scaler ──
    print(f"\n  [3/4] Setting up optimizer & loss...")
    # Progressive unfreezing (two-phase fine-tune): keep the encoder frozen for
    # the first `freeze_epochs` epochs, then restore the configured mode.
    freeze_epochs = int(train_cfg.get("freeze_epochs", 0))
    progressive = freeze_epochs > 0 and encoder_will_train(config)
    if progressive:
        model.encoder.freeze()
        print(f"  🧊 Progressive unfreezing: encoder frozen for first {freeze_epochs} epoch(s)")

    # Separate LR for unfrozen encoder blocks (fine-tuning) vs the heads.
    # Under progressive unfreezing the encoder params are collected by NAME
    # (currently frozen, so `requires_grad` is False) so the optimizer param
    # group exists from the start and begins stepping after the transition.
    if progressive:
        encoder_params = [p for n, p in model.named_parameters() if "encoder" in n]
    else:
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
    from src.heads import ood_head_enabled
    competition_known_count = int(
        config.get("model", {}).get("competition_num_known", 446))
    criterion = build_criterion(
        train_cfg, use_ood=ood_head_enabled(config),
        competition_known_count=competition_known_count,
        speaker_target_scope=getattr(model, "speaker_target_scope", "metric"),
    )
    amp_dtype = train_cfg.get("amp_dtype", "fp16")
    autocast_fn, scaler = build_amp(
        amp_enabled=hw_profile["mixed_precision"], amp_dtype=amp_dtype, device=device,
    )
    ema = EMA(model, decay=float(train_cfg.get("ema_decay", 0.999))) \
        if train_cfg.get("ema_enabled", False) else None
    if ema is not None and ema.enabled:
        print(f"  🧊 EMA enabled (decay={ema.decay:.4f})")

    # Prototypical loss (EMA centroids) — optional, aligns training with the
    # nearest-centroid readout. Disabled by default (training.loss.proto.enabled).
    proto_cfg = (train_cfg.get("loss", {}) or {}).get("proto", {}) or {}
    proto_criterion = None
    proto_weight = 0.0
    if bool(proto_cfg.get("enabled", False)):
        emb_dim = getattr(model.head_speaker, "embedding_dim", None)
        if emb_dim is None:
            emb_dim = model.encoder.output_dim * model.pooling.output_multiplier
        proto_scope = str(proto_cfg.get("scope", "metric")).lower().strip()
        proto_classes = (competition_known_count
                         if proto_scope == "known" else num_known)
        proto_criterion = PrototypicalLoss(
            num_classes=proto_classes,
            embedding_dim=int(emb_dim),
            scale=float(proto_cfg.get("scale", 30.0)),
            margin=float(proto_cfg.get("margin", 0.2)),
            decay=float(proto_cfg.get("decay", 0.9)),
        ).to(device)
        proto_weight = float(proto_cfg.get("weight", 0.1))
        print(f"  🎯 Prototypical loss enabled (weight={proto_weight}, "
              f"scale={proto_cfg.get('scale', 30.0)}, margin={proto_cfg.get('margin', 0.2)})")

    # ── Training Loop ──
    print(f"\n  [4/4] Starting training...")
    print(f"  {'='*50}\n")

    checkpoint_dir = Path(log_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_f1 = -float("inf")
    best_epoch = -1
    best_ema_f1 = -float("inf")
    best_ema_epoch = -1
    history = []
    split_scheme = str((config.get("data", {}).get("split", {}) or {})
                       .get("scheme", "single")).lower().strip()
    selection_mode = str(train_cfg.get(
        "selection_mode", "last_epoch" if split_scheme == "full" else "best_macro_f1"
    )).lower().strip()
    if split_scheme == "full":
        selection_mode = "last_epoch"
        print("  🧾 Full-data selection: final epoch only; overlapping val is diagnostic")

    for epoch in range(1, train_cfg["epochs"] + 1):
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch - 1)
        epoch_start = time.time()

        print(f"\n  ── Epoch {epoch}/{train_cfg['epochs']} ──")

        # Progressive unfreezing: at the transition epoch, restore the configured
        # fine-tune mode and add the newly-trainable encoder params to the EMA.
        if progressive and epoch == freeze_epochs + 1:
            apply_encoder_finetune_mode(model, config)
            if ema is not None and ema.enabled:
                ema.extend(model)
            print(f"  🔓 Encoder unfrozen (progressive schedule, epoch {epoch})")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion,
            scaler, device, train_cfg["max_grad_norm"],
            ood_grad_norm=train_cfg.get("ood_grad_norm", 1.0),
            autocast_fn=autocast_fn,
            ema=ema,
            proto_criterion=proto_criterion,
            proto_weight=proto_weight,
            consistency_weight=(
                consistency_weight if consistency_enabled else 0.0
            ),
            consistency_pairing=consistency_pairing,
            inter_class_weight=inter_class_weight,
        )

        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        val_logit_m = evaluate_macro_f1(
            val_metrics["ood_logits"], val_metrics["speaker_logits"],
            val_metrics["labels"],
            num_classes=num_output_classes,
            num_unknown_clusters=output_unknown_clusters,
        )
        val_m = evaluate_competition_probs(
            val_metrics["competition_probs"], val_metrics["labels"])
        val_metrics["logit_avg_macro_f1"] = val_logit_m["macro_f1"]
        val_metrics["macro_f1"] = val_m["macro_f1"]

        # Score EMA independently.  The raw metric remains the early-stopping
        # signal to preserve the original recipe; EMA is a separately selected
        # deployment candidate and never borrows the raw metric again.
        ema_val_metrics = None
        if ema is not None and ema.enabled:
            with ema.average_parameters(model):
                ema_val_metrics = validate_epoch(model, val_loader, criterion, device)
            ema_logit_m = evaluate_macro_f1(
                ema_val_metrics["ood_logits"], ema_val_metrics["speaker_logits"],
                ema_val_metrics["labels"],
                num_classes=num_output_classes,
                num_unknown_clusters=output_unknown_clusters,
            )
            ema_m = evaluate_competition_probs(
                ema_val_metrics["competition_probs"], ema_val_metrics["labels"])
            ema_val_metrics["logit_avg_macro_f1"] = ema_logit_m["macro_f1"]
            ema_val_metrics["macro_f1"] = ema_m["macro_f1"]

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
        print(f"     Macro-F1: {val_metrics['macro_f1']:.4f} (prob-avg val)  |  "
              f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")
        print(f"     Logit-avg diagnostic F1: "
              f"{val_metrics['logit_avg_macro_f1']:.4f}")
        if ema_val_metrics is not None:
            print(f"     EMA Macro-F1: {ema_val_metrics['macro_f1']:.4f} "
                  "(prob-avg independent val)")

        # Save best checkpoint (based on competition Macro-F1). Named by
        # encoder (<enc>_best.pt) so the submission package can ship one per
        # encoder; best_model.pt is kept as a backward-compat copy.
        should_save = (
            (selection_mode == "last_epoch" and epoch == train_cfg["epochs"])
            or (selection_mode != "last_epoch" and
                val_metrics["macro_f1"] > best_val_f1)
        )
        if should_save:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            ood_threshold = tune_ood_threshold(
                val_metrics["ood_logits"], val_metrics["labels"],
                competition_known_count=competition_known_count)
            ckpt = enrich_checkpoint({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": config,
                "class_map": class_map,
                "weight_variant": "raw",
                "reproducibility": reproducibility,
                "ood_threshold": ood_threshold,
                "val_loss": val_metrics["loss"],
                "val_ood_acc": val_metrics["ood_acc"],
                "val_speaker_acc": val_metrics["speaker_acc"],
                "val_macro_f1": val_metrics["macro_f1"],
            }, config, class_map, metrics={
                "val_loss": val_metrics["loss"],
                "val_ood_acc": val_metrics["ood_acc"],
                "val_speaker_acc": val_metrics["speaker_acc"],
                "val_macro_f1": val_metrics["macro_f1"],
            }, history=history)
            torch.save(ckpt, checkpoint_dir / f"{encoder_type}_best.pt")
            torch.save(ckpt, checkpoint_dir / f"{encoder_type}_best_raw.pt")
            torch.save(ckpt, checkpoint_dir / "best_model.pt")
            threshold_text = (f"{ood_threshold:.2f}" if ood_threshold is not None
                              else "n/a")
            print(f"     💾 Saved new best model (val_macro_f1={best_val_f1:.4f}, "
                  f"ood_thr={threshold_text}, weights=raw)")

        if (ema_val_metrics is not None and
                ema_val_metrics["macro_f1"] > best_ema_f1):
            best_ema_f1 = ema_val_metrics["macro_f1"]
            best_ema_epoch = epoch
            ema_threshold = tune_ood_threshold(
                ema_val_metrics["ood_logits"], ema_val_metrics["labels"],
                competition_known_count=competition_known_count)
            ema_ckpt = enrich_checkpoint({
                "epoch": epoch,
                "model_state_dict": ema.state_dict(model),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": config,
                "class_map": class_map,
                "weight_variant": "ema",
                "reproducibility": reproducibility,
                "ood_threshold": ema_threshold,
                "val_loss": ema_val_metrics["loss"],
                "val_ood_acc": ema_val_metrics["ood_acc"],
                "val_speaker_acc": ema_val_metrics["speaker_acc"],
                "val_macro_f1": ema_val_metrics["macro_f1"],
            }, config, class_map, metrics={
                "val_loss": ema_val_metrics["loss"],
                "val_ood_acc": ema_val_metrics["ood_acc"],
                "val_speaker_acc": ema_val_metrics["speaker_acc"],
                "val_macro_f1": ema_val_metrics["macro_f1"],
            }, history=history)
            torch.save(ema_ckpt, checkpoint_dir / f"{encoder_type}_best_ema.pt")
            print(f"     💾 Saved EMA candidate (val_macro_f1={best_ema_f1:.4f})")

        # Save latest checkpoint (encoder-named + back-compat copy)
        latest_ckpt = enrich_checkpoint({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "class_map": class_map,
        }, config, class_map, history=history)
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
    best_path = checkpoint_dir / f"{encoder_type}_best.pt"
    raw_path = checkpoint_dir / f"{encoder_type}_best_raw.pt"
    ema_path = checkpoint_dir / f"{encoder_type}_best_ema.pt"
    selected_variant = "raw"
    if ema_path.exists() and best_ema_f1 > best_val_f1:
        selected_variant = "ema"
        torch.save(torch.load(ema_path, map_location="cpu", weights_only=False), best_path)
    elif raw_path.exists():
        torch.save(torch.load(raw_path, map_location="cpu", weights_only=False), best_path)
    if best_path.exists():
        final_metrics = {
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_f1,
            "best_raw_epoch": best_epoch,
            "best_raw_val_macro_f1": best_val_f1,
            "best_ema_epoch": best_ema_epoch,
            "best_ema_val_macro_f1": (
                best_ema_f1 if best_ema_epoch > 0 else None),
            "selected_weight_variant": selected_variant,
            "selected_val_macro_f1": max(best_val_f1, best_ema_f1),
        }
        final_ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        final_ckpt = enrich_checkpoint(
            final_ckpt, config, class_map, final_metrics, history)
        torch.save(final_ckpt, best_path)
        torch.save(final_ckpt, checkpoint_dir / "best_model.pt")
        create_training_bundle(best_path, config, class_map, history, final_metrics)
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
