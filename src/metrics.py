"""
Competition metric utilities for Open-Set Speaker Identification.

The competition's primary metric is **Macro-Averaged F1 over all 447 classes**
(446 known speakers + 1 aggregated "unknown" class). Predictions are the
argmax over a 447-way probability distribution.

This module is the single source of truth for that metric so that training,
validation, checkpoint selection and the final local evaluation all optimize
the exact quantity the organizers score.
"""

from typing import Dict, Optional, Sequence

import numpy as np
import torch


# ─────────────────────────────────────────────────────────
#  Label-space helpers
# ─────────────────────────────────────────────────────────
#
# Convention used across the project:
#   class index 0            -> "unknown"
#   class indices 1..446     -> known speakers (sorted UUID order)
#
# The speaker (ArcFace) head outputs 446 logits indexed 0..445, i.e.
# speaker-head index j corresponds to global class j + 1.


def speaker_head_to_global(speaker_head_idx: np.ndarray) -> np.ndarray:
    """Map speaker-head indices (0..445) to global class ids (1..446)."""
    return np.asarray(speaker_head_idx) + 1


def global_to_speaker_head(global_idx: np.ndarray) -> np.ndarray:
    """Map global class ids (1..446) to speaker-head indices (0..445).

    Only valid for known classes (>= 1). Class 0 (unknown) has no
    speaker-head counterpart.
    """
    return np.asarray(global_idx) - 1


# ─────────────────────────────────────────────────────────
#  Core metric
# ─────────────────────────────────────────────────────────

def macro_f1_score(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    num_classes: int = 447,
) -> float:
    """Macro-averaged F1 across exactly `num_classes` classes.

    Uses the full label set [0 .. num_classes-1] so that classes absent from
    either y_true or y_pred still count (with F1 = 0), matching the
    competition's macro average over all 447 classes.

    Args:
        y_true: ground-truth global class ids (0 = unknown, 1..446 = known).
        y_pred: predicted global class ids, same convention.
        num_classes: total number of classes (default 447).

    Returns:
        Macro-averaged F1 in [0, 1].
    """
    from sklearn.metrics import f1_score

    labels = list(range(num_classes))
    return float(
        f1_score(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )


def per_class_f1(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    num_classes: int = 447,
) -> np.ndarray:
    """Per-class F1 array of length `num_classes` (for diagnostics)."""
    from sklearn.metrics import f1_score

    labels = list(range(num_classes))
    return np.asarray(
        f1_score(
            y_true,
            y_pred,
            labels=labels,
            average=None,
            zero_division=0,
        )
    )


# ─────────────────────────────────────────────────────────
#  From model outputs (logits) to competition predictions
# ─────────────────────────────────────────────────────────

@torch.no_grad()
def fused_probs_from_logits(
    ood_logits: torch.Tensor,
    speaker_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Fuse the two heads into a proper 447-way probability distribution.

    Formula (matches TwoHeadedSpeakerModel.predict_proba):
        p[0] = sigmoid(ood_logit)
        p[i] = (1 - p[0]) * softmax(speaker_logits / T)[i]

    Args:
        ood_logits:     (batch, 1) raw OOD logits.
        speaker_logits: (batch, 446) cosine/logit scores (no margin).
        temperature:    softmax temperature for the speaker head (calibration).

    Returns:
        probs: (batch, 447) rows summing to 1.
    """
    p_unknown = torch.sigmoid(ood_logits.float())  # (batch, 1)
    p_known = torch.softmax(speaker_logits.float() / max(temperature, 1e-6), dim=1)

    num_known = speaker_logits.size(1)
    p_known_scaled = (1.0 - p_unknown.expand(-1, num_known)) * p_known
    probs = torch.cat([p_unknown, p_known_scaled], dim=1)
    probs = probs.clamp(min=1e-9)
    probs = probs / probs.sum(dim=1, keepdim=True)
    return probs


@torch.no_grad()
def predict_global_classes(
    ood_logits: torch.Tensor,
    speaker_logits: torch.Tensor,
    ood_threshold: Optional[float] = None,
    temperature: float = 1.0,
) -> np.ndarray:
    """Turn head logits into global class predictions (0..446).

    If `ood_threshold` is None (default), predictions are the plain argmax
    over the fused 447-way distribution — exactly what the competition scores.

    If `ood_threshold` is given, a sample whose P(unknown) exceeds the
    threshold is forced to class 0; otherwise the argmax over the fused
    distribution is used. (Useful for local OOD operating-point analysis; the
    competition itself uses plain argmax, so keep None for reporting.)
    """
    probs = fused_probs_from_logits(ood_logits, speaker_logits, temperature)
    preds = probs.argmax(dim=1).cpu().numpy()

    if ood_threshold is not None:
        p_unknown = torch.sigmoid(ood_logits.float()).squeeze(1).cpu().numpy()
        preds = np.where(p_unknown > ood_threshold, 0, preds)
    return preds


@torch.no_grad()
def calibrate_temperature(
    all_ood_logits: torch.Tensor,
    all_speaker_logits: torch.Tensor,
    all_labels: torch.Tensor,
    num_classes: int = 447,
    temps: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """Grid-search the speaker-softmax temperature that maximises Macro-F1.

    `fused_probs_from_logits` already supports a temperature for the speaker
    head; this finds the T that maximises the competition metric (Macro-F1)
    on a validation set, and reports the Macro-F1 at T=1.0 vs best T.

    Args:
        all_ood_logits:     (N, 1)
        all_speaker_logits: (N, 446) — no margin
        all_labels:         (N,) global ids (0 = unknown)
        num_classes:        447
        temps:              temperature grid (default 0.5..2.0 step 0.1)

    Returns:
        {"best_temperature", "macro_f1_at_best_t", "macro_f1_at_t1"}
    """
    if temps is None:
        temps = np.arange(0.5, 2.01, 0.1)

    base = evaluate_macro_f1(
        all_ood_logits, all_speaker_logits, all_labels, num_classes=num_classes,
    )
    best = {"temperature": 1.0, "macro_f1": base["macro_f1"]}
    for t in temps:
        m = evaluate_macro_f1(
            all_ood_logits, all_speaker_logits, all_labels,
            num_classes=num_classes, temperature=float(t),
        )
        if m["macro_f1"] > best["macro_f1"]:
            best = {"temperature": float(t), "macro_f1": m["macro_f1"]}

    return {
        "best_temperature": best["temperature"],
        "macro_f1_at_best_t": best["macro_f1"],
        "macro_f1_at_t1": base["macro_f1"],
    }


# ─────────────────────────────────────────────────────────
#  One-call evaluation helper
# ─────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_macro_f1(
    all_ood_logits: torch.Tensor,
    all_speaker_logits: torch.Tensor,
    all_labels: torch.Tensor,
    num_classes: int = 447,
    ood_threshold: Optional[float] = None,
    temperature: float = 1.0,
) -> Dict[str, float]:
    """Compute the competition metric from concatenated validation outputs.

    Args:
        all_ood_logits:     (N, 1) OOD logits for the whole eval set.
        all_speaker_logits: (N, 446) speaker logits (no margin).
        all_labels:         (N,) global ground-truth ids (0 = unknown).
        num_classes:        447.
        ood_threshold:      optional forced-unknown threshold (see above).
        temperature:        speaker-softmax temperature.

    Returns:
        dict with:
          macro_f1        — the competition metric (primary).
          ood_f1          — binary F1 of the unknown class only (diagnostic).
          known_acc       — accuracy restricted to known-labelled samples.
          overall_acc     — plain accuracy over all 447-way predictions.
    """
    from sklearn.metrics import accuracy_score, f1_score

    y_true = all_labels.cpu().numpy()
    y_pred = predict_global_classes(
        all_ood_logits, all_speaker_logits,
        ood_threshold=ood_threshold, temperature=temperature,
    )

    macro_f1 = macro_f1_score(y_true, y_pred, num_classes=num_classes)

    # Binary unknown-class F1 (diagnostic for the OOD head)
    ood_f1 = float(
        f1_score((y_true == 0).astype(int), (y_pred == 0).astype(int),
                 zero_division=0)
    )

    known_mask = y_true != 0
    known_acc = (
        float(accuracy_score(y_true[known_mask], y_pred[known_mask]))
        if known_mask.sum() > 0 else 0.0
    )
    overall_acc = float(accuracy_score(y_true, y_pred))

    return {
        "macro_f1": macro_f1,
        "ood_f1": ood_f1,
        "known_acc": known_acc,
        "overall_acc": overall_acc,
    }
