"""
Classification Heads for Speaker Identification.

Classes:
    OODHead              — Binary OOD/known detector (Linear → Sigmoid)
    LinearSpeakerHead    — Linear classifier for known speakers (current behavior)
    ArcFaceHead          — Additive Angular Margin head (Phase 2.2)
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════
#  OOD (Out-of-Distribution) Head
# ═══════════════════════════════════════════════════════════

class OODHead(nn.Module):
    """
    Binary classifier: detects whether a speaker is unknown (class 0).

    Architecture:
        LayerNorm → Linear(hidden) → ReLU → Dropout → Linear(→1)

    Output: single logit — sigmoid(output) = P(unknown)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.7):
        super().__init__()
        self.input_dim = input_dim

        if hidden_dim > 0:
            self.mlp = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.mlp = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Dropout(dropout),
                nn.Linear(input_dim, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim)

        Returns:
            logits: (batch, 1) — raw logit (use sigmoid for probability)
        """
        return self.mlp(x)


# ═══════════════════════════════════════════════════════════
#  Linear Speaker Head
# ═══════════════════════════════════════════════════════════

class LinearSpeakerHead(nn.Module):
    """
    Simple linear classifier for known speaker identification.

    Architecture:
        LayerNorm → Linear(input_dim → num_classes)

    Output: logits over known speaker classes (0..num_known-1).
    """

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, num_classes),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (batch, input_dim)

        Returns:
            logits: (batch, num_classes)
        """
        return self.classifier(embeddings)


# ═══════════════════════════════════════════════════════════
#  ArcFace Head (Additive Angular Margin)
# ═══════════════════════════════════════════════════════════

class ArcFaceHead(nn.Module):
    """
    Additive Angular Margin Softmax for speaker recognition.

    Formula: s * cos(theta + m)

    where theta is the angle between the normalized embedding and the
    class weight vector, m is the angular margin, and s is the scale.

    This produces well-separated, L2-normalized speaker embeddings
    suitable for cosine-similarity based OOD detection (FAISS).

    Reference:
        Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face
        Recognition" (CVPR 2019) — adapted for speaker recognition.

    Args:
        input_dim:   Dimension of the pooled embedding (e.g., 1536 for WavLM+pool)
        num_classes: Number of known speakers
        embedding_dim: Projection dim before ArcFace (default: 192)
        margin:      Angular margin in radians (default: 0.3)
        scale:       Feature scale (default: 15.0)
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        embedding_dim: int = 192,
        margin: float = 0.3,
        scale: float = 15.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.m = margin
        self.s = scale

        # Pre-compute constants for cos(theta + m)
        import math
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

        # Project to embedding space
        self.embedding_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, embedding_dim),
            nn.Dropout(0.2),
        )

        # Classification weight matrix (L2-normalized rows after each forward)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_normal_(self.weight, gain=1)

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            embeddings: (batch, input_dim) — pooled encoder output
            labels:     (batch,) — speaker labels (0..num_classes-1), REQUIRED for
                        training (ArcFace needs labels to apply margin).
                        If None, returns cosine logits without margin (for inference).

        Returns:
            logits: (batch, num_classes) — margin-adjusted logits (training)
                    or raw cosine logits (inference, no labels)
        """
        # Project to embedding space + L2 normalize
        emb = self.embedding_proj(embeddings)
        emb_norm = F.normalize(emb, p=2, dim=1)  # (batch, embedding_dim)
        weight_norm = F.normalize(self.weight, p=2, dim=1)  # (num_classes, embedding_dim)

        # Cosine similarity: (batch, num_classes)
        cosine = F.linear(emb_norm, weight_norm)
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        if labels is None:
            # Inference mode: return cosine logits (will apply softmax externally)
            return cosine * self.s

        # Training mode: apply angular margin
        # sin(theta) = sqrt(1 - cos^2(theta))
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))

        # cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        # Numerical stability: where cos(theta) > cos(pi - m),
        # use cos(theta) - m*sin(pi - m) instead for monotonicity
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # One-hot encode labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # Replace target class with margin-adjusted cosine
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s

        return output


# ═══════════════════════════════════════════════════════════
#  Sub-center ArcFace Head
# ═══════════════════════════════════════════════════════════

class SubCenterArcFaceHead(nn.Module):
    """
    Sub-center ArcFace (Additive Angular Margin with K sub-centers per class).

    Standard ArcFace keeps ONE centre per speaker. With only ~4-5 files per
    speaker, a single centre is sensitive to intra-speaker variability
    (different devices / environments / codecs across a speaker's files).
    Sub-center ArcFace keeps ``sub_centers`` centres per class and uses the
    **max cosine** to any of them as the class score — it is more robust for
    few-shot classes (root causes R1/R3, C3).

    Formula:  s * cos(theta_j + m)   where theta_j = argmax_k angle to w_{j,k}.

    Reference:
        Deng et al., "Sub-center ArcFace: Boosting Face Recognition by
        Large-scale Noisy Web Faces" (ECCV 2020).
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        embedding_dim: int = 192,
        margin: float = 0.3,
        scale: float = 32.0,
        sub_centers: int = 3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.m = margin
        self.s = scale
        self.K = max(1, int(sub_centers))

        import math
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

        # Project to embedding space
        self.embedding_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, embedding_dim),
            nn.Dropout(0.2),
        )

        # (num_classes * K, embedding_dim) — reshaped to (num_classes, K, D).
        self.weight = nn.Parameter(
            torch.FloatTensor(num_classes * self.K, embedding_dim)
        )
        nn.init.xavier_normal_(self.weight, gain=1)

    def _max_cosine(self, embeddings: torch.Tensor) -> torch.Tensor:
        """(batch, embedding_dim) → (batch, num_classes) max-over-sub-centers."""
        emb = self.embedding_proj(embeddings)
        emb_norm = F.normalize(emb, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(emb_norm, weight_norm)          # (batch, num_classes*K)
        cosine = cosine.view(-1, self.num_classes, self.K)  # (batch, num_classes, K)
        cosine, _ = cosine.max(dim=2)                     # (batch, num_classes)
        return cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            embeddings: (batch, input_dim) — pooled encoder output
            labels:     (batch,) — 0..num_classes-1 speaker labels. REQUIRED for
                        training (margin). If None, returns margin-free cosine
                        logits (inference).

        Returns:
            logits: (batch, num_classes)
        """
        cosine = self._max_cosine(embeddings)

        if labels is None:
            return cosine * self.s

        # ArcFace margin applied to the max sub-centre of the target class.
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.s


# ═══════════════════════════════════════════════════════════
#  Head Factories
# ═══════════════════════════════════════════════════════════

def create_ood_head(config: dict, input_dim: int) -> OODHead:
    """Build OOD head from config."""
    ood_cfg = config["model"].get("ood_head_config", {})
    hidden_dim = ood_cfg.get("hidden_dim", 256)
    return OODHead(input_dim, hidden_dim)


def create_speaker_head(config: dict, input_dim: int, num_known: int) -> nn.Module:
    """
    Build speaker classification head from config.

    Supports: "linear" | "arcface" | "arcface_subcenter"
    """
    model_cfg = config["model"]
    head_type = model_cfg.get("speaker_head_type", "linear").lower().strip()

    if head_type == "linear":
        return LinearSpeakerHead(input_dim, num_known)

    elif head_type == "arcface":
        arc_cfg = model_cfg.get("speaker_head_config", {}).get("arcface", {})
        return ArcFaceHead(
            input_dim=input_dim,
            num_classes=num_known,
            embedding_dim=arc_cfg.get("embedding_dim", 192),
            margin=arc_cfg.get("margin", 0.3),
            scale=arc_cfg.get("scale", 15.0),
        )

    elif head_type == "arcface_subcenter":
        arc_cfg = model_cfg.get("speaker_head_config", {}).get("arcface_subcenter", {})
        return SubCenterArcFaceHead(
            input_dim=input_dim,
            num_classes=num_known,
            embedding_dim=arc_cfg.get("embedding_dim", 192),
            margin=arc_cfg.get("margin", 0.3),
            scale=arc_cfg.get("scale", 32.0),
            sub_centers=arc_cfg.get("sub_centers", 3),
        )

    else:
        raise ValueError(
            f"Unknown speaker_head_type: '{head_type}'. "
            f"Expected 'linear', 'arcface' or 'arcface_subcenter'."
        )


# ═══════════════════════════════════════════════════════════
#  Smoke Test
