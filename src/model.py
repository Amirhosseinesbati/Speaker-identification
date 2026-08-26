"""
Modular Two-Headed Speaker Identification Model.

Architecture:
    Encoder (WavLM/ECAPA/CAM++/ERes2NetV2/TitaNet) → Pooling →
    ├── OOD Head (Linear → Sigmoid)      → P(unknown)
    └── Speaker Head (Linear/ArcFace)     → P(known_i)

Fusion:
    p[0] = P_unknown
    p[i] = (1 - P_unknown) * P_known_i

The model is constructed from composable components via create_model_from_config().
For backward compatibility, TwoHeadedWavLM is kept as an alias.
"""

from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.encoders import BaseEncoder
from src.pooling import StatisticalPooling


class TwoHeadedSpeakerModel(nn.Module):
    """
    Modular two-headed architecture for open-set speaker identification.

    Components are injected via constructor — use create_model_from_config()
    to build from a YAML config dict.

    Output:
        ood_logit:    (batch, 1)       — raw logit, sigmoid → P(unknown)
        speaker_logits: (batch, N)     — logits over known speakers
    """

    def __init__(
        self,
        encoder: BaseEncoder,
        pooling: nn.Module,
        speaker_head: nn.Module,
        ood_head: nn.Module,
        num_known_speakers: int,
        encoder_name: str = "unknown",
        num_unknown_clusters: int = 0,
    ):
        super().__init__()
        self.encoder = encoder
        self.pooling = pooling
        self.head_speaker = speaker_head
        self.head_ood = ood_head
        self.num_known_speakers = num_known_speakers
        self.encoder_name = encoder_name
        # Closed-set 1000-class experiment: when > 0, the speaker head spans
        # num_known_speakers = 446 known + `num_unknown_clusters` pseudo
        # identities recovered by clustering the unlabelled unknown train
        # files. The competition output stays 1 + 446 = 447 classes — the
        # cluster columns are summed into column 0 (unknown) at output time.
        self.num_unknown_clusters = int(num_unknown_clusters)
        if self.num_unknown_clusters > self.num_known_speakers:
            raise ValueError(
                f"num_unknown_clusters={self.num_unknown_clusters} exceeds "
                f"num_known_speakers={self.num_known_speakers} — a speaker head "
                f"cannot collapse more cluster columns than it has outputs."
            )

    @property
    def num_output_classes(self) -> int:
        """Width of the competition output vector (447 for this problem)."""
        return self.num_known_speakers - self.num_unknown_clusters + 1

    def _collapse_probs(self, p_unknown: torch.Tensor, p_known_scaled: torch.Tensor) -> torch.Tensor:
        """(B,1) unknown + (B, N) scaled known → (B, 1 + known_out).

        With num_unknown_clusters=0 this is exactly the legacy concat. With
        clusters, the cluster columns are summed into column 0.
        """
        if self.num_unknown_clusters > 0:
            n_out_known = self.num_known_speakers - self.num_unknown_clusters
            cluster_sum = p_known_scaled[:, n_out_known:].sum(dim=1, keepdim=True)
            return torch.cat([p_unknown + cluster_sum, p_known_scaled[:, :n_out_known]], dim=1)
        return torch.cat([p_unknown, p_known_scaled], dim=1)

    def forward(
        self,
        waveforms: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_embedding: bool = False,
    ):
        """
        Args:
            waveforms: (batch, 1, T) — raw audio, 16kHz
            labels:    Optional (batch,) — speaker labels for ArcFace training.
                       None for inference.
            return_embedding: also return the L2-normalised speaker embedding
                       (ArcFace projection, or pooled features for linear head) —
                       used by the prototypical loss.

        Returns:
            ood_logit:      (batch, 1)  — raw logit (sigmoid → P(unknown))
            speaker_logits: (batch, N)  — logits over known speakers
            embedding:      (batch, D)  — only when return_embedding=True
        """
        # ── Encoder ──
        hidden_states, lengths = self.encoder(waveforms)
        # hidden_states: (batch, seq_len, hidden_dim)

        # ── Pooling ──
        pooled = self.pooling(hidden_states)  # (batch, pooled_dim)

        # ── OOD Head ──
        # In cluster mode (num_unknown_clusters > 0) the head is disabled by
        # default (model.ood_head=false): the pseudo-identity columns already
        # encode P(unknown) via the collapse, and their BCE supervision would
        # be distorted (unknown files are relabeled to cluster ids). The
        # legacy 447-way path keeps the OOD head exactly as before.
        ood_logit = self.head_ood(pooled) if self.head_ood is not None else None  # (batch, 1) | None

        # ── Speaker Head ──
        # ArcFace needs labels during training, Linear ignores them
        if labels is not None and hasattr(self.head_speaker, 'forward'):
            import inspect
            sig = inspect.signature(self.head_speaker.forward)
            if 'labels' in sig.parameters:
                # Remap one-indexed metric labels to zero-indexed ArcFace
                # labels.  In ``speaker_target_scope=known`` experiments the
                # data can still carry pseudo-OOD ids above the 446-class head;
                # map those rows to the harmless dummy class 0.  TwoPartLoss
                # masks them from speaker CE while an auxiliary metric loss can
                # still learn from their returned embeddings.
                #
                # This collision is HARMLESS because:
                # (a) TwoPartLoss ignores every label outside this head's
                #     target scope, so its speaker-logit gradient is always 0.
                # (b) ArcFace weight[0] is ONLY trained by speaker #1 samples.
                # (c) The ArcFace margin on unknown's output[0] never backprops.
                remapped = torch.zeros_like(labels)
                mask_in_head = (labels > 0) & (labels <= self.num_known_speakers)
                remapped[mask_in_head] = labels[mask_in_head] - 1
                speaker_logits = self.head_speaker(pooled, labels=remapped)
            else:
                speaker_logits = self.head_speaker(pooled)
        else:
            speaker_logits = self.head_speaker(pooled)

        if return_embedding:
            if hasattr(self.head_speaker, "embedding_proj"):
                emb = F.normalize(self.head_speaker.embedding_proj(pooled), p=2, dim=1)
            else:
                emb = F.normalize(pooled, p=2, dim=1)
            return ood_logit, speaker_logits, emb

        return ood_logit, speaker_logits

    def _embed_single(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Raw (unnormalised) speaker embedding for a (B, 1, T) batch.

        Uses the ArcFace head's ``embedding_proj`` output (the L2-normalisable
        speaker space, default 192-d) when present; otherwise falls back to the
        pooled encoder features (linear-head models).
        """
        hidden_states, _ = self.encoder(waveforms)
        pooled = self.pooling(hidden_states)
        if hasattr(self.head_speaker, "embedding_proj"):
            return self.head_speaker.embedding_proj(pooled)
        return pooled

    def embed(self, waveforms: torch.Tensor) -> torch.Tensor:
        """L2-normalised speaker embedding (centroid / cosine-decision space).

        Multi-window inputs ``(B, W, 1, T)`` are embedded window-by-window and
        **averaged before normalisation** (``mean_then_l2norm``) so the result
        matches how centroids are built and how inference scores cosine
        similarity. Single-window ``(B, 1, T)`` inputs are embedded directly.

        Returns:
            emb: (batch, D) rows of unit norm (D = embedding_dim, default 192).
        """
        if waveforms.dim() == 4:
            B, W = waveforms.shape[0], waveforms.shape[1]
            embs = [self._embed_single(waveforms[:, w]) for w in range(W)]
            emb = torch.stack(embs, dim=0).mean(dim=0)  # (B, D)
        else:
            emb = self._embed_single(waveforms)
        return F.normalize(emb, p=2, dim=1)

    def predict_proba(self, waveforms: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Get proper probability vector over all classes (0..num_known).

        Formula:
            p[0] = sigmoid(ood_logit)
            p[i] = (1 - p[0]) * softmax(speaker_logits / temperature)[i]

        Args:
            waveforms: (batch, 1, T) or (batch, W, 1, T) — multi-window inputs
                       are run window-by-window and the logits averaged.
            temperature: speaker-softmax temperature (calibration knob, T≥~1e-6).

        Returns:
            probs: (batch, 1 + num_known) — sum(dim=1) ≈ 1.0
        """
        if waveforms.dim() == 4:
            # Loop windows (peak stays at (B, 1, T)) and average the LOGITS,
            # then fuse once — identical math to a flattened (B*W, ...) batch.
            B, W = waveforms.shape[0], waveforms.shape[1]
            ood_sum = spk_sum = None
            for w in range(W):
                o, s = self.forward(waveforms[:, w], labels=None)
                if o is not None:  # OOD head disabled (cluster mode)
                    ood_sum = o if ood_sum is None else ood_sum + o
                spk_sum = s if spk_sum is None else spk_sum + s
            ood_logit = None if ood_sum is None else ood_sum / W
            speaker_logits = spk_sum / W
        else:
            ood_logit, speaker_logits = self.forward(waveforms, labels=None)

        # P(unknown) = sigmoid(ood_logit); with the OOD head disabled the
        # unknown mass comes entirely from the cluster collapse (p_unknown=0).
        if ood_logit is not None:
            p_unknown = torch.sigmoid(ood_logit)  # (batch, 1)
        else:
            p_unknown = torch.zeros(speaker_logits.shape[0], 1,
                                    device=speaker_logits.device)

        # P(known_i) = softmax(speaker_logits / temperature)
        p_known = F.softmax(speaker_logits / max(float(temperature), 1e-6), dim=1)  # (batch, N)

        # Fusion: p_0 = P_unknown, p_i = (1 - P_unknown) * P_known_i
        p_unknown_expanded = p_unknown.expand(-1, self.num_known_speakers)
        p_known_scaled = (1.0 - p_unknown_expanded) * p_known

        # Concatenate: (batch, 1 + N) = (batch, 447); with clusters the 554
        # pseudo-identity columns are summed into column 0 (unknown).
        probs = self._collapse_probs(p_unknown, p_known_scaled)

        # Numerical safety
        probs = torch.clamp(probs, min=1e-7, max=1.0 - 1e-7)
        probs = probs / probs.sum(dim=1, keepdim=True)

        return probs

    def predict_proba_and_embed(
        self,
        waveforms: torch.Tensor,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Head probabilities AND the L2-normalised speaker embedding from ONE
        encoder forward (used by the submission decision layer — avoids a second
        encoder pass and keeps inference inside the time budget).

        Input ``(W, 1, T)`` is treated as W windows of a single file:
          - probs: window-averaged 447-way probabilities (probability-averaging,
            matching ``predict_proba(...).mean(0)`` exactly).
          - emb:   mean(window embeddings) then L2-normalised (``mean_then_l2norm``),
            matching ``embed(batch.unsqueeze(0))`` exactly.

        Returns:
            probs: (1 + num_known,)  rows sum to 1.
            emb:   (embedding_dim,)  unit norm.
        """
        probs, emb, _ = self.predict_proba_embed_and_evidence(
            waveforms, temperature=temperature,
        )
        return probs, emb

    def predict_proba_embed_and_evidence(
        self,
        waveforms: torch.Tensor,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Inference probabilities, embedding, and pre-collapse open-set evidence.

        ``speaker_probs`` retains the pseudo-unknown tail instead of summing all
        of it into class 0.  This permits cardinality-normalised decision rules
        (for example top-k mean evidence) while preserving the existing
        ``predict_proba_and_embed`` API for training and legacy submissions.
        """
        if waveforms.dim() == 4:
            waveforms = waveforms.reshape(-1, 1, waveforms.size(-1))

        hidden_states, _ = self.encoder(waveforms)
        pooled = self.pooling(hidden_states)          # (W, pooled_dim)
        ood_logit = self.head_ood(pooled) if self.head_ood is not None else None  # (W, 1) | None
        speaker_logits = self.head_speaker(pooled)    # (W, N)
        if hasattr(self.head_speaker, "embedding_proj"):
            raw_emb = self.head_speaker.embedding_proj(pooled)  # (W, D)
        else:
            raw_emb = pooled

        # Per-window probabilities → average (prob-averaging, existing TTA path).
        if ood_logit is not None:
            p_unknown = torch.sigmoid(ood_logit)      # (W, 1)
        else:
            p_unknown = torch.zeros(pooled.shape[0], 1, device=pooled.device)
        p_known = F.softmax(speaker_logits / max(float(temperature), 1e-6), dim=1)
        p_known_scaled = (1.0 - p_unknown.expand(-1, self.num_known_speakers)) * p_known
        probs = self._collapse_probs(p_unknown, p_known_scaled)  # (W, 447)
        probs = torch.clamp(probs, min=1e-7, max=1.0 - 1e-7)
        probs = probs / probs.sum(dim=1, keepdim=True)
        probs = probs.mean(dim=0)                     # (447,)

        # mean_then_l2norm embedding over windows.
        emb = F.normalize(raw_emb.mean(dim=0, keepdim=True), p=2, dim=1)[0]  # (D,)
        speaker_probs = p_known.mean(dim=0)
        ood_prob = p_unknown.mean()
        num_competition_known = int(self.num_output_classes) - 1
        window_top = p_known[:, :num_competition_known].argmax(dim=1)
        aggregate_top = speaker_probs[:num_competition_known].argmax()
        window_agreement = (window_top == aggregate_top).float().mean()
        evidence = {
            "speaker_probs": speaker_probs,
            "ood_prob": ood_prob,
            "window_agreement": window_agreement,
        }
        return probs, emb, evidence

    def get_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def print_summary(self):
        """Print model architecture summary."""
        total = sum(p.numel() for p in self.parameters())
        trainable = self.get_trainable_params()
        frozen = total - trainable



# ═══════════════════════════════════════════════════════════
#  Backward Compatibility: TwoHeadedWavLM
# ═══════════════════════════════════════════════════════════

class TwoHeadedWavLM(TwoHeadedSpeakerModel):
    """
    Backward-compatible wrapper that builds the original WavLM architecture
    from a flat config dict (old-style or new-style).

    This class exists so existing checkpoint loading code and ZenML steps
    continue to work without modification.
    """

    def __init__(
        self,
        config: dict,
        num_known_speakers: int = 446,
    ):
        from src.encoders import WavLMEncoder
        from src.pooling import create_pooling
        from src.heads import OODHead, LinearSpeakerHead, create_speaker_head, create_ood_head

        model_cfg = config["model"]

        # ── Encoder ──
        encoder_type = model_cfg.get("encoder_type", "wavlm")
        if "encoder_config" in model_cfg:
            enc_cfg = model_cfg["encoder_config"].get(encoder_type, {})
            base_model = enc_cfg.get(
                "base_model",
                model_cfg.get("base_model", "microsoft/wavlm-base-plus"),
            )
            freeze_fe = enc_cfg.get(
                "freeze_feature_extractor",
                model_cfg.get("freeze_feature_extractor", False),
            )
        else:
            base_model = model_cfg.get("base_model", "microsoft/wavlm-base-plus")
            freeze_fe = model_cfg.get("freeze_feature_extractor", False)

        encoder = WavLMEncoder(
            base_model=base_model,
            freeze_feature_extractor=freeze_fe,
        )

        # ── Pooling ──
        pooling_type = model_cfg.get("pooling_type", "statistical")
        pooling = create_pooling(pooling_type, encoder.output_dim)
        pooled_dim = encoder.output_dim * pooling.output_multiplier

        # ── OOD Head ──
        ood_head = create_ood_head(config, pooled_dim)

        # ── Speaker Head ──
        speaker_head = create_speaker_head(config, pooled_dim, num_known_speakers)

        super().__init__(
            encoder=encoder,
            pooling=pooling,
            speaker_head=speaker_head,
            ood_head=ood_head,
            num_known_speakers=num_known_speakers,
            encoder_name=base_model,
            num_unknown_clusters=model_cfg.get("num_unknown_clusters", 0),
        )

    # Keep old method signatures for backward compat
    def _freeze_feature_extractor(self):
        self.encoder.freeze()

    def unfreeze_feature_extractor(self):
        self.encoder.unfreeze()

    @property
    def wavlm(self):
        """Backward-compat: expose encoder.wavlm."""
        return self.encoder.wavlm


# ═══════════════════════════════════════════════════════════
#  Smoke Test
