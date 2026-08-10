"""
Speaker Encoder Backbones.

Each encoder accepts raw audio waveforms and produces frame-level hidden states.
All encoders share a common interface for seamless swapping.

Classes:
    BaseEncoder           — Abstract interface
    WavLMEncoder          — Microsoft WavLM (HuggingFace), base or large
    ECAPAEncoder          — ECAPA-TDNN (SpeechBrain)
    HuBERTEncoder         — Facebook HuBERT (HuggingFace) — REMOVED in Phase 3
    CAMPlusPlusEncoder    — CAM++ (ModelScope) — Phase 2a
    ERes2NetV2Encoder     — ERes2NetV2 (ModelScope) — Phase 2c
    TitaNetEncoder        — TitaNet-Large (NeMo) — Phase 2d

All encoders load from LOCAL paths at inference time (offline). Hub downloads
happen only on the dev machine when ``allow_hub_download=True``.
"""

import os
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import WavLMModel


# ═══════════════════════════════════════════════════════════
#  Offline source resolution
# ═══════════════════════════════════════════════════════════

def resolve_model_source(
    enc_cfg: dict,
    hub_default: str,
) -> Tuple[Optional[str], str, bool]:
    """
    Resolve where an encoder loads its weights from (offline-first).

    Priority:
        1. ``enc_cfg.local_path`` — the ONLY source used at submission time
           (inference is fully offline; no hub calls are made).
        2. Hub id + ``allow_hub_download=True`` — dev-machine mode (training),
           where weights are fetched once and then copied into ``weights/``.

    Args:
        enc_cfg: Per-encoder config dict (``model.encoder_config.<type>``).
        hub_default: Fallback hub id if the config omits one.

    Returns:
        (local_path, hub_id, allow_hub_download)
    """
    local_path = enc_cfg.get("local_path")
    hub_id = (
        enc_cfg.get("hub_id")
        or enc_cfg.get("model_id")
        or enc_cfg.get("base_model")
        or enc_cfg.get("source")
        or hub_default
    )
    allow_hub = bool(enc_cfg.get("allow_hub_download", False))
    return local_path, hub_id, allow_hub


def is_offline_mode() -> bool:
    """True when any hub-offline env flag is set (used to fail loudly)."""
    return any(
        os.environ.get(k, "") == "1"
        for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "MODELSCOPE_OFFLINE")
    )


# ═══════════════════════════════════════════════════════════
#  Base interface
# ═══════════════════════════════════════════════════════════

class BaseEncoder(nn.Module, ABC):
    """Abstract encoder interface for speaker recognition backbones."""

    @abstractmethod
    def forward(
        self,
        waveforms: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            waveforms: (batch, 1, T) — raw audio, 16kHz mono

        Returns:
            hidden_states: (batch, seq_len, hidden_dim)
            lengths: Optional (batch,) — valid frame lengths (for masking)
        """
        ...

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Hidden dimension of the encoder output."""
        ...

    @abstractmethod
    def freeze(self) -> None:
        """Freeze encoder parameters (for feature extraction mode)."""
        ...

    @abstractmethod
    def unfreeze(self) -> None:
        """Unfreeze encoder parameters (for fine-tuning)."""
        ...


# ═══════════════════════════════════════════════════════════
#  WavLM Encoder
# ═══════════════════════════════════════════════════════════

class WavLMEncoder(BaseEncoder):
    """
    Microsoft WavLM encoder for speaker recognition.

    WavLM is a HuBERT-based model with gated relative position bias
    and utterance mixing, designed to preserve speaker identity.

    Supported models:
        microsoft/wavlm-base       (94M, 768-dim)
        microsoft/wavlm-base-plus  (94M, 768-dim)
        microsoft/wavlm-large      (317M, 1024-dim) ← default (Phase 2b)

    Offline loading: pass ``local_path`` pointing at a snapshot directory
    (``weights/wavlm_large/``) — the model is then loaded with
    ``local_files_only=True`` and never touches the hub.
    """

    def __init__(
        self,
        base_model: str = "microsoft/wavlm-large",
        freeze_feature_extractor: bool = True,
        local_path: Optional[str] = None,
        allow_hub_download: bool = False,
    ):
        super().__init__()
        self.base_model_name = base_model
        self.local_path = local_path

        if local_path is not None:
            # Offline / submission path — never hit the hub.
            if not os.path.isdir(local_path):
                raise FileNotFoundError(
                    f"WavLM local weights dir not found: {local_path}. "
                    "Run `python scripts/download_all_weights.py` on the dev "
                    "machine first."
                )
            self.wavlm = WavLMModel.from_pretrained(
                local_path, local_files_only=True,
            )
            print(f"  ⬇️  WavLM: loaded from local {local_path}")
        elif allow_hub_download:
            self.wavlm = WavLMModel.from_pretrained(base_model)
            print(f"  ⬇️  WavLM: downloaded from hub {base_model}")
        else:
            raise RuntimeError(
                f"WavLM '{base_model}': no local_path and allow_hub_download=False. "
                "At inference the model must load from a local directory; on the "
                "dev machine set allow_hub_download=True."
            )

        if freeze_feature_extractor:
            self.freeze()
            print(f"  🔒 WavLM feature extractor: FROZEN")
        else:
            print(f"  🔓 WavLM feature extractor: UNFROZEN (full fine-tune)")

    def forward(
        self,
        waveforms: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            waveforms: (batch, 1, T)

        Returns:
            hidden_states: (batch, seq_len, hidden_dim)
            lengths: None (WavLM handles masking internally when attention_mask
                     is not provided)
        """
        # WavLM expects (batch, T) — squeeze channel dim
        input_values = waveforms.squeeze(1)  # (batch, T)

        outputs = self.wavlm(
            input_values=input_values,
            output_hidden_states=False,
        )
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)
        return hidden_states, None

    @property
    def output_dim(self) -> int:
        return self.wavlm.config.hidden_size

    def freeze(self) -> None:
        """Freeze only the CNN feature extractor (stem). Transformer stays trainable."""
        if hasattr(self.wavlm, "feature_extractor"):
            for param in self.wavlm.feature_extractor.parameters():
                param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze CNN feature extractor for full fine-tuning."""
        if hasattr(self.wavlm, "feature_extractor"):
            for param in self.wavlm.feature_extractor.parameters():
                param.requires_grad = True
        print("  🔓 WavLM feature extractor UNFROZEN.")


# ═══════════════════════════════════════════════════════════
#  HuBERT Encoder
#  ⚠️  REMOVED in Phase 3 — kept here only until the removal commit.
# ═══════════════════════════════════════════════════════════

class HuBERTEncoder(BaseEncoder):
    """
    Facebook HuBERT encoder for speaker recognition. (REMOVED — Phase 3)

    Deprecated: this encoder is being removed from the ensemble. The class is
    kept in the tree only so the Phase 3 removal commit can delete it cleanly.
    """

    def __init__(
        self,
        base_model: str = "facebook/hubert-large-ls960-ft",
        freeze_feature_extractor: bool = True,
    ):
        super().__init__()
        self.base_model_name = base_model

        from transformers import HubertModel
        self.hubert = HubertModel.from_pretrained(base_model)

        if freeze_feature_extractor:
            self.freeze()
            print(f"  🔒 HuBERT feature extractor: FROZEN")
        else:
            print(f"  🔓 HuBERT feature extractor: UNFROZEN")

    def forward(
        self,
        waveforms: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_values = waveforms.squeeze(1)  # (batch, T)
        outputs = self.hubert(input_values=input_values, output_hidden_states=False)
        return outputs.last_hidden_state, None

    @property
    def output_dim(self) -> int:
        return self.hubert.config.hidden_size

    def freeze(self) -> None:
        if hasattr(self.hubert, "feature_extractor"):
            for param in self.hubert.feature_extractor.parameters():
                param.requires_grad = False

    def unfreeze(self) -> None:
        if hasattr(self.hubert, "feature_extractor"):
            for param in self.hubert.feature_extractor.parameters():
                param.requires_grad = True
        print("  🔓 HuBERT feature extractor UNFROZEN.")


# ═══════════════════════════════════════════════════════════
#  ECAPA-TDNN Encoder (SpeechBrain)
# ═══════════════════════════════════════════════════════════

def _patch_speechbrain_lazy_modules():
    """
    SpeechBrain (>=1.0) lazily exports optional-dependency modules
    (e.g. ``speechbrain.integrations.k2_fsa`` → requires ``k2``, which is not
    installed). Any attribute access on these LazyModules — including the
    ``hasattr(module, '__file__')`` that ``inspect.getmodule``/``inspect.stack``
    performs — triggers ``__getattr__`` → import of the missing dependency →
    ``ImportError``.

    This breaks unrelated libraries whose import machinery inspects the stack:
    - ``lazy_loader`` (used by librosa 0.11) calls ``inspect.stack()`` while
      importing ``librosa.core.audio`` and walks frames touching every module
      in ``sys.modules``, including speechbrain's LazyModules.

    We neutralise this by force-loading every speechbrain LazyModule and
    replacing the ones that fail (missing optional deps) with plain module
    stubs, so attribute access never raises. The stubbed targets are optional
    integrations the project does not use.
    """
    import sys
    import types

    try:
        from speechbrain.utils.importutils import LazyModule
    except Exception:
        return

    for name, mod in list(sys.modules.items()):
        if not isinstance(mod, LazyModule):
            continue
        try:
            _ = mod.__file__  # force-load if possible
        except Exception:
            stub = types.ModuleType(name)
            stub.__file__ = "<stub>"
            sys.modules[name] = stub
            # also patch the attribute on the parent package if present
            if "." in name:
                parent, attr = name.rsplit(".", 1)
                if parent in sys.modules:
                    try:
                        setattr(sys.modules[parent], attr, stub)
                    except Exception:
                        pass


class ECAPAEncoder(BaseEncoder):
    """
    ECAPA-TDNN encoder from SpeechBrain, pretrained on VoxCeleb.

    ECAPA-TDNN is a state-of-the-art speaker embedding model that uses:
    - Squeeze-Excitation Res2Blocks
    - Multi-layer feature aggregation
    - Attentive Statistical Pooling (built-in)

    The built-in pooling produces utterance-level 192-dim embeddings.
    When using this encoder, set pooling_type="identity" in config.

    Reference:
        Desplanques et al., "ECAPA-TDNN: Emphasized Channel Attention,
        Propagation and Aggregation in TDNN Based Speaker Verification"
        (INTERSPEECH 2020)

    Supported sources:
        speechbrain/spkrec-ecapa-voxceleb  (192-dim, VoxCeleb1+2, 0.80% EER)

    Offline loading: pass ``local_path`` pointing at the SpeechBrain savedir
    (``weights/ecapa/`` containing hyperparams.yaml + model.ckpt + normalizer).
    ``EncoderClassifier.from_hparams(source=local_path, savedir=local_path)``
    then reads everything from disk with no hub fetch (C1).
    """

    def __init__(
        self,
        source: str = "speechbrain/spkrec-ecapa-voxceleb",
        freeze_encoder: bool = True,
        unfreeze_last_n_blocks: int = 0,
        local_path: Optional[str] = None,
        allow_hub_download: bool = False,
    ):
        super().__init__()
        self.source = source
        self.local_path = local_path
        self._output_dim = 192  # ECAPA-TDNN embedding dimension
        self._frozen = freeze_encoder
        self._unfreeze_last_n_blocks = max(0, int(unfreeze_last_n_blocks))

        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

        import speechbrain  # noqa: F401  (registers lazy modules)
        # Neutralise speechbrain's broken lazy modules (k2_fsa etc.) before
        # any further import/load — see _patch_speechbrain_lazy_modules docstring.
        _patch_speechbrain_lazy_modules()

        from speechbrain.inference.speaker import EncoderClassifier

        # ── Source resolution: local savedir (offline) vs hub (dev) ──
        if local_path is not None:
            if not os.path.isdir(local_path):
                raise FileNotFoundError(
                    f"ECAPA savedir not found: {local_path}. Run "
                    "`python scripts/download_all_weights.py` on the dev machine."
                )
            src = local_path
            print(f"  ⬇️  ECAPA-TDNN: loading from local savedir {local_path}")
        elif allow_hub_download:
            src = source
            print(f"  ⬇️  ECAPA-TDNN: downloading from hub {source}")
        else:
            raise RuntimeError(
                f"ECAPA '{source}': no local_path and allow_hub_download=False. "
                "At inference the model must load from a local directory; on the "
                "dev machine set allow_hub_download=True."
            )

        # Load on CPU first — device will be synced via self.to()
        self.classifier = EncoderClassifier.from_hparams(
            source=src,
            savedir=src if local_path is not None else None,
            run_opts={"device": "cpu"},
        )

        if freeze_encoder:
            self.freeze()
            print(f"  🔒 ECAPA-TDNN encoder: FROZEN")
        elif self._unfreeze_last_n_blocks > 0:
            self.unfreeze_last_n_blocks(self._unfreeze_last_n_blocks)
        else:
            self.unfreeze()
            print(f"  🔓 ECAPA-TDNN encoder: UNFROZEN (full fine-tune)")

        # Put SpeechBrain modules in eval mode. Even when the outer
        # model.train() is called, this encoder stays in eval to prevent
        # BatchNorm running statistics from being corrupted by training data.
        self.classifier.mods.eval()
        self.eval()  # also set nn.Module training flag

    def to(self, *args, **kwargs):
        """
        Override nn.Module.to() to also move the SpeechBrain classifier.

        SpeechBrain's EncoderClassifier is NOT an nn.Module — it's a
        Pretrainer wrapper that stores its own `device` attribute.
        Without this override, model.to('cuda') leaves the internal
        ECAPA-TDNN modules on CPU, causing device mismatch errors.
        """
        super().to(*args, **kwargs)
        if hasattr(self, 'classifier'):
            self.classifier.to(*args, **kwargs)
        return self

    def train(self, mode: bool = True):
        """
        Override to keep encoder frozen + eval even when outer model trains.
        This prevents BatchNorm running stats from being silently corrupted
        by training data flowing through the frozen encoder.
        """
        super().train(False)  # encoder always stays in eval
        return self

    def forward(
        self,
        waveforms: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Bypasses SpeechBrain's encode_batch() to avoid:
          - Implicit wav.to(self.device) that causes CPU/CUDA mismatch
          - Implicit wav.float() inside AMP autocast → Half conversion crash
          - BatchNorm stats corruption when outer model is in train mode

        Directly calls: compute_features → mean_var_norm → embedding_model
        """
        # SpeechBrain modules are always in eval (see __init__ + train() override)
        mods = self.classifier.mods
        dev = next(mods.embedding_model.parameters()).device

        # (batch, 1, T) → (batch, T)
        wav = waveforms.squeeze(1).to(dev)

        # Full-length assumption (no padding)
        wav_lens = torch.ones(wav.shape[0], device=dev)

        if self._frozen:
            # Fully frozen encoder → pure feature extraction, no graph kept.
            with torch.no_grad():
                feats = mods.compute_features(wav)
                feats = mods.mean_var_norm(feats, wav_lens)
                embeddings = mods.embedding_model(feats, wav_lens)  # (batch, 192)
        else:
            # Partially unfrozen (fine-tuning) → keep the graph so gradients
            # flow back into the last blocks.
            feats = mods.compute_features(wav)
            feats = mods.mean_var_norm(feats, wav_lens)
            embeddings = mods.embedding_model(feats, wav_lens)  # (batch, 192)

        # If output has extra dim, squeeze it
        if embeddings.ndim == 3:
            embeddings = embeddings.squeeze(1)  # (batch, 192)

        # Add dummy sequence dimension for pooling compatibility
        # IdentityPooling will squeeze this back
        hidden_states = embeddings.unsqueeze(1)  # (batch, 1, 192)

        return hidden_states, None

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def freeze(self) -> None:
        """Freeze all ECAPA-TDNN parameters."""
        for param in self.classifier.parameters():
            param.requires_grad = False
        self._frozen = True

    def unfreeze(self) -> None:
        """Unfreeze all ECAPA-TDNN parameters (full fine-tune)."""
        for param in self.classifier.parameters():
            param.requires_grad = True
        self._frozen = False
        print("  🔓 ECAPA-TDNN encoder UNFROZEN.")

    def unfreeze_last_n_blocks(self, n: int = 2) -> None:
        """
        Unfreeze only the last `n` SE-Res2Blocks of the ECAPA-TDNN trunk.

        Everything else (feature extractor, MFA, attentive pooling, final fc)
        stays frozen, so the trainable parameter count (and VRAM) stays small
        enough to fine-tune on a 6 GB GPU. BatchNorm layers are kept in eval
        mode (see train() override) to avoid corrupting running statistics.

        The SpeechBrain ECAPA trunk lives in
        `self.classifier.mods.embedding_model.blocks` (ModuleList of
        SE-Res2Blocks).
        """
        self.freeze()  # freeze everything first
        embedding = self.classifier.mods.embedding_model
        blocks = getattr(embedding, "blocks", None)

        if blocks is not None and len(blocks) > 0:
            n_blocks = len(blocks)
            n = max(1, min(n, n_blocks))
            for i in range(n_blocks - n, n_blocks):
                for p in blocks[i].parameters():
                    p.requires_grad = True
            self._frozen = False
            n_train = sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)
            print(f"  🔓 ECAPA-TDNN: unfroze last {n}/{n_blocks} block(s) — "
                  f"{n_train:,} trainable encoder params")
            return

        # Fallback for unusual ECAPA variants: unfreeze the last n top-level
        # trainable children of the embedding model.
        children = [(nm, m) for nm, m in embedding.named_children()
                    if sum(p.numel() for p in m.parameters()) > 0]
        targets = set(nm for nm, _ in children[-n:])
        for nm, m in children:
            if nm in targets:
                for p in m.parameters():
                    p.requires_grad = True
        self._frozen = False
        n_train = sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)
        print(f"  🔓 ECAPA-TDNN: unfroze last {n} module(s) — "
              f"{n_train:,} trainable encoder params")


# ═══════════════════════════════════════════════════════════
#  ModelScope helpers (CAM++ / ERes2NetV2 share this path)
# ═══════════════════════════════════════════════════════════

def _modelscope_embedding_model(model_id: str, local_cache: Optional[str]):
    """
    Build a ModelScope ``SpeechEmbedding`` model, loading purely from the local
    cache at inference time.

    ModelScope resolves models from its cache (``MODELSCOPE_CACHE``). On the
    dev machine we download the snapshot with ``snapshot_download(cache_dir=...)``
    into ``weights/<model>/``; at inference we point ``MODELSCOPE_CACHE`` at the
    same dir so ``Model.from_pretrained`` reads from disk and never fetches.

    Returns:
        The ``SpeechEmbedding`` wrapper (an ``nn.Module``).
    """
    try:
        from modelscope.models.audio.sv import SpeechEmbedding
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "modelscope is not installed. Add `modelscope>=1.38.1` to your "
            f"environment (leaderboard server has it). Original error: {e}"
        ) from e

    if local_cache is not None:
        if not os.path.isdir(local_cache):
            raise FileNotFoundError(
                f"ModelScope cache dir not found: {local_cache}. Run "
                "`python scripts/download_all_weights.py` on the dev machine."
            )
        # Point the cache at the local dir so the pipeline reads from disk.
        os.environ["MODELSCOPE_CACHE"] = local_cache
        os.environ["MODELSCOPE_HOME"] = local_cache

    return SpeechEmbedding(model_id=model_id, model_revision="v1.0.2")


def _modelscope_forward(model, waveforms: torch.Tensor) -> torch.Tensor:
    """
    Run a ModelScope SpeechEmbedding on a batch of raw 16 kHz waveforms.

    ModelScope embeddings are extracted via a numpy-based frontend (fbank),
    so we convert to numpy, run the model, and convert back to torch on the
    original device.

    Returns:
        embeddings: (batch, 1, D) — utterance-level, identity-pooled shape.
    """
    dev = waveforms.device
    wav = waveforms.squeeze(1)  # (batch, T)
    wav_np = wav.detach().float().cpu().numpy()

    result = model.forward(wav_np)

    if isinstance(result, dict):
        # Some model wrappers return dicts (e.g. {"spk_embedding": ...}).
        for key in ("spk_embedding", "embedding", "embeddings"):
            if key in result:
                result = result[key]
                break
        else:
            raise RuntimeError(
                f"ModelScope forward returned dict without a known embedding "
                f"key: {sorted(result.keys())}"
            )

    emb = torch.as_tensor(result, dtype=torch.float32, device=dev)
    if emb.ndim == 2:
        emb = emb.unsqueeze(1)  # (batch, 1, D)
    elif emb.ndim == 3 and emb.size(1) > 1:
        # Frame-level output — mean-pool to one vector per sample.
        emb = emb.mean(dim=1, keepdim=True)
    return emb


class _ModelScopeEncoderBase(BaseEncoder):
    """
    Shared behaviour for ModelScope speaker encoders (CAM++, ERes2NetV2).

    Both models wrap ``SpeechEmbedding`` and produce utterance-level
    embeddings, so they use ``pooling_type: identity`` and stay frozen
    (trainable heads only). The ``train()`` override keeps the encoder in
    eval mode so BatchNorm running stats are never corrupted.
    """

    def __init__(
        self,
        model_id: str,
        local_path: Optional[str] = None,
        allow_hub_download: bool = False,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.model_id = model_id
        self.local_path = local_path
        self._frozen = freeze_encoder

        if local_path is not None:
            self.model = _modelscope_embedding_model(model_id, local_cache=local_path)
        elif allow_hub_download:
            # Dev machine — download into the default ModelScope cache.
            self.model = _modelscope_embedding_model(model_id, local_cache=None)
        else:
            raise RuntimeError(
                f"ModelScope '{model_id}': no local_path and allow_hub_download=False. "
                "At inference the model must load from a local cache dir; on the "
                "dev machine set allow_hub_download=True."
            )

        if freeze_encoder:
            self.freeze()
            print(f"  🔒 {type(self).__name__}: encoder FROZEN")
        else:
            print(f"  🔓 {type(self).__name__}: encoder UNFROZEN")

        # Keep the wrapper in eval mode at all times (BN safety).
        self.model.eval()
        self.eval()

    def to(self, *args, **kwargs):
        """Move both the wrapper and internal module (ModelScope is nn.Module)."""
        super().to(*args, **kwargs)
        if hasattr(self, "model"):
            self.model.to(*args, **kwargs)
        return self

    def train(self, mode: bool = True):
        """Frozen encoder → always eval (protect BatchNorm running stats)."""
        super().train(False)
        return self

    def forward(
        self,
        waveforms: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        emb = _modelscope_forward(self.model, waveforms)
        return emb, None  # (batch, 1, D), lengths=None

    def freeze(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False
        self._frozen = True

    def unfreeze(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = True
        self._frozen = False
        print(f"  🔓 {type(self).__name__}: encoder UNFROZEN.")


# ═══════════════════════════════════════════════════════════
#  CAM++ Encoder (ModelScope) — Phase 2a
# ═══════════════════════════════════════════════════════════

class CAMPlusPlusEncoder(_ModelScopeEncoderBase):
    """
    CAM++ speaker verification model (ModelScope), 512-dim embeddings.

    CAM++ (Context-Aware Masking) is a large-scale speaker embedding model
    from Alibaba's 3D-Speaker family, trained on VoxCeleb 1+2. The ModelScope
    ``SpeechEmbedding`` wrapper returns utterance-level 512-dim vectors, so
    the encoder output is (B, 1, 512) and pooling is ``identity``.

    Source (dev download):
        iic/speech_campplus_sv_en_voxceleb_16k

    Offline: point ``MODELSCOPE_CACHE`` at ``weights/campp/``.
    """

    def __init__(
        self,
        model_id: str = "iic/speech_campplus_sv_en_voxceleb_16k",
        local_path: Optional[str] = None,
        allow_hub_download: bool = False,
        freeze_encoder: bool = True,
    ):
        super().__init__(
            model_id=model_id,
            local_path=local_path,
            allow_hub_download=allow_hub_download,
            freeze_encoder=freeze_encoder,
        )
        self._output_dim = 512

    @property
    def output_dim(self) -> int:
        return self._output_dim


# ═══════════════════════════════════════════════════════════
#  ERes2NetV2 Encoder (ModelScope) — Phase 2c
# ═══════════════════════════════════════════════════════════

class ERes2NetV2Encoder(_ModelScopeEncoderBase):
    """
    ERes2NetV2 speaker verification model (ModelScope), 512-dim embeddings.

    ERes2NetV2 is an enhanced Res2Net-based speaker embedding model from the
    3D-Speaker family. It uses the SAME ModelScope ``SpeechEmbedding`` wrapper
    as CAM++, so it shares the offline cache pattern and returns (B, 1, 512).

    Source (dev download):
        iic/speech_eres2netv2_sv_en_voxceleb_16k

    ⚠️ Dependency note: if the ModelScope pipeline for this model requires the
    standalone ``3dspeaker`` package (NOT installed on the leaderboard server),
    the download/load fails and this encoder must be skipped — verified during
    Phase 2c on the dev machine.
    """

    def __init__(
        self,
        model_id: str = "iic/speech_eres2netv2_sv_en_voxceleb_16k",
        local_path: Optional[str] = None,
        allow_hub_download: bool = False,
        freeze_encoder: bool = True,
    ):
        super().__init__(
            model_id=model_id,
            local_path=local_path,
            allow_hub_download=allow_hub_download,
            freeze_encoder=freeze_encoder,
        )
        self._output_dim = 512

    @property
    def output_dim(self) -> int:
        return self._output_dim


# ═══════════════════════════════════════════════════════════
#  TitaNet-Large Encoder (NeMo) — Phase 2d
# ═══════════════════════════════════════════════════════════

class TitaNetEncoder(BaseEncoder):
    """
    NeMo TitaNet-Large speaker verification model, 192-dim embeddings.

    TitaNet is NVIDIA's speaker embedding model (SEResNet-34 backbone with
    attentive statistical pooling). ``EncDecSpeakerLabelModel.get_embedding``
    returns utterance-level 192-dim vectors → (B, 1, 192), identity pooling.

    Source (dev download):
        nvidia/speakerverification_en_titanet_large

    Offline: ``EncDecSpeakerLabelModel.restore_from("weights/titanet/titanet_large.nemo")``
    — the .nemo file bundles config + weights, so no hub fetch happens.

    NeMo imports are heavy → wrapped in try/except with an informative error.
    """

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_id: str = "nvidia/speakerverification_en_titanet_large",
        allow_hub_download: bool = False,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.local_path = local_path
        self.model_id = model_id
        self._output_dim = 192  # TitaNet-Large embedding dim
        self._frozen = freeze_encoder

        try:
            from nemo.collections.asr.models import EncDecSpeakerLabelModel
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "NeMo is not installed. Add `nemo-toolkit[asr]>=2.7.3` to your "
                f"environment (leaderboard server has it). Original error: {e}"
            ) from e

        if local_path is not None:
            if not os.path.isfile(local_path):
                raise FileNotFoundError(
                    f"TitaNet .nemo file not found: {local_path}. Run "
                    "`python scripts/download_all_weights.py` on the dev machine."
                )
            print(f"  ⬇️  TitaNet-Large: restoring from {local_path}")
            self.titanet = EncDecSpeakerLabelModel.restore_from(
                local_path, map_location="cpu",
            )
        elif allow_hub_download:
            print(f"  ⬇️  TitaNet-Large: downloading from hub {model_id}")
            self.titanet = EncDecSpeakerLabelModel.from_pretrained(
                model_id, map_location="cpu",
            )
        else:
            raise RuntimeError(
                f"TitaNet '{model_id}': no local_path and allow_hub_download=False. "
                "At inference the model must load from a local .nemo file; on the "
                "dev machine set allow_hub_download=True."
            )

        self.titanet.eval()
        self.eval()

        if freeze_encoder:
            self.freeze()
            print(f"  🔒 TitaNet-Large: encoder FROZEN")
        else:
            print(f"  🔓 TitaNet-Large: encoder UNFROZEN")

    def to(self, *args, **kwargs):
        """Move the internal NeMo model (an nn.Module)."""
        super().to(*args, **kwargs)
        if hasattr(self, "titanet"):
            self.titanet.to(*args, **kwargs)
        return self

    def train(self, mode: bool = True):
        """Frozen encoder → always eval (protect BatchNorm running stats)."""
        super().train(False)
        return self

    def forward(
        self,
        waveforms: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Bypasses NeMo's encode_batch-style preprocessing by calling the
        underlying TitaNet model directly with raw 16 kHz waveforms.

        Returns:
            (batch, 1, 192), None
        """
        # NeMo expects (batch, T) raw audio.
        wav = waveforms.squeeze(1).to(self.titanet.device)
        wav_lens = torch.ones(wav.shape[0], device=wav.device)

        if self._frozen:
            with torch.no_grad():
                emb = self.titanet.get_embedding(
                    input_signal=wav, input_signal_length=wav_lens,
                )
        else:
            emb = self.titanet.get_embedding(
                input_signal=wav, input_signal_length=wav_lens,
            )

        if emb.ndim == 2:
            emb = emb.unsqueeze(1)  # (batch, 1, 192)
        return emb, None

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def freeze(self) -> None:
        for param in self.titanet.parameters():
            param.requires_grad = False
        self._frozen = True

    def unfreeze(self) -> None:
        for param in self.titanet.parameters():
            param.requires_grad = True
        self._frozen = False
        print("  🔓 TitaNet-Large encoder UNFROZEN.")


# ═══════════════════════════════════════════════════════════
#  Encoder Registry + Factory
# ═══════════════════════════════════════════════════════════

#: string → encoder class (used by create_encoder and by UI tooling)
ENCODER_REGISTRY = {
    "wavlm": WavLMEncoder,
    "ecapa": ECAPAEncoder,
    "hubert": HuBERTEncoder,          # ⚠️ removed in Phase 3
    "campp": CAMPlusPlusEncoder,
    "eres2net": ERes2NetV2Encoder,
    "titanet": TitaNetEncoder,
}


def create_encoder(config: dict) -> BaseEncoder:
    """
    Build an encoder from config.

    Reads:
        model.encoder_type         → "wavlm" | "ecapa" | "hubert" | "campp"
                                     | "eres2net" | "titanet"
        model.encoder_config.<type> → encoder-specific kwargs
        model.allow_hub_download    → global offline override (default False)

    Every encoder receives the shared offline params:
        local_path           — load weights from this local dir/file (inference)
        allow_hub_download   — permit a hub download (dev machine only)

    Args:
        config: Full project config dict

    Returns:
        BaseEncoder instance
    """
    model_cfg = config["model"]
    encoder_type = model_cfg.get("encoder_type", "wavlm").lower().strip()

    # ── Resolve encoder config (backward-compat with old flat format) ──
    if "encoder_config" in model_cfg:
        enc_cfg = dict(model_cfg["encoder_config"].get(encoder_type, {}))
        # Merge old flat keys as fallback
        if "base_model" not in enc_cfg and "base_model" in model_cfg:
            enc_cfg["base_model"] = model_cfg["base_model"]
        if "freeze_feature_extractor" not in enc_cfg and "freeze_feature_extractor" in model_cfg:
            enc_cfg["freeze_feature_extractor"] = model_cfg["freeze_feature_extractor"]
    else:
        # Old flat config format
        enc_cfg = {
            "base_model": model_cfg.get("base_model", "microsoft/wavlm-large"),
            "freeze_feature_extractor": model_cfg.get("freeze_feature_extractor", True),
        }

    # ── Global offline flag: model.allow_hub_download (default False) ──
    enc_cfg.setdefault("allow_hub_download", model_cfg.get("allow_hub_download", False))

    cls = ENCODER_REGISTRY.get(encoder_type)
    if cls is None:
        raise ValueError(
            f"Unknown encoder_type: '{encoder_type}'. "
            f"Expected one of: {sorted(ENCODER_REGISTRY)}."
        )

    # ── Per-type kwargs (each encoder has a slightly different __init__) ──
    if encoder_type == "wavlm":
        return WavLMEncoder(
            base_model=enc_cfg.get("base_model", "microsoft/wavlm-large"),
            freeze_feature_extractor=enc_cfg.get("freeze_feature_extractor", True),
            local_path=enc_cfg.get("local_path"),
            allow_hub_download=enc_cfg.get("allow_hub_download", False),
        )
    elif encoder_type == "ecapa":
        return ECAPAEncoder(
            source=enc_cfg.get("source", "speechbrain/spkrec-ecapa-voxceleb"),
            freeze_encoder=enc_cfg.get("freeze_encoder", True),
            unfreeze_last_n_blocks=enc_cfg.get("unfreeze_last_n_blocks", 0),
            local_path=enc_cfg.get("local_path"),
            allow_hub_download=enc_cfg.get("allow_hub_download", False),
        )
    elif encoder_type == "hubert":  # ⚠️ removed in Phase 3
        return HuBERTEncoder(
            base_model=enc_cfg.get("base_model", "facebook/hubert-large-ls960-ft"),
            freeze_feature_extractor=enc_cfg.get("freeze_feature_extractor", True),
        )
    elif encoder_type in ("campp", "eres2net"):
        model_id = enc_cfg.get("model_id")
        if not model_id:
            default_id = (
                "iic/speech_campplus_sv_en_voxceleb_16k"
                if encoder_type == "campp"
                else "iic/speech_eres2netv2_sv_en_voxceleb_16k"
            )
            model_id = default_id
        kwargs = dict(
            model_id=model_id,
            local_path=enc_cfg.get("local_path"),
            allow_hub_download=enc_cfg.get("allow_hub_download", False),
            freeze_encoder=enc_cfg.get("freeze_encoder", True),
        )
        if encoder_type == "campp":
            return CAMPlusPlusEncoder(**kwargs)
        return ERes2NetV2Encoder(**kwargs)
    elif encoder_type == "titanet":
        return TitaNetEncoder(
            local_path=enc_cfg.get("local_path"),
            model_id=enc_cfg.get("model_id", "nvidia/speakerverification_en_titanet_large"),
            allow_hub_download=enc_cfg.get("allow_hub_download", False),
            freeze_encoder=enc_cfg.get("freeze_encoder", True),
        )
    raise ValueError(f"Unhandled encoder_type: '{encoder_type}'")


# ═══════════════════════════════════════════════════════════
#  Smoke Test
# ═══════════════════════════════════════════════════════════

def _factory_resolve_smoke():
    """
    Offline test: the factory resolves every registered encoder_type string
    without instantiating (no downloads, no GPU). This is the Phase 1
    acceptance gate for the 4 new encoder keys.
    """
    print("=" * 60)
    print("  Encoder Factory Resolution Smoke Test (offline)")
    print("=" * 60)

    for name in sorted(ENCODER_REGISTRY):
        cls = ENCODER_REGISTRY[name]
        print(f"  ✅ {name:<12} → {cls.__name__}")

    expected = {"ecapa", "wavlm", "hubert", "campp", "eres2net", "titanet"}
    assert set(ENCODER_REGISTRY) == expected, (
        f"Registry mismatch: {sorted(ENCODER_REGISTRY)} vs {sorted(expected)}"
    )
    print("\n  ALL REGISTRY KEYS RESOLVE ✅")


if __name__ == "__main__":
    _factory_resolve_smoke()
