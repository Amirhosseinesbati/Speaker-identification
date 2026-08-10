"""
Speaker Encoder Backbones.

Each encoder accepts raw audio waveforms and produces frame-level hidden states.
All encoders share a common interface for seamless swapping.

Classes:
    BaseEncoder           — Abstract interface
    WavLMEncoder          — Microsoft WavLM (HuggingFace), base or large
    ECAPAEncoder          — ECAPA-TDNN (SpeechBrain)
    CAMPlusPlusEncoder    — CAM++ (ModelScope) — Phase 2a
    ERes2NetV2Encoder     — ERes2NetV2 (vendored arch + official ckpt) — Phase 2c
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

    WavLM is a self-supervised speech model with gated relative position bias
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

        # WavLM-Large is numerically unstable under fp16 autocast on some GPUs
        # (attention logits overflow → NaN in softmax — verified on GTX 1660 Ti
        # with both noise and real speech). Run the transformer in fp32; the
        # heads still compute under the outer autocast.
        dev = input_values.device
        with torch.autocast(device_type=dev.type, enabled=False):
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
        from speechbrain.utils.fetching import LocalStrategy

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

        # Load on CPU first — device will be synced via self.to().
        # local_strategy=COPY: self-contained savedir (no symlinks), which is
        # required on Windows and makes the weights dir zip-portable.
        self.classifier = EncoderClassifier.from_hparams(
            source=src,
            savedir=src if local_path is not None else None,
            run_opts={"device": "cpu"},
            local_strategy=LocalStrategy.COPY,
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

def _modelscope_load_model(
    model_id: str,
    local_cache: Optional[str],
    allow_hub_download: bool,
    revision: str = "master",
):
    """
    Load a ModelScope speaker model (CAM++ / ERes2NetV2), offline-first.

    ModelScope resolves models via its cache. On the dev machine the snapshot
    is downloaded into ``weights/<model>/`` (with ``cache_dir=...``); at
    inference we point ``MODELSCOPE_CACHE`` at the same dir and resolve the
    snapshot with ``local_files_only=True`` so nothing touches the network.

    The returned object is the registered ``TorchModel`` subclass for the
    model id (e.g. ``SpeakerVerificationCAMPPlus``), whose ``forward`` takes
    raw ``[N, T]`` waveforms and returns ``[N, D]`` embeddings.

    Returns:
        The modelscope model (an ``nn.Module``).
    """
    try:
        from modelscope.models import Model
        from modelscope.hub.snapshot_download import snapshot_download
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
        os.environ["MODELSCOPE_CACHE"] = local_cache
        os.environ["MODELSCOPE_HOME"] = local_cache
        # Resolve the snapshot from disk only — zero network calls (C1).
        model_dir = snapshot_download(
            model_id, revision=revision,
            cache_dir=local_cache, local_files_only=True,
        )
        model = Model.from_pretrained(model_dir, device="cpu")
    elif allow_hub_download:
        # Dev machine — download into the default ModelScope cache.
        model = Model.from_pretrained(model_id, revision=revision, device="cpu")
    else:
        raise RuntimeError(
            f"ModelScope '{model_id}': no local_path and allow_hub_download=False. "
            "At inference the model must load from a local cache dir; on the "
            "dev machine set allow_hub_download=True."
        )

    return model


def _modelscope_forward(model, waveforms: torch.Tensor) -> torch.Tensor:
    """
    Run a ModelScope speaker model on a batch of raw 16 kHz waveforms.

    Both CAM++ and ERes2NetV2 are called PER SAMPLE:
      - CAM++'s ``__extract_feature`` already loops per row internally.
      - ERes2NetV2's ``__extract_feature`` mean-normalises over dim 0, which
        is only correct when the batch dim is 1 → must be called per sample
        (the official ModelScope pipeline does exactly this).
      - ``torchaudio.compliance.kaldi.fbank`` drops a batch dim of 1, so a
        ``(1, T)`` input yields the per-sample fbank the nets expect.

    Device handling: the wrapper stores ``model.device`` at load time, but the
    outer ``nn.Module.to()`` moves parameters via ``_apply`` WITHOUT calling
    our ``to()`` override — so we re-sync ``model.device`` from the actual
    parameter device here and run the fbank + net on that device.

    The whole call runs OUTSIDE autocast (``enabled=False``): the torchaudio
    fbank frontend is float32-only and the ModelScope nets mix non-autocast
    ops (BatchNorm) — fp32 compute on these small models costs nothing and
    avoids FloatTensor-vs-HalfTensor mismatches under an outer autocast.

    Returns:
        embeddings: (batch, 1, D) — identity-pooled shape.
    """
    params = list(model.parameters())
    model_dev = params[0].device if params else torch.device("cpu")
    try:
        model.device = model_dev  # sync wrapper's stored device attr
    except Exception:
        pass

    wav = waveforms.squeeze(1).float().to(model_dev)  # (batch, T)

    embs = []
    with torch.autocast(device_type=model_dev.type, enabled=False):
        for i in range(wav.shape[0]):
            emb = model.forward(wav[i : i + 1])  # (1, D)
            embs.append(emb)
    out = torch.cat(embs, dim=0)  # (batch, D)

    if out.ndim == 3 and out.size(1) == 1:
        out = out.squeeze(1)
    return out.to(waveforms.device).unsqueeze(1)  # (batch, 1, D)


class _ModelScopeEncoderBase(BaseEncoder):
    """
    Shared behaviour for ModelScope speaker encoders (CAM++, ERes2NetV2).

    Both models are loaded via ``Model.from_pretrained`` (registered
    ``TorchModel`` subclasses) and produce utterance-level embeddings, so they
    use ``pooling_type: identity`` and stay frozen (trainable heads only).

    Device handling: the ModelScope wrapper holds its own ``self.device`` and
    an ``embedding_model`` nn.Module. ``to()`` syncs both so forward runs on
    the outer model's device; the fbank frontend runs on CPU (torchaudio
    compliance kaldi is CPU-only) — safe and correct, the nets are small.
    """

    def __init__(
        self,
        model_id: str,
        local_path: Optional[str] = None,
        allow_hub_download: bool = False,
        freeze_encoder: bool = True,
        revision: str = "master",
    ):
        super().__init__()
        self.model_id = model_id
        self.local_path = local_path
        self.revision = revision
        self._frozen = freeze_encoder
        self._device = torch.device("cpu")

        self.model = _modelscope_load_model(
            model_id, local_cache=local_path,
            allow_hub_download=allow_hub_download,
            revision=revision,
        )
        print(f"  ⬇️  {type(self).__name__}: loaded {type(self.model).__name__}")

        if freeze_encoder:
            self.freeze()
            print(f"  🔒 {type(self).__name__}: encoder FROZEN")
        else:
            print(f"  🔓 {type(self).__name__}: encoder UNFROZEN")

        # Keep the wrapper in eval mode at all times (BN safety).
        self.model.eval()
        self.eval()

    def to(self, *args, **kwargs):
        """Move both the wrapper and its internal embedding model + device."""
        super().to(*args, **kwargs)
        target = None
        for a in args:
            if isinstance(a, torch.device):
                target = a
                break
            if isinstance(a, (str, int)):
                target = torch.device(a)
                break
        if target is None and "device" in kwargs:
            target = torch.device(kwargs["device"])
        if target is not None and hasattr(self, "model"):
            emb_model = getattr(self.model, "embedding_model", None)
            if emb_model is not None:
                emb_model.to(target)
            try:
                self.model.device = target
            except Exception:
                pass
            self._device = target
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
        revision: str = "v1.0.2",
    ):
        super().__init__(
            model_id=model_id,
            local_path=local_path,
            allow_hub_download=allow_hub_download,
            freeze_encoder=freeze_encoder,
            revision=revision,
        )
        self._output_dim = 512

    @property
    def output_dim(self) -> int:
        return self._output_dim


# ═══════════════════════════════════════════════════════════
#  ERes2NetV2 Encoder — Phase 2c
# ═══════════════════════════════════════════════════════════

def _eres2net_fbank(audio: torch.Tensor) -> torch.Tensor:
    """
    Per-sample fbank frontend matching the official 3D-Speaker pipeline.

    ``torchaudio.compliance.kaldi.fbank`` drops the batch dim of 1, so a
    ``(1, T)`` input yields ``(T_f, 80)``; we mean-normalise over time and add
    the batch dim back, giving the ``(1, T_f, 80)`` the ERes2NetV2 net expects.
    """
    from torchaudio.compliance import kaldi as Kaldi
    feature = Kaldi.fbank(audio, num_mel_bins=80)          # (T_f, 80)
    feature = feature - feature.mean(dim=0, keepdim=True)  # per-mel time mean
    return feature.unsqueeze(0)                            # (1, T_f, 80)


class ERes2NetV2Encoder(BaseEncoder):
    """
    ERes2NetV2 speaker embedding model, 192-dim embeddings (official release).

    ERes2NetV2 (BDFF + BLFF) is Alibaba 3D-Speaker's enhanced Res2Net model.
    The architecture is VENDORED in ``src/sv_arch.py`` (needs only torch +
    torchaudio), and the official pretrained checkpoint loads with a plain
    ``torch.load`` + ``load_state_dict`` — NO modelscope / 3dspeaker needed.

    Verified on dev (Phase 2c): the official VoxCeleb checkpoint is 192-dim
    (not 512) with 17.86 M params; strict state_dict load succeeds.

    Offline weights: ``weights/eres2net/eres2netv2.ckpt``.
    """

    def __init__(
        self,
        local_path: Optional[str] = None,
        allow_hub_download: bool = False,
        freeze_encoder: bool = True,
        ckpt_name: str = "eres2netv2.ckpt",
    ):
        super().__init__()
        self.local_path = local_path
        self._output_dim = 192  # verified from official checkpoint (seg_1)
        self._frozen = freeze_encoder

        from src.sv_arch import ERes2NetV2

        self.model = ERes2NetV2(
            embed_dim=self._output_dim, baseWidth=26, scale=2, expansion=2,
        )

        if local_path is not None:
            ckpt_path = os.path.join(local_path, ckpt_name)
            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError(
                    f"ERes2NetV2 checkpoint not found: {ckpt_path}. Run "
                    "`python scripts/download_all_weights.py` on the dev machine."
                )
            print(f"  ⬇️  ERes2NetV2: loading checkpoint from {ckpt_path}")
            state = torch.load(ckpt_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            self.model.load_state_dict(state, strict=True)
        elif allow_hub_download:
            # Dev machine — fetch the official release from HF (reliable CDN).
            from huggingface_hub import hf_hub_download
            ckpt_path = hf_hub_download(
                "bandad/eres2netv2_pretrained",
                "pretrained_eres2netv2.ckpt",
            )
            print(f"  ⬇️  ERes2NetV2: downloading from HF hub → {ckpt_path}")
            state = torch.load(ckpt_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            self.model.load_state_dict(state, strict=True)
        else:
            raise RuntimeError(
                "ERes2NetV2: no local_path and allow_hub_download=False. "
                "At inference the model must load from a local checkpoint; on "
                "the dev machine set allow_hub_download=True."
            )

        self.model.eval()
        self.eval()

        if freeze_encoder:
            self.freeze()
            print(f"  🔒 ERes2NetV2: encoder FROZEN")
        else:
            print(f"  🔓 ERes2NetV2: encoder UNFROZEN")

    def to(self, *args, **kwargs):
        """Move the internal ERes2NetV2 nn.Module."""
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
        """
        Per-sample forward (the official wrapper does the same — its
        mean-normalisation over dim 0 is only correct for a batch of 1).

        Returns:
            (batch, 1, 192), None
        """
        dev = waveforms.device
        wav = waveforms.squeeze(1).float()  # (batch, T)

        embs = []
        # fp32 compute (like the ModelScope wrappers): BatchNorm/Conv mixing
        # breaks under an outer autocast (Float vs Half weights).
        with torch.autocast(device_type=dev.type, enabled=False):
            for i in range(wav.shape[0]):
                feature = _eres2net_fbank(wav[i : i + 1])  # (1, T_f, 80)
                feature = feature.to(dev)
                emb = self.model(feature)  # (1, 192)
                embs.append(emb)
        out = torch.cat(embs, dim=0)  # (batch, 192)
        return out.unsqueeze(1), None  # (batch, 1, 192)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def freeze(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False
        self._frozen = True

    def unfreeze(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = True
        self._frozen = False
        print("  🔓 ERes2NetV2 encoder UNFROZEN.")


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
        Extract TitaNet embeddings directly from raw 16 kHz waveforms.

        NeMo 2.7+ dropped the tensor-based ``get_embedding(input_signal, ...)``
        (it now takes a file path), so we run the model's internal pipeline
        manually — the same ops ``forward`` performs:
            preprocessor (mel) → encoder (TitaNet) → decoder (SpeakerDecoder)
        whose second output is the ``(B, 192)`` embedding.

        Returns:
            (batch, 1, 192), None
        """
        dev = next(self.titanet.parameters()).device
        wav = waveforms.squeeze(1).to(dev)  # (batch, T) raw audio
        # NeMo preprocessor expects `length` in SAMPLES per utterance, not ones.
        wav_lens = torch.full(
            (wav.shape[0],), wav.shape[1], dtype=torch.long, device=wav.device,
        )

        if self._frozen:
            with torch.no_grad():
                emb = self._embed_from_waveform(wav, wav_lens)
        else:
            emb = self._embed_from_waveform(wav, wav_lens)

        if emb.ndim == 2:
            emb = emb.unsqueeze(1)  # (batch, 1, 192)
        return emb, None

    def _embed_from_waveform(
        self, wav: torch.Tensor, wav_lens: torch.Tensor
    ) -> torch.Tensor:
        """Run preprocessor → encoder → decoder and return the embedding.

        Computed in fp32 (autocast disabled): NeMo's mel preprocessor is
        float32-only and TitaNet mixes BatchNorm ops that mismatch Half weights.
        """
        dev = wav.device
        with torch.autocast(device_type=dev.type, enabled=False):
            processed, plen = self.titanet.preprocessor(
                input_signal=wav, length=wav_lens,
            )
            encoded, elen = self.titanet.encoder(
                audio_signal=processed, length=plen,
            )
            if elen is None:
                elen = plen
            logits, embs = self.titanet.decoder(
                encoder_output=encoded, length=elen,
            )
        if embs is None:
            # Some decoder configs return only logits — fall back to logits
            # only if it has the embedding dim (192); otherwise error loudly.
            if logits.ndim == 2 and logits.size(1) == self._output_dim:
                embs = logits
            else:
                raise RuntimeError(
                    "TitaNet decoder returned no embedding (embs=None) and "
                    f"logits shape {tuple(logits.shape)} doesn't match "
                    f"embedding dim {self._output_dim}."
                )
        return embs  # (batch, 192)

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
    "campp": CAMPlusPlusEncoder,
    "eres2net": ERes2NetV2Encoder,
    "titanet": TitaNetEncoder,
}


def create_encoder(config: dict) -> BaseEncoder:
    """
    Build an encoder from config.

    Reads:
        model.encoder_type         → "wavlm" | "ecapa" | "campp"
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
    elif encoder_type == "campp":
        model_id = enc_cfg.get("model_id", "iic/speech_campplus_sv_en_voxceleb_16k")
        return CAMPlusPlusEncoder(
            model_id=model_id,
            local_path=enc_cfg.get("local_path"),
            allow_hub_download=enc_cfg.get("allow_hub_download", False),
            freeze_encoder=enc_cfg.get("freeze_encoder", True),
            revision=enc_cfg.get("revision", "v1.0.2"),
        )
    elif encoder_type == "eres2net":
        return ERes2NetV2Encoder(
            local_path=enc_cfg.get("local_path"),
            allow_hub_download=enc_cfg.get("allow_hub_download", False),
            freeze_encoder=enc_cfg.get("freeze_encoder", True),
            ckpt_name=enc_cfg.get("ckpt_name", "eres2netv2.ckpt"),
        )
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

    expected = {"ecapa", "wavlm", "campp", "eres2net", "titanet"}
    assert set(ENCODER_REGISTRY) == expected, (
        f"Registry mismatch: {sorted(ENCODER_REGISTRY)} vs {sorted(expected)}"
    )
    print("\n  ALL REGISTRY KEYS RESOLVE ✅")


if __name__ == "__main__":
    _factory_resolve_smoke()
