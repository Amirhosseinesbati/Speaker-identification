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
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import WavLMModel


# ═══════════════════════════════════════════════════════════
#  Vendored deps for the leaderboard env
# ═══════════════════════════════════════════════════════════
# ModelScope 1.39+ ships REDUCED wheel metadata and does NOT declare its
# runtime imports (addict, easydict, simplejson, yapf) — and those are NOT in
# the leaderboard's package list either. The submission vendors them under
# <root>/vendor/ so importing modelscope works in the evaluation env.
_VENDOR_DIR = Path(__file__).resolve().parent.parent / "vendor"
if _VENDOR_DIR.is_dir() and str(_VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(_VENDOR_DIR))

# Package root (zip root / submission/ dir). All weight local_paths in the
# checkpoints are RELATIVE ("weights/ecapa", ...) and MUST resolve against this
# dir, not the process CWD — the leaderboard may run submission.py from any
# directory.
_PKG_ROOT = Path(__file__).resolve().parent.parent


def _resolve_local_path(local_path: Optional[str]) -> Optional[str]:
    """Resolve a possibly-relative weight path against the package root."""
    if local_path is None:
        return None
    p = Path(local_path)
    if not p.is_absolute():
        p = _PKG_ROOT / p
    return str(p)


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

class WavLMLayerAdapter(nn.Module):
    """Paper-faithful adapter from one WavLM layer to the downstream head.

    The L-adapter variant in ``adapter-wavlm`` projects each transformer-layer
    output directly into a shared downstream dimension, applies an activation
    and LayerNorm, then combines all adapted layers with a learned softmax.
    There is deliberately no up-projection or residual path in this variant.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: str = "relu",
        use_layer_norm: bool = True,
        init_std: float = 1.0e-3,
        layer_norm_eps: float = 1.0e-5,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("WavLM L-adapter dimensions must be positive")
        if init_std <= 0.0:
            raise ValueError("WavLM L-adapter init_std must be positive")

        activation_name = str(activation).lower().strip()
        if activation_name == "relu":
            activation_module: nn.Module = nn.ReLU()
        elif activation_name == "gelu":
            activation_module = nn.GELU()
        else:
            raise ValueError(
                "WavLM L-adapter activation must be relu or gelu, "
                f"got {activation!r}"
            )

        self.projection = nn.Linear(input_dim, output_dim)
        self.activation = activation_module
        self.layer_norm = (
            nn.LayerNorm(output_dim, eps=layer_norm_eps)
            if use_layer_norm else nn.Identity()
        )
        nn.init.normal_(self.projection.weight, mean=0.0, std=float(init_std))
        nn.init.zeros_(self.projection.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.layer_norm(self.activation(self.projection(hidden_states)))


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
        freeze_encoder: bool = False,
        frozen_backbone_eval: bool = True,
        layer_aggregation: str = "last_hidden",
        layer_adapter_dim: int = 512,
        layer_adapter_activation: str = "relu",
        layer_adapter_layer_norm: bool = True,
        layer_adapter_init_std: float = 1.0e-3,
        layer_adapter_tune_backbone_layer_norms: bool = True,
        local_path: Optional[str] = None,
        allow_hub_download: bool = False,
    ):
        super().__init__()
        self.base_model_name = base_model
        self.local_path = local_path
        self.layer_aggregation = str(layer_aggregation).lower().strip()
        if self.layer_aggregation not in {
            "last_hidden", "weighted_sum", "layer_adapter",
        }:
            raise ValueError(
                "WavLM layer_aggregation must be last_hidden, weighted_sum, "
                "or layer_adapter, "
                f"got {layer_aggregation!r}"
            )
        self.layer_adapter_dim = int(layer_adapter_dim)
        self.layer_adapter_tune_backbone_layer_norms = bool(
            layer_adapter_tune_backbone_layer_norms
        )
        self.frozen_backbone_eval = bool(frozen_backbone_eval)

        if local_path is not None and os.path.isdir(local_path):
            # Offline / submission path — never hit the hub.
            self.wavlm = WavLMModel.from_pretrained(
                local_path, local_files_only=True,
            )
        elif allow_hub_download:
            if local_path is not None:
                # Fresh machine (Vast.ai): local_path configured but missing —
                # fall back to the hub once (dev/training mode only).
                self.wavlm = WavLMModel.from_pretrained(base_model)
            else:
                self.wavlm = WavLMModel.from_pretrained(base_model)
        else:
            raise RuntimeError(
                f"WavLM '{base_model}': local weights missing and "
                "allow_hub_download=False. At inference the model must load "
                "from a local directory; on the dev machine set "
                "allow_hub_download=True."
            )

        # The reference L-adapter recipe disables transformer LayerDrop.  The
        # Hugging Face implementation copies this value onto the encoder at
        # construction time, so keep both locations aligned for an auditable
        # paper-faithful configuration.
        if self.layer_aggregation == "layer_adapter":
            self.wavlm.config.layerdrop = 0.0
            wavlm_encoder = getattr(self.wavlm, "encoder", None)
            if wavlm_encoder is not None and hasattr(wavlm_encoder, "layerdrop"):
                wavlm_encoder.layerdrop = 0.0

        if self.layer_aggregation == "weighted_sum":
            layer_count = int(self.wavlm.config.num_hidden_layers) + 1
            self.layer_weights = nn.Parameter(torch.zeros(layer_count))
            self.layer_adapters = nn.ModuleList()
        elif self.layer_aggregation == "layer_adapter":
            layer_count = int(self.wavlm.config.num_hidden_layers)
            if layer_count <= 0:
                raise ValueError("WavLM must expose at least one transformer layer")
            layer_norm_eps = float(
                getattr(self.wavlm.config, "layer_norm_eps", 1.0e-5)
            )
            self.layer_weights = nn.Parameter(torch.zeros(layer_count))
            self.layer_adapters = nn.ModuleList([
                WavLMLayerAdapter(
                    input_dim=int(self.wavlm.config.hidden_size),
                    output_dim=self.layer_adapter_dim,
                    activation=layer_adapter_activation,
                    use_layer_norm=bool(layer_adapter_layer_norm),
                    init_std=float(layer_adapter_init_std),
                    layer_norm_eps=layer_norm_eps,
                )
                for _ in range(layer_count)
            ])
        else:
            self.register_parameter("layer_weights", None)
            self.layer_adapters = nn.ModuleList()

        if freeze_encoder:
            self.freeze()
        elif freeze_feature_extractor:
            self.unfreeze()
            self.freeze_feature_extractor_only()
        else:
            self.unfreeze()

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
                output_hidden_states=self.layer_aggregation in {
                    "weighted_sum", "layer_adapter",
                },
            )
        if self.layer_aggregation == "weighted_sum":
            hidden_layers = outputs.hidden_states
            if hidden_layers is None or len(hidden_layers) != len(self.layer_weights):
                observed = None if hidden_layers is None else len(hidden_layers)
                raise RuntimeError(
                    "WavLM hidden-state count differs from layer weights: "
                    f"hidden_states={observed}, weights={len(self.layer_weights)}"
                )
            weights = torch.softmax(self.layer_weights, dim=0)
            hidden_states = torch.zeros_like(hidden_layers[0])
            for weight, layer in zip(weights, hidden_layers):
                hidden_states = hidden_states + weight.to(layer.dtype) * layer
        elif self.layer_aggregation == "layer_adapter":
            all_hidden_layers = outputs.hidden_states
            expected_count = len(self.layer_adapters) + 1
            if (all_hidden_layers is None or
                    len(all_hidden_layers) != expected_count):
                observed = (
                    None if all_hidden_layers is None else len(all_hidden_layers)
                )
                raise RuntimeError(
                    "WavLM L-adapter expects the convolutional state plus one "
                    "state per transformer layer: "
                    f"hidden_states={observed}, expected={expected_count}"
                )
            # Hugging Face returns the projected convolutional state first.
            # The reference L-adapter consumes only outputs *after* each
            # transformer layer, hence the intentional [1:] slice.
            transformer_layers = all_hidden_layers[1:]
            adapted_layers = [
                adapter(layer)
                for adapter, layer in zip(
                    self.layer_adapters, transformer_layers,
                )
            ]
            weights = torch.softmax(self.layer_weights, dim=0)
            hidden_states = torch.zeros_like(adapted_layers[0])
            for weight, layer in zip(weights, adapted_layers):
                hidden_states = hidden_states + weight.to(layer.dtype) * layer
        else:
            hidden_states = outputs.last_hidden_state
        return hidden_states, None

    @property
    def output_dim(self) -> int:
        if self.layer_aggregation == "layer_adapter":
            return self.layer_adapter_dim
        return self.wavlm.config.hidden_size

    def freeze(self) -> None:
        """Freeze the complete pretrained WavLM backbone.

        Lightweight downstream layer weights/adapters intentionally remain
        trainable.  In ``layer_adapter`` mode the reference recipe also tunes
        only LayerNorm parameters inside the transformer layers.
        """
        for param in self.wavlm.parameters():
            param.requires_grad = False
        if (self.layer_aggregation == "layer_adapter" and
                self.layer_adapter_tune_backbone_layer_norms):
            for name, param in self.wavlm.named_parameters():
                if (name.startswith("encoder.layers.") and
                        "layer_norm" in name):
                    param.requires_grad = True

    def train(self, mode: bool = True):
        """Keep a completely frozen WavLM backbone deterministic.

        ``model.train()`` recursively switches every child into train mode.
        Without this override, a frozen WavLM still applies dropout and
        LayerDrop, so the lightweight learned aggregation sees a moving feature
        distribution even though no backbone weight can adapt.  Keep only the
        pretrained backbone in eval mode when all of its parameters are frozen;
        downstream layer weights/adapters retain the requested mode.  A
        paper-faithful L-adapter with trainable transformer LayerNorms remains in
        train mode (with LayerDrop disabled above).
        """
        super().train(mode)
        if mode and self.frozen_backbone_eval and not any(
            parameter.requires_grad for parameter in self.wavlm.parameters()
        ):
            self.wavlm.eval()
        return self

    def freeze_feature_extractor_only(self) -> None:
        """Freeze only the convolutional feature extractor (legacy full-FT mode)."""
        if hasattr(self.wavlm, "feature_extractor"):
            for param in self.wavlm.feature_extractor.parameters():
                param.requires_grad = False

    def unfreeze_feature_extractor_only(self) -> None:
        """Unfreeze only the convolutional feature extractor."""
        if hasattr(self.wavlm, "feature_extractor"):
            for param in self.wavlm.feature_extractor.parameters():
                param.requires_grad = True

    def unfreeze(self) -> None:
        """Unfreeze the complete pretrained WavLM backbone."""
        for param in self.wavlm.parameters():
            param.requires_grad = True


# ═══════════════════════════════════════════════════════════
#  ECAPA-TDNN Encoder (SpeechBrain)
# ═══════════════════════════════════════════════════════════

def _patch_ruamel_max_depth():
    """
    hyperpyyaml (<2.0) calls ``yaml.load(stream, Loader=ruamel.yaml.Loader)``,
    which breaks on ruamel.yaml >=0.18 (``'Loader' object has no attribute
    'max_depth'``). The leaderboard pins NO ruamel-yaml, so pip installs the
    latest (0.19.x) and speechbrain's ECAPA config load crashes with exactly
    that AttributeError. Give the Loader a ``max_depth`` attribute (0 = no
    limit) so the composer's depth check is skipped. No-op on ruamel <0.18.
    """
    try:
        import ruamel.yaml
        if not hasattr(ruamel.yaml.Loader, "max_depth"):
            ruamel.yaml.Loader.max_depth = 0
    except Exception:
        pass


def _patch_torchaudio_list_backends():
    """
    speechbrain 1.0.x (which the leaderboard resolves, pin >=1.0.3,<2.0) calls
    ``torchaudio.list_audio_backends()`` unconditionally, but torchaudio 2.9+
    removed that function — AttributeError at import of speechbrain.dataio.
    Provide a stub returning a non-empty backend list so the old check passes.
    No-op on speechbrain 1.1+ (which guards with hasattr) / torchaudio <2.9.
    """
    try:
        import torchaudio
        if not hasattr(torchaudio, "list_audio_backends"):
            torchaudio.list_audio_backends = lambda: ["soundfile", "sox_io"]
    except Exception:
        pass


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


def _is_speechbrain_se_block(module: nn.Module) -> bool:
    """Return True for the pinned SpeechBrain ECAPA squeeze/excitation block."""
    return module.__class__.__name__.lower() == "seblock"


def _enable_se_bn_adapter_parameters(module: nn.Module) -> dict:
    """Freeze ``module`` except ECAPA SE weights and BN affine parameters.

    The SE/BN adapter paper freezes the core speaker encoder while adapting
    channel re-weighting (SE) and activation shift/scale (BN). SpeechBrain's
    ECAPA uses native ``SEBlock`` modules and nested PyTorch BatchNorm layers,
    so the selection can be made without private parameter names.

    Returns an auditable parameter-count receipt. Parameters shared by nested
    modules are de-duplicated by identity.
    """
    for parameter in module.parameters():
        parameter.requires_grad = False

    se_parameters = {}
    bn_parameters = {}
    se_module_names = []
    bn_module_names = []

    for name, child in module.named_modules():
        if _is_speechbrain_se_block(child):
            se_module_names.append(name)
            for parameter in child.parameters(recurse=True):
                parameter.requires_grad = True
                se_parameters[id(parameter)] = parameter
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            bn_module_names.append(name)
            if child.affine:
                for parameter in (child.weight, child.bias):
                    parameter.requires_grad = True
                    bn_parameters[id(parameter)] = parameter

    if not se_module_names or not bn_module_names:
        raise RuntimeError(
            "SE/BN adapter selection found no SEBlock or BatchNorm modules; "
            "the pinned SpeechBrain ECAPA structure may have changed."
        )

    adapter_parameters = dict(se_parameters)
    adapter_parameters.update(bn_parameters)
    total = sum(parameter.numel() for parameter in module.parameters())
    adapter_total = sum(parameter.numel() for parameter in adapter_parameters.values())
    return {
        "mode": "se_bn",
        "total_parameters": total,
        "se_parameters": sum(parameter.numel() for parameter in se_parameters.values()),
        "bn_affine_parameters": sum(
            parameter.numel() for parameter in bn_parameters.values()),
        "adapter_parameters": adapter_total,
        "adapter_fraction": adapter_total / total if total else 0.0,
        "se_module_names": tuple(se_module_names),
        "bn_module_names": tuple(bn_module_names),
    }


def _set_se_bn_adapter_mode(module: nn.Module, training: bool) -> None:
    """Keep the ECAPA core in eval, enabling train mode only for SE and BN.

    BN must see target-domain batch statistics during adaptation; forcing the
    complete SpeechBrain encoder to eval (the legacy partial-FT behaviour)
    would silently turn the proposed BN adapter into affine-only inference.
    Validation/inference calls this with ``training=False`` and therefore uses
    the adapted running statistics without further mutation.
    """
    module.eval()
    if not training:
        return
    for child in module.modules():
        if _is_speechbrain_se_block(child) or isinstance(
                child, nn.modules.batchnorm._BatchNorm):
            child.train(True)


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
        adapter_mode: str = "none",
        local_path: Optional[str] = None,
        allow_hub_download: bool = False,
    ):
        super().__init__()
        self.source = source
        self.local_path = local_path
        self._output_dim = 192  # ECAPA-TDNN embedding dimension
        self._frozen = freeze_encoder
        self._unfreeze_last_n_blocks = max(0, int(unfreeze_last_n_blocks))
        self._adapter_mode = str(adapter_mode or "none").lower().strip()
        self._adapter_receipt = None
        if self._adapter_mode not in {"none", "se_bn"}:
            raise ValueError(
                f"Unsupported ECAPA adapter_mode={adapter_mode!r}; "
                "expected 'none' or 'se_bn'."
            )

        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

        # ruamel.yaml >=0.18 breaks hyperpyyaml's yaml.load(Loader=...)
        # (AttributeError: 'Loader' has no attribute 'max_depth'); the
        # leaderboard doesn't pin ruamel-yaml so the latest (0.19.x) is
        # installed there. Patch BEFORE speechbrain/hyperpyyaml import.
        _patch_ruamel_max_depth()

        # speechbrain 1.0.x calls torchaudio.list_audio_backends() directly,
        # which torchaudio 2.9+ removed — patch before any speechbrain import.
        _patch_torchaudio_list_backends()

        import speechbrain  # noqa: F401  (registers lazy modules)
        # Neutralise speechbrain's broken lazy modules (k2_fsa etc.) before
        # any further import/load — see _patch_speechbrain_lazy_modules docstring.
        _patch_speechbrain_lazy_modules()

        # Version-tolerant imports: the leaderboard pins speechbrain>=1.0.3,<2.0
        # and pip may install a different minor than this dev machine. Both the
        # EncoderClassifier location and LocalStrategy moved across 1.0/1.1+.
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:
            from speechbrain.pretrained import EncoderClassifier  # pre-1.0 path

        try:
            from speechbrain.utils.fetching import LocalStrategy
        except Exception:
            LocalStrategy = None  # absent in some speechbrain 1.0.x

        # ── Source resolution: local savedir (offline) vs hub (dev) ──
        if local_path is not None and os.path.isdir(local_path):
            src = local_path
            savedir = local_path
        elif allow_hub_download:
            src = source
            if local_path is not None:
                # Fresh machine (Vast.ai): local savedir missing — download from
                # the hub INTO it so the offline layout gets populated.
                os.makedirs(local_path, exist_ok=True)
                savedir = local_path
            else:
                savedir = None
        else:
            raise RuntimeError(
                f"ECAPA '{source}': local savedir missing and "
                "allow_hub_download=False. At inference the model must load "
                "from a local directory; on the dev machine set "
                "allow_hub_download=True."
            )

        # Load on CPU first — device will be synced via self.to().
        # local_strategy=COPY is preferred (self-contained savedir, Windows-safe)
        # but not available in every speechbrain version — fall back gracefully.
        _load_kwargs = dict(source=src, savedir=savedir, run_opts={"device": "cpu"})
        if LocalStrategy is not None:
            try:
                self.classifier = EncoderClassifier.from_hparams(
                    **_load_kwargs, local_strategy=LocalStrategy.COPY)
            except TypeError:
                self.classifier = EncoderClassifier.from_hparams(**_load_kwargs)
        else:
            self.classifier = EncoderClassifier.from_hparams(**_load_kwargs)

        if self._adapter_mode == "se_bn":
            self.enable_se_bn_adapter()
        elif freeze_encoder:
            self.freeze()
        elif self._unfreeze_last_n_blocks > 0:
            self.unfreeze_last_n_blocks(self._unfreeze_last_n_blocks)
        else:
            self.unfreeze()

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
        super().train(False)  # wrapper itself stays deterministic
        if hasattr(self, "classifier"):
            self.classifier.mods.eval()
            if self._adapter_mode == "se_bn":
                _set_se_bn_adapter_mode(
                    self.classifier.mods.embedding_model, training=bool(mode))
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
        self._adapter_mode = "none"
        self._adapter_receipt = None

    def unfreeze(self) -> None:
        """Unfreeze all ECAPA-TDNN parameters (full fine-tune)."""
        for param in self.classifier.parameters():
            param.requires_grad = True
        self._frozen = False
        self._adapter_mode = "none"
        self._adapter_receipt = None

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
            self._adapter_mode = "none"
            n_train = sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)
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
        self._adapter_mode = "none"
        n_train = sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)

    def enable_se_bn_adapter(self) -> dict:
        """Enable paper-motivated SE/BN adaptation on the ECAPA embedding net."""
        for parameter in self.classifier.parameters():
            parameter.requires_grad = False
        receipt = _enable_se_bn_adapter_parameters(
            self.classifier.mods.embedding_model)
        self._adapter_mode = "se_bn"
        self._adapter_receipt = receipt
        # A graph is required through the frozen core to reach adapter params.
        self._frozen = False
        _set_se_bn_adapter_mode(
            self.classifier.mods.embedding_model, training=False)
        return dict(receipt)


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

    if local_cache is not None and os.path.isdir(local_cache):
        os.environ["MODELSCOPE_CACHE"] = local_cache
        os.environ["MODELSCOPE_HOME"] = local_cache
        # Resolve the snapshot from disk only — zero network calls (C1).
        # `local_files_only` was added in later modelscope versions; if absent,
        # fall back to plain snapshot_download (cache is populated already).
        try:
            model_dir = snapshot_download(
                model_id, revision=revision,
                cache_dir=local_cache, local_files_only=True,
            )
        except TypeError:
            model_dir = snapshot_download(
                model_id, revision=revision, cache_dir=local_cache,
            )
        model = _ms_from_pretrained(Model, model_dir)
    elif allow_hub_download and local_cache is not None:
        # Fresh machine (Vast.ai): local cache configured but missing — download
        # INTO it (populating weights/<enc>) instead of the default cache.
        os.makedirs(local_cache, exist_ok=True)
        os.environ["MODELSCOPE_CACHE"] = local_cache
        os.environ["MODELSCOPE_HOME"] = local_cache
        model_dir = snapshot_download(
            model_id, revision=revision, cache_dir=local_cache,
        )
        model = _ms_from_pretrained(Model, model_dir)
    elif allow_hub_download:
        # Dev machine — download into the default ModelScope cache.
        model = _ms_from_pretrained(Model, model_id, revision=revision)
    else:
        raise RuntimeError(
            f"ModelScope '{model_id}': local cache missing and "
            "allow_hub_download=False. At inference the model must load from "
            "a local cache dir; on the dev machine set "
            "allow_hub_download=True."
        )

    return model


def _ms_from_pretrained(Model, source, revision: Optional[str] = None, device: str = "cpu"):
    """``Model.from_pretrained`` across modelscope versions (device kwarg moved)."""
    kwargs = {}
    if revision is not None:
        kwargs["revision"] = revision
    try:
        return Model.from_pretrained(source, device=device, **kwargs)
    except TypeError:
        model = Model.from_pretrained(source, **kwargs)
        model.to(device)
        return model


def _modelscope_extract_features(model, wav: torch.Tensor) -> torch.Tensor:
    """
    Per-sample fbank + mean-normalisation for the CAM++ ModelScope model,
    stacked into (B, T_f, F).

    ``torchaudio.compliance.kaldi.fbank`` is NOT batch-safe: with a ``(B, T)``
    input it silently treats dim 0 as channels and averages the batch (a
    correctness trap), so the fbank must stay per-sample. The fbank is cheap
    (~ms) — the expensive part is the embedding net, which is batch-native.

    Mirrors ``SpeakerVerificationCAMPPlus.__extract_feature`` but keeps the
    tensor on device and returns the stacked batch (no detach / cpu).
    """
    from torchaudio.compliance import kaldi as Kaldi

    feat_dim = int(getattr(model, "feature_dim", 80))
    features = []
    for i in range(wav.shape[0]):
        feature = Kaldi.fbank(wav[i : i + 1], num_mel_bins=feat_dim)
        feature = feature - feature.mean(dim=0, keepdim=True)
        features.append(feature.unsqueeze(0))
    return torch.cat(features, dim=0)


def _modelscope_forward(
    model,
    waveforms: torch.Tensor,
    frozen: bool = False,
) -> torch.Tensor:
    """
    Run the CAM++ ModelScope model on a batch of raw 16 kHz waveforms.

    The ModelScope wrapper's ``forward()`` calls its net PER SAMPLE and returns
    ``embedding.detach().cpu()`` — which (a) forces a GPU→CPU→GPU round-trip +
    device sync per sample (the training bottleneck) and (b) silently kills
    gradient flow (so "full fine-tune" never reached the encoder). We therefore
    bypass it:

      1. extract the (per-sample) fbank features — cheap (~ms per batch),
      2. run ``embedding_model`` as ONE batched call on the stacked features —
         the net is batch-native and stays in eval mode (BatchNorm uses running
         stats), so this is numerically identical to the per-sample loop
         (verified by an equivalence test),
      3. keep the result on device, and run under ``no_grad`` when the encoder
         is frozen (mirrors ECAPA/TitaNet) so no autograd graph is built.

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

    def _compute() -> torch.Tensor:
        with torch.autocast(device_type=model_dev.type, enabled=False):
            features = _modelscope_extract_features(model, wav)  # (B, T_f, F)
            embs = model.embedding_model(features)               # (B, D)
        if embs.ndim == 3 and embs.size(1) == 1:
            embs = embs.squeeze(1)
        return embs

    if frozen:
        with torch.no_grad():
            out = _compute()
    else:
        out = _compute()
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
        unfreeze_last_n_blocks: int = 0,
    ):
        super().__init__()
        self.model_id = model_id
        self.local_path = local_path
        self.revision = revision
        self._frozen = freeze_encoder
        self._unfreeze_last_n_blocks = max(0, int(unfreeze_last_n_blocks))
        self._device = torch.device("cpu")

        self.model = _modelscope_load_model(
            model_id, local_cache=local_path,
            allow_hub_download=allow_hub_download,
            revision=revision,
        )

        if freeze_encoder:
            self.freeze()
        elif self._unfreeze_last_n_blocks > 0:
            self.unfreeze_last_n_blocks(self._unfreeze_last_n_blocks)
        else:
            self.unfreeze()

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
        # frozen → no_grad (no autograd graph for the frozen encoder).
        emb = _modelscope_forward(self.model, waveforms, frozen=self._frozen)
        return emb, None  # (batch, 1, D), lengths=None

    def freeze(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False
        self._frozen = True

    def unfreeze(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = True
        self._frozen = False

    def unfreeze_last_n_blocks(self, n: int = 2) -> None:
        """Unfreeze the last ``n`` dense-TDNN blocks of the CAM++ backbone.

        The ModelScope CAMPPlus net stores its trunk in ``embedding_model.xvector``
        (``block1..block3`` interleaved with ``transit`` layers). Only the deepest
        ``n`` blocks become trainable; the FCM head, tdnn stem, transits, stats
        pooling and dense projection stay frozen — the few-shot-safe fine-tune
        mode for CAM++.
        """
        self.freeze()  # freeze everything first
        emb = getattr(self.model, "embedding_model", None)
        xvector = getattr(emb, "xvector", None)
        if xvector is not None:
            xvector_mods = dict(xvector.named_children())
            block_names = [nm for nm in xvector_mods if nm.startswith("block")]
            n = max(1, min(int(n), len(block_names)))
            for nm in block_names[-n:]:
                for p in xvector_mods[nm].parameters():
                    p.requires_grad = True
            self._frozen = False
            return
        # Fallback for unusual CAMPPlus variants: unfreeze the last n top-level
        # trainable children of the embedding model.
        children = [(nm, m) for nm, m in emb.named_children()
                    if sum(p.numel() for p in m.parameters()) > 0]
        targets = set(nm for nm, _ in children[-n:])
        for nm, m in children:
            if nm in targets:
                for p in m.parameters():
                    p.requires_grad = True
        self._frozen = False


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
        unfreeze_last_n_blocks: int = 0,
    ):
        super().__init__(
            model_id=model_id,
            local_path=local_path,
            allow_hub_download=allow_hub_download,
            freeze_encoder=freeze_encoder,
            revision=revision,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
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
        unfreeze_last_n_blocks: int = 0,
    ):
        super().__init__()
        self.local_path = local_path
        self._output_dim = 192  # verified from official checkpoint (seg_1)
        self._frozen = freeze_encoder
        self._unfreeze_last_n_blocks = max(0, int(unfreeze_last_n_blocks))

        from src.sv_arch import ERes2NetV2

        self.model = ERes2NetV2(
            embed_dim=self._output_dim, baseWidth=26, scale=2, expansion=2,
        )

        if local_path is not None and os.path.isfile(
            os.path.join(local_path, ckpt_name)
        ):
            ckpt_path = os.path.join(local_path, ckpt_name)
            state = torch.load(ckpt_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            self.model.load_state_dict(state, strict=True)
        elif allow_hub_download:
            # Dev machine — fetch the official release from HF (reliable CDN).
            from huggingface_hub import hf_hub_download
            if local_path is not None:
                # Fresh machine (Vast.ai): local checkpoint missing — download
                # INTO the configured dir so the offline layout is populated.
                os.makedirs(local_path, exist_ok=True)
                ckpt_path = os.path.join(local_path, ckpt_name)
                downloaded = hf_hub_download(
                    "bandad/eres2netv2_pretrained",
                    "pretrained_eres2netv2.ckpt",
                )
                import shutil
                shutil.copyfile(downloaded, ckpt_path)
            else:
                ckpt_path = hf_hub_download(
                    "bandad/eres2netv2_pretrained",
                    "pretrained_eres2netv2.ckpt",
                )
            state = torch.load(ckpt_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            self.model.load_state_dict(state, strict=True)
        else:
            raise RuntimeError(
                "ERes2NetV2: local checkpoint missing and "
                "allow_hub_download=False. At inference the model must load "
                "from a local checkpoint; on the dev machine set "
                "allow_hub_download=True."
            )

        self.model.eval()
        self.eval()

        if freeze_encoder:
            self.freeze()
        elif self._unfreeze_last_n_blocks > 0:
            self.unfreeze_last_n_blocks(self._unfreeze_last_n_blocks)
        else:
            self.unfreeze()

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
        Batched net forward.

        The fbank frontend is per-sample (``Kaldi.fbank`` is not batch-safe),
        but the ERes2NetV2 net itself is batch-native, so all per-sample fbank
        features are stacked and ONE batched net call is made. The encoder is
        always in eval mode (BatchNorm uses running statistics), so this is
        numerically identical to the old per-sample loop — verified by an
        equivalence test. Runs under ``no_grad`` when frozen (mirrors
        ECAPA/TitaNet) so no autograd graph is built for the frozen encoder.

        Returns:
            (batch, 1, 192), None
        """
        dev = waveforms.device
        wav = waveforms.squeeze(1).float()  # (batch, T)

        def _compute() -> torch.Tensor:
            feats = []
            # fp32 compute (like the ModelScope wrappers): BatchNorm/Conv
            # mixing breaks under an outer autocast (Float vs Half weights).
            with torch.autocast(device_type=dev.type, enabled=False):
                for i in range(wav.shape[0]):
                    feature = _eres2net_fbank(wav[i : i + 1]).to(dev)  # (1, T_f, 80)
                    feats.append(feature)
                features = torch.cat(feats, dim=0)  # (B, T_f, 80)
                return self.model(features)         # (B, 192)

        if self._frozen:
            with torch.no_grad():
                out = _compute()
        else:
            out = _compute()
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

    def unfreeze_last_n_blocks(self, n: int = 2) -> None:
        """Unfreeze the last ``n`` conv stages (``layer1..layer4``) of ERes2NetV2.

        The trunk of the vendored ERes2NetV2 is four ``nn.Sequential`` stages
        (``layer1..layer4``). Only the deepest ``n`` become trainable; the stem
        (``conv1``/``bn1``), downsampling, AFF fusion, pooling and embedding
        projection stay frozen — the few-shot-safe fine-tune mode for ERes2NetV2.
        """
        self.freeze()
        children = dict(self.model.named_children())
        layer_names = [nm for nm in children if nm.startswith("layer")]
        n = max(1, min(int(n), len(layer_names)))
        targets = set(layer_names[-n:])
        for nm, m in children.items():
            if nm in targets:
                for p in m.parameters():
                    p.requires_grad = True
        self._frozen = False


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

        if local_path is not None and os.path.isfile(local_path):
            try:
                self.titanet = EncDecSpeakerLabelModel.restore_from(
                    local_path, map_location="cpu",
                )
            except TypeError:  # restore_from signature varies across NeMo 2.x
                self.titanet = EncDecSpeakerLabelModel.restore_from(local_path)
        elif allow_hub_download:
            if local_path is not None:
                # Fresh machine (Vast.ai): local .nemo missing — download from
                # the hub and persist a copy into the configured path.
                model = EncDecSpeakerLabelModel.from_pretrained(
                    model_id, map_location="cpu",
                )
                os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
                model.save_to(str(local_path))
                self.titanet = model
            else:
                self.titanet = EncDecSpeakerLabelModel.from_pretrained(
                    model_id, map_location="cpu",
                )
        else:
            raise RuntimeError(
                f"TitaNet '{model_id}': local .nemo missing and "
                "allow_hub_download=False. At inference the model must load "
                "from a local .nemo file; on the dev machine set "
                "allow_hub_download=True."
            )

        self.titanet.eval()
        self.eval()

        if freeze_encoder:
            self.freeze()
        else:
            self.unfreeze()

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

    # Weights ship inside the package (weights/<enc>/...). The checkpoints
    # record RELATIVE local_paths, so resolve them against the package root —
    # the leaderboard may run submission.py from any CWD.
    enc_cfg["local_path"] = _resolve_local_path(enc_cfg.get("local_path"))

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
            freeze_encoder=enc_cfg.get("freeze_encoder", False),
            frozen_backbone_eval=enc_cfg.get("frozen_backbone_eval", True),
            layer_aggregation=enc_cfg.get("layer_aggregation", "last_hidden"),
            layer_adapter_dim=enc_cfg.get("layer_adapter_dim", 512),
            layer_adapter_activation=enc_cfg.get(
                "layer_adapter_activation", "relu",
            ),
            layer_adapter_layer_norm=enc_cfg.get(
                "layer_adapter_layer_norm", True,
            ),
            layer_adapter_init_std=enc_cfg.get(
                "layer_adapter_init_std", 1.0e-3,
            ),
            layer_adapter_tune_backbone_layer_norms=enc_cfg.get(
                "layer_adapter_tune_backbone_layer_norms", True,
            ),
            local_path=enc_cfg.get("local_path"),
            allow_hub_download=enc_cfg.get("allow_hub_download", False),
        )
    elif encoder_type == "ecapa":
        return ECAPAEncoder(
            source=enc_cfg.get("source", "speechbrain/spkrec-ecapa-voxceleb"),
            freeze_encoder=enc_cfg.get("freeze_encoder", True),
            unfreeze_last_n_blocks=enc_cfg.get("unfreeze_last_n_blocks", 0),
            adapter_mode=enc_cfg.get("adapter_mode", "none"),
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
            unfreeze_last_n_blocks=enc_cfg.get("unfreeze_last_n_blocks", 0),
        )
    elif encoder_type == "eres2net":
        return ERes2NetV2Encoder(
            local_path=enc_cfg.get("local_path"),
            allow_hub_download=enc_cfg.get("allow_hub_download", False),
            freeze_encoder=enc_cfg.get("freeze_encoder", True),
            ckpt_name=enc_cfg.get("ckpt_name", "eres2netv2.ckpt"),
            unfreeze_last_n_blocks=enc_cfg.get("unfreeze_last_n_blocks", 0),
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
