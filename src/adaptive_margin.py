"""Deterministic duration-adaptive margin utilities for ALMFT.

The implementation follows the duration-based ALMFT mechanism from Zhang,
Chen & Qian (ICASSP 2023): training chunks are sampled across a duration
interval and the ArcFace/AAM margin grows linearly with the amount of speech
available to the sample.  This module deliberately contains no model or
dataset policy; it only validates the preregistered mapping and supplies
deterministic sampling/cropping primitives used by the training loop.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping, Optional

import torch


def build_duration_adaptive_margin(
    experiment_config: Mapping[str, object],
) -> Optional["DurationAdaptiveMargin"]:
    """Build and integrity-check the D-ALMFT contract from a full config.

    These checks intentionally reject the shortcuts present in the historical
    margin-only run: D-ALMFT must be a warm-started second stage, disable the
    augmentation pipeline, use a known-only ArcFace head, and leave the full
    encoder trainable.  A run that does not satisfy those invariants must use a
    different experiment family/name instead of silently claiming ALMFT.
    """

    training = dict(experiment_config.get("training", {}) or {})
    adaptive = dict(training.get("adaptive_margin", {}) or {})
    audio = dict(experiment_config.get("audio", {}) or {})
    contract = DurationAdaptiveMargin.from_config(
        adaptive,
        sample_rate=int(audio.get("sample_rate", 16_000)),
        max_input_duration_seconds=float(audio.get("duration_seconds", 8.0)),
    )
    if contract is None:
        return None

    if not training.get("warm_start_checkpoint"):
        raise ValueError("D-ALMFT is a secondary stage and requires warm_start_checkpoint.")
    if not bool(adaptive.get("disable_augmentation", False)):
        raise ValueError(
            "D-ALMFT requires adaptive_margin.disable_augmentation=true; "
            "the cited protocol disables augmentation in the second stage."
        )

    model = dict(experiment_config.get("model", {}) or {})
    if str(model.get("speaker_head_type", "arcface")).lower().strip() != "arcface":
        raise ValueError("D-ALMFT currently requires speaker_head_type=arcface.")
    if str(model.get("speaker_target_scope", "metric")).lower().strip() != "known":
        raise ValueError("D-ALMFT requires speaker_target_scope=known.")
    encoder_type = str(model.get("encoder_type", "")).lower().strip()
    encoder_cfg = dict(
        (model.get("encoder_config", {}) or {}).get(encoder_type, {}) or {}
    )
    if bool(encoder_cfg.get("freeze_encoder", True)):
        raise ValueError("D-ALMFT requires the encoder to be trainable.")
    if int(encoder_cfg.get("unfreeze_last_n_blocks", 0) or 0) != 0:
        raise ValueError(
            "D-ALMFT requires full-encoder fine-tuning "
            "(unfreeze_last_n_blocks=0)."
        )
    proto = dict((training.get("loss", {}) or {}).get("proto", {}) or {})
    if bool(proto.get("enabled", False)):
        raise ValueError("D-ALMFT must not be confounded with prototype loss.")
    return contract


@dataclass(frozen=True)
class DurationAdaptiveMargin:
    """Validated duration-to-margin contract for one ALMFT run."""

    min_duration_seconds: float
    max_duration_seconds: float
    margin_anchor_min_duration_seconds: float
    margin_anchor_max_duration_seconds: float
    min_margin: float
    max_margin: float
    sample_rate: int = 16_000
    seed_offset: int = 73_921

    @classmethod
    def from_config(
        cls,
        config: Optional[Mapping[str, object]],
        *,
        sample_rate: int,
        max_input_duration_seconds: float,
    ) -> Optional["DurationAdaptiveMargin"]:
        """Return a validated contract, or ``None`` when ALMFT is disabled."""

        cfg = dict(config or {})
        if not bool(cfg.get("enabled", False)):
            return None
        strategy = str(cfg.get("strategy", "duration_linear")).lower().strip()
        if strategy != "duration_linear":
            raise ValueError(
                "Only adaptive_margin.strategy='duration_linear' is supported; "
                f"got {strategy!r}."
            )

        min_duration = float(cfg.get("min_duration_seconds", 1.0))
        max_duration = float(cfg.get("max_duration_seconds", 6.0))
        anchor_min_duration = float(
            cfg.get("margin_anchor_min_duration_seconds", 2.0)
        )
        anchor_max_duration = float(
            cfg.get("margin_anchor_max_duration_seconds", 6.0)
        )
        min_margin = float(cfg.get("min_margin", 0.2))
        max_margin = float(cfg.get("max_margin", 0.5))
        if not (0.0 < min_duration < max_duration):
            raise ValueError(
                "adaptive_margin requires 0 < min_duration_seconds < "
                "max_duration_seconds."
            )
        if max_duration > float(max_input_duration_seconds) + 1e-9:
            raise ValueError(
                "adaptive_margin.max_duration_seconds cannot exceed the "
                f"dataset window ({max_input_duration_seconds}s)."
            )
        if not (
            min_duration
            <= anchor_min_duration
            < anchor_max_duration
            <= max_duration
        ):
            raise ValueError(
                "adaptive_margin requires sample_min_duration <= "
                "margin_anchor_min_duration < margin_anchor_max_duration <= "
                "sample_max_duration."
            )
        if not (0.0 <= min_margin < max_margin < 1.5707963267948966):
            raise ValueError(
                "adaptive_margin requires 0 <= min_margin < max_margin < pi/2."
            )
        if int(sample_rate) <= 0:
            raise ValueError("sample_rate must be positive.")

        return cls(
            min_duration_seconds=min_duration,
            max_duration_seconds=max_duration,
            margin_anchor_min_duration_seconds=anchor_min_duration,
            margin_anchor_max_duration_seconds=anchor_max_duration,
            min_margin=min_margin,
            max_margin=max_margin,
            sample_rate=int(sample_rate),
            seed_offset=int(cfg.get("seed_offset", 73_921)),
        )

    def margin_for_duration(self, duration_seconds: torch.Tensor) -> torch.Tensor:
        """Map effective duration through the paper's 2s/6s margin anchors.

        Training durations are sampled from 1s to 6s, but Zhang et al. fit the
        duration-margin line from the established 2s -> 0.2 and 6s -> 0.5
        operating points.  Values below/above those anchors are clamped, so the
        advertised margin range remains exactly [0.2, 0.5].
        """

        duration = duration_seconds.to(dtype=torch.float32)
        duration = duration.clamp(
            min=self.margin_anchor_min_duration_seconds,
            max=self.margin_anchor_max_duration_seconds,
        )
        alpha = (
            (duration - self.margin_anchor_min_duration_seconds)
            / (
                self.margin_anchor_max_duration_seconds
                - self.margin_anchor_min_duration_seconds
            )
        )
        return self.min_margin + alpha * (self.max_margin - self.min_margin)

    def sample_duration_seconds(
        self,
        *,
        training_seed: int,
        epoch: int,
        step: int,
        window_index: int,
    ) -> float:
        """Sample one reproducible batch-global chunk duration.

        A local CPU generator avoids consuming the model/dropout RNG stream, so
        enabling ALMFT changes only the preregistered duration/margin mechanism.
        """

        payload = (
            f"{int(training_seed)}:{int(epoch)}:{int(step)}:"
            f"{int(window_index)}:{self.seed_offset}"
        ).encode("ascii")
        mixed_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(mixed_seed)
        unit = float(torch.rand((), generator=generator).item())
        return self.min_duration_seconds + unit * (
            self.max_duration_seconds - self.min_duration_seconds
        )

    def crop_batch(
        self,
        waveforms: torch.Tensor,
        duration_seconds: float,
        *,
        source_durations_seconds: Optional[torch.Tensor] = None,
        training_seed: int,
        epoch: int,
        step: int,
        window_index: int,
    ) -> torch.Tensor:
        """Take deterministic independent sub-crops from a ``(B,1,T)`` batch."""

        if waveforms.ndim != 3:
            raise ValueError(
                f"Expected waveforms shaped (B,1,T), got {tuple(waveforms.shape)}."
            )
        target = int(round(float(duration_seconds) * self.sample_rate))
        target = max(1, min(target, int(waveforms.shape[-1])))
        if source_durations_seconds is None and target == int(waveforms.shape[-1]):
            return waveforms
        if source_durations_seconds is None:
            valid_samples = torch.full(
                (int(waveforms.shape[0]),),
                int(waveforms.shape[-1]),
                dtype=torch.long,
            )
        else:
            if source_durations_seconds.ndim != 1 or int(
                source_durations_seconds.numel()
            ) != int(waveforms.shape[0]):
                raise ValueError(
                    "source_durations_seconds must be shaped (B,) and match "
                    "the waveform batch."
                )
            valid_samples = torch.round(
                source_durations_seconds.detach().float().cpu() * self.sample_rate
            ).to(dtype=torch.long)
            valid_samples.clamp_(min=1, max=int(waveforms.shape[-1]))

        payload = (
            f"crop:{int(training_seed)}:{int(epoch)}:{int(step)}:"
            f"{int(window_index)}:{self.seed_offset}"
        ).encode("ascii")
        mixed_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(mixed_seed)
        units = torch.rand((int(waveforms.shape[0]),), generator=generator)
        crops = []
        for i, (valid, unit) in enumerate(zip(valid_samples.tolist(), units.tolist())):
            available = min(int(valid), int(waveforms.shape[-1]))
            if available >= target:
                max_start = available - target
                start = min(max_start, int(float(unit) * (max_start + 1)))
                crop = waveforms[i, :, start : start + target]
            else:
                # SpeakerDataset right-pads short recordings.  Re-pad from the
                # declared valid prefix so a random crop can never land wholly
                # in that padding or consume samples beyond the source length.
                crop = waveforms[i, :, :available]
                crop = torch.nn.functional.pad(crop, (0, target - available))
            crops.append(crop)
        return torch.stack(crops, dim=0)
